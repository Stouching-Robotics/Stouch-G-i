"""runtime.compat.imu — 公开的 IMU 原始数据接口（纯源码，不加密）。

``ImuStream`` 在后台起一个 STM32 USB CDC 接收线程（复用
``glove_lite.usb_receiver.hand_usb_receiver``），对外提供两类视图：
  - ``frames()``    → 已按 ``channel_to_hand`` 重映射、带有效/拒绝掩码的 ``ImuFrame``；
  - ``raw_frames()``→ 物理通道原始四元数逐帧值 ``PhysicalFrame``（"IMU 原始数据"语义，
                      不重映射、不滤波、不轴校正），仅构造时 ``publish_physical=True`` 可用。

同串口的触觉数据会推入 ``pressure_buffer``，可用 ``tactile_frames()`` 消费，
或交给 :class:`runtime.compat.tactile.TactileStream` 复用（避免同一串口双线程）。
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np

from common.usb_cdc import (
    DEFAULT_CHANNEL_TO_HAND,
    STM32_USB_PID,
    STM32_USB_VID,
)
from common.ring_buffer import RingBuffer
from algorithm.lite.usb_receiver import hand_usb_receiver

_DEFAULT_BUFFER_SIZE = 256
_DEFAULT_PRESSURE_BUFFER_SIZE = 2048
_POLL_INTERVAL_S = 0.005


@dataclass
class ImuFrame:
    """一帧 16 路 IMU 四元数（已重映射 + 有效判定）。

    ``quaternions_xyzw`` 为 (16, 4) float64，列序为 x/y/z/w（scipy 约定，
    与 ``scipy.spatial.transform.Rotation.from_quat`` 一致）。
    """

    sys_ticks: int
    seq: int
    quaternions_xyzw: np.ndarray  # (16, 4) float64
    valid_mask: np.ndarray  # (16,) bool
    sensor_age_s: np.ndarray  # (16,) float
    rejected_mask: np.ndarray  # (16,) bool
    calibration_status: np.ndarray  # (16,) uint8

    @classmethod
    def from_imu_msg(cls, msg) -> "ImuFrame":
        return cls(
            sys_ticks=int(msg.sys_ticks),
            seq=int(msg.seq),
            quaternions_xyzw=msg.imu_rotation.as_quat().astype(np.float64),
            valid_mask=np.asarray(msg.valid_mask, dtype=bool).copy(),
            sensor_age_s=np.asarray(msg.sensor_age_s, dtype=float).copy(),
            rejected_mask=np.asarray(msg.rejected_mask, dtype=bool).copy(),
            calibration_status=np.asarray(
                msg.calibration_status, dtype=np.uint8).copy(),
        )

    def to_rotation(self):
        """→ scipy Rotation (16,)。"""
        from scipy.spatial.transform import Rotation as R
        return R.from_quat(self.quaternions_xyzw)


@dataclass
class PhysicalFrame:
    """一帧 16 路物理通道原始四元数（不重映射 / 不滤波 / 不轴校正）。

    ``present_mask`` 为固件帧 flags 位掩码：bit i = 物理通道 i 本帧有效。
    """

    sys_ticks: int
    seq: int
    quaternions_xyzw: np.ndarray  # (16, 4) float64，x/y/z/w
    present_mask: np.ndarray  # (16,) bool

    def to_rotation(self):
        from scipy.spatial.transform import Rotation as R
        return R.from_quat(self.quaternions_xyzw)


class ImuStream:
    """USB CDC → 16 路 IMU 四元数流（后台接收线程 + 环形缓冲）。"""

    def __init__(
        self,
        serial_port: str | None = None,
        usb_vid: int = STM32_USB_VID,
        usb_pid: int = STM32_USB_PID,
        channel_to_hand=None,
        status_callback: Callable[[list[str]], None] | None = None,
        publish_physical: bool = False,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        pressure_buffer_size: int = _DEFAULT_PRESSURE_BUFFER_SIZE,
        poll_interval_s: float = _POLL_INTERVAL_S,
    ):
        self._serial_port = serial_port
        self._usb_vid = int(usb_vid)
        self._usb_pid = int(usb_pid)
        self._channel_to_hand = (
            list(channel_to_hand) if channel_to_hand
            else list(DEFAULT_CHANNEL_TO_HAND))
        self._publish_physical = bool(publish_physical)
        self._poll_interval_s = poll_interval_s

        self._kill = threading.Event()
        self._buffer = RingBuffer(buffer_size)
        self._pressure_frame_buffer = RingBuffer(pressure_buffer_size)
        self._physical_buffer = (
            RingBuffer(buffer_size) if publish_physical else None)
        self._thread: threading.Thread | None = None
        self._missing: list[str] = []

        def _status(values: list[str]) -> None:
            self._missing = list(values)
            if status_callback is not None:
                status_callback(values)

        self._status_callback = _status

    # -- 生命周期 ----------------------------------------------------------
    def start(self) -> "ImuStream":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._kill.clear()
        self._thread = threading.Thread(
            target=hand_usb_receiver,
            args=(
                self._kill,
                self._serial_port,
                self._usb_vid,
                self._usb_pid,
                self._channel_to_hand,
                self._buffer,
                self._status_callback,
                None,                       # pressure_buffer（samples 旧路径）
                self._physical_buffer,      # physical_buffer
                self._pressure_frame_buffer,  # pressure_frame_buffer（完整帧）
            ),
            name="glove-sdk-imu", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 3.0) -> None:
        self._kill.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

    def __enter__(self) -> "ImuStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- 状态 --------------------------------------------------------------
    @property
    def missing(self) -> list[str]:
        """当前缺失/无效的 IMU 通道标签（最近一次接收线程上报）。"""
        return list(self._missing)

    @property
    def pressure_frame_buffer(self) -> RingBuffer:
        """接收线程写入的触觉完整帧缓冲（UsbAdcMatrixFrame，供 TactileStream 复用）。"""
        return self._pressure_frame_buffer

    # -- 数据流 ------------------------------------------------------------
    def frames(self) -> Iterator[ImuFrame]:
        """阻塞产出已重映射的 ``ImuFrame``；``stop()`` 后正常结束。"""
        while not self._kill.is_set():
            items, error = self._buffer.pull("glove-sdk-imu")
            if error is not None:
                raise RuntimeError(f"IMU 缓冲错误: {error}")
            for item in items:
                yield ImuFrame.from_imu_msg(item)
            time.sleep(self._poll_interval_s)

    def raw_frames(self) -> Iterator[PhysicalFrame]:
        """阻塞产出物理通道原始 ``PhysicalFrame``（需 ``publish_physical=True``）。"""
        if self._physical_buffer is None:
            raise RuntimeError(
                "ImuStream(publish_physical=True) 未开启，无法读取 raw_frames()")
        while not self._kill.is_set():
            items, error = self._physical_buffer.pull("glove-sdk-imu-raw")
            if error is not None:
                raise RuntimeError(f"IMU 原始缓冲错误: {error}")
            for quats, mask, seq, ticks in items:
                yield PhysicalFrame(
                    sys_ticks=int(ticks),
                    seq=int(seq),
                    quaternions_xyzw=np.asarray(quats, dtype=np.float64),
                    present_mask=np.asarray(mask, dtype=bool),
                )
            time.sleep(self._poll_interval_s)

    def tactile_frames(self):
        """便捷方法：消费同串口的 16×16 触觉帧（等价 TactileStream(imu=self)）。"""
        from runtime.compat.tactile import TactileStream
        return TactileStream(imu=self).frames()
