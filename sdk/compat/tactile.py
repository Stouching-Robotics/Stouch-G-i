"""sdk.compat.tactile — 公开的触觉传感器数据接口（纯源码，不加密）。

``TactileStream`` 产出 16×16 触觉压力矩阵帧。两种用法：
  - 复用已有的 :class:`sdk.compat.imu.ImuStream`（``TactileStream(imu=stream)``，
    与 IMU 共享同一接收线程，避免同一串口双线程）；
  - 独立打开串口（``TactileStream(serial_port=...)``，只消费触觉）。

触觉原始数据为固件逐帧下发的 16×16 ADC 采样值（float32，无需解算）。
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Iterator, Optional

import numpy as np

from common.usb_cdc import (
    DEFAULT_CHANNEL_TO_HAND,
    STM32_USB_PID,
    STM32_USB_VID,
)
from common.ring_buffer import RingBuffer
from algorithm.lite.usb_receiver import hand_usb_receiver

_POLL_INTERVAL_S = 0.005


@dataclass
class TactileFrame:
    """一帧 16×16 触觉压力矩阵。

    ``samples`` 为 (16, 16) float32，原始 ADC 采样值（0..4095），未做任何
    噪声/基线处理；如需预处理请用 ``pc.tactile_preprocess.TactilePreprocessor``。
    """

    sequence: int
    scan_time_us: int
    samples: np.ndarray  # (16, 16) float32

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float32)


class TactileStream:
    """USB CDC → 16×16 触觉压力矩阵流。"""

    def __init__(
        self,
        imu: Optional["object"] = None,
        *,
        serial_port: str | None = None,
        usb_vid: int = STM32_USB_VID,
        usb_pid: int = STM32_USB_PID,
        channel_to_hand=None,
        buffer_size: int = 2048,
        poll_interval_s: float = _POLL_INTERVAL_S,
    ):
        self._imu = imu
        self._poll_interval_s = poll_interval_s

        if imu is not None:
            # 复用 ImuStream 的接收线程与完整帧缓冲（避免同串口双线程）。
            self._shared = True
            self._frame_buffer = imu.pressure_frame_buffer
            self._kill = imu._kill
            self._thread: threading.Thread | None = None
        else:
            self._shared = False
            self._kill = threading.Event()
            self._frame_buffer = RingBuffer(buffer_size)
            self._thread = threading.Thread(
                target=hand_usb_receiver,
                args=(
                    self._kill,
                    serial_port,
                    int(usb_vid),
                    int(usb_pid),
                    list(channel_to_hand) if channel_to_hand
                    else list(DEFAULT_CHANNEL_TO_HAND),
                    RingBuffer(16),          # IMU 缓冲（本流不消费，仅占位）
                    None,                    # status_callback
                    None,                    # pressure_buffer（samples 旧路径）
                    None,                    # physical_buffer
                    self._frame_buffer,      # pressure_frame_buffer（完整帧）
                ),
                name="glove-sdk-tactile", daemon=True)
            self._started = False

    # -- 生命周期 ----------------------------------------------------------
    def start(self) -> "TactileStream":
        if self._shared:
            if self._imu._thread is None or not self._imu._thread.is_alive():
                self._imu.start()
        elif not self._started:
            self._kill.clear()
            self._thread.start()
            self._started = True
        return self

    def stop(self, timeout_s: float = 3.0) -> None:
        if self._shared:
            self._imu.stop(timeout_s=timeout_s)
        else:
            self._kill.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=timeout_s)

    def __enter__(self) -> "TactileStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- 数据流 ------------------------------------------------------------
    def frames(self) -> Iterator[TactileFrame]:
        """阻塞产出 16×16 触觉帧；``stop()`` 后正常结束。"""
        while not self._kill.is_set():
            items, error = self._frame_buffer.pull("glove-sdk-tactile")
            if error is not None:
                raise RuntimeError(f"触觉缓冲错误: {error}")
            for matrix_frame in items:
                yield TactileFrame(
                    sequence=int(matrix_frame.sequence),
                    scan_time_us=int(matrix_frame.scan_time_us),
                    samples=np.asarray(
                        matrix_frame.samples, dtype=np.float32),
                )
            time.sleep(self._poll_interval_s)
