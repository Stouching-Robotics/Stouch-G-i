#!/usr/bin/env python3
"""USB capture backend shared by the Friston calibration GUI and scripted SDK."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

from common.paths import SDK_ROOT

_PROJECT_ROOT = SDK_ROOT
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from loguru import logger

from common.enums import MoCapHandIDEnum
from glove_io.device_registry import (
    list_matching_ports,
    load_device_registry,
    resolve_glove,
)
from algorithm.utils import get_average_rotation
import common.usb_cdc as usb_cdc_mod

# 接收线程断连/重连时刷 INFO，标定后端只需要告警级
logger.remove()
logger.add(sys.stderr, level="WARNING")


class ConfigError(Exception):
    """配置或环境问题。"""


class StreamError(Exception):
    """IMU 数据流不可用。"""


def compute_motion_rms(buffer, n: int = 30, imu_ids=None) -> tuple[float, float]:
    """最近 n 帧的运动 RMS（度）。返回 (整体 RMS, 最差单 IMU RMS)。

    imu_ids 限定检查范围；根标定只检查手腕 (0,)，后续阶段检查全手。
    只用 peek() 读，不推进任何 pull 游标——与 sample_imu_quat_from_msg_buffer
    的采集逻辑互不干扰。
    """
    counter = buffer.get_counter()
    frames = []
    for k in range(min(n, counter)):
        frame, _, err = buffer.peek(counter - 1 - k)
        if err is None and frame is not None and getattr(frame, "imu_rotation", None) is not None:
            frames.append(frame)
    if len(frames) < 5:
        return 0.0, 0.0
    rots = [f.imu_rotation for f in frames]
    mean = get_average_rotation(rots)
    dev_deg = np.degrees(np.stack([(r * mean.inv()).magnitude() for r in rots]))
    rms_per_imu = np.sqrt(np.mean(dev_deg ** 2, axis=0))
    if imu_ids is not None:
        rms_per_imu = rms_per_imu[np.asarray(tuple(imu_ids), dtype=int)]
    return float(np.mean(rms_per_imu)), float(np.max(rms_per_imu))


def default_out_path(side: str = "right") -> Path:
    """默认输出路径（calibration/ 目录，与 imu_calibration.json 同目录）。
    同秒重复运行时自动加 -1/-2… 后缀，绝不覆盖旧文件。"""
    base = _PROJECT_ROOT / "calibration"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "imu_calibration_left" if side == "left" else "imu_calibration_right"
    candidate = base / f"{prefix}_{ts}.json"
    if not candidate.exists():
        return candidate
    for k in range(1, 1000):
        alt = base / f"{prefix}_{ts}-{k}.json"
        if not alt.exists():
            return alt
    raise RuntimeError(
        f"Could not allocate a unique calibration output name in {base}")


class ImuCalibrationCLI:
    def __init__(self, args):
        self.args = args
        self.hand = None
        self.hand_config = None
        self.config = None
        self.bound_serial = ""

    @staticmethod
    def _serial_for_port(serial_port: str | None) -> str:
        """Reverse-lookup the USB serial for an explicit COM port, else ""."""
        wanted = str(serial_port or "").strip()
        if not wanted:
            return ""
        for port in list_matching_ports():
            if str(port.device) == wanted:
                return str(port.serial_number or "").upper()
        return ""

    # ---------------------------------------------------------------- boot
    def boot(self) -> None:
        from common.datamodels import MoCapConfig
        from algorithm.friston_calibration import FristonCalibrationHand

        config, err = MoCapConfig.read_from_disk(str(self.args.config))
        if err is not None:
            raise ConfigError(f"读取配置失败: {err}")
        hand_side = MoCapHandIDEnum(self.args.side)
        hc = (config.left_hand_config if hand_side == MoCapHandIDEnum.LEFT
              else config.right_hand_config)
        if hc is None:
            raise ConfigError(f"config.json 没有 {hand_side.value}_hand_config")
        if hc.transport != "usb_cdc":
            raise ConfigError(
                f"{hand_side.value}手输入不是 STM32 USB CDC，请先切换为 USB 模式")
        if self.args.serial_port:
            hc.serial_port = self.args.serial_port
            self.bound_serial = self._serial_for_port(hc.serial_port)
        else:
            hc.serial_port, self.bound_serial = resolve_glove(
                hand_side.value, self.args.registry,
                allow_single_unbound=True)
        if self.args.fps:
            hc.fps = self.args.fps
        self.hand_config = hc
        self.config = config

        self.hand = FristonCalibrationHand(hc, hand_side.value)
        self.hand.set_calibration_workflow(
            getattr(self.args, "calibration_workflow", "simplified"),
            include_return_open=getattr(self.args, "include_return_open", False))
        self.hand.start()

        # 阶段3：16 路全部在线
        if not self.wait_stream_healthy(self.args.startup_timeout):
            raise StreamError(self._port_help_text())

        # 阶段4：BNO055 冷启动漂移稳定。掉电/重连后 gyro-bias 需要数秒收敛，
        # 且固件不上报 CALIB_STAT（host 无从得知）；采集标定前先等手套静止
        # 稳定，避免把冷启动漂移采进标定（与 live 的预热门控同一策略）。
        # GUI 连接阶段不需要这一步：GUI 的自动采集在每次采集前都用更严的
        # RMS 阈值做静止门控，冷启动漂移同样会被拦住。连接因此只做串口 +
        # 16 路 IMU 就绪检查，保持快速。
        if not getattr(self.args, "skip_settle", False):
            self.wait_stream_settle()

    def wait_stream_settle(self, rms_deg: float = 1.0, duration_s: float = 2.0,
                           timeout_s: float = 30.0) -> None:
        """Wait until the 16-IMU stream has been continuously statically settled."""
        deadline = time.monotonic() + float(timeout_s)
        settle_start = None
        while time.monotonic() < deadline:
            rms, _ = compute_motion_rms(self.hand.buffer, 30)
            if rms < rms_deg:
                if settle_start is None:
                    settle_start = time.monotonic()
                if time.monotonic() - settle_start >= duration_s:
                    return
            else:
                settle_start = None
            time.sleep(0.1)
        logger.warning(
            f"手套在 {timeout_s:.0f}s 内未完全静止（运动 RMS 未稳定到 "
            f"{rms_deg}°）；直接继续。若标定结果异常，请把手套放平静止数秒再重试。")

    def _port_help_text(self) -> str:
        ports = usb_cdc_mod.list_serial_ports()
        if not ports:
            head = "未检测到任何串口设备。"
        else:
            head = "未找到 / 未收到 STM32 IMU 数据。当前串口:"
            for dev, desc, vid, pid in ports:
                vid_s = f"{vid:#06x}" if vid is not None else "?"
                pid_s = f"{pid:#06x}" if pid is not None else "?"
                head += f"\n  {dev}  {desc}  (VID={vid_s} PID={pid_s})"
        head += "\n请确认手套已插入 USB、串口未被其他上位机占用，然后重试。"
        return head

    def wait_stream_healthy(self, timeout: float, min_frames: int = 5) -> bool:
        baseline = self.hand.buffer.get_counter()
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if (not self.hand.missing_imu_ids
                    and self.hand.buffer.get_counter() > baseline + min_frames):
                return True
            time.sleep(0.25)
        return False

    # --------------------------------------------------------------- output
    def build_payload(self) -> dict:
        serial = self.bound_serial or load_device_registry(self.args.registry)[self.args.side]["usb_serial"]
        return self.hand.build_payload(serial)

    def shutdown(self) -> None:
        if self.hand is not None:
            self.hand.shutdown()
