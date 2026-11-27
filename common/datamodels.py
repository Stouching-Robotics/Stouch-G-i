import json
import os
import tempfile
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from common.enums import MoCapCalibrateInstallationType, MoCapCalibrateShapeType, MoCapHandIDEnum, \
    MoCapCalibrateRootType, MoCapCalibrateWristType, MoCapCalibrateShapeKeyPointType
from common.usb_cdc import DEFAULT_CHANNEL_TO_HAND, validate_channel_to_hand


class MoCapHandConfig(BaseModel):
    transport: str = "websocket"     # "websocket" (ESP) or "usb_cdc" (STM32)
    ws_host: str | None = None       # WebSocket 主机地址 (如 "192.168.1.100")
    ws_port: int = 18890
    serial_port: str | None = None   # None = 按 VID/PID 自动查找 STM32 CDC
    usb_vid: int = 0x0483
    usb_pid: int = 0x5740
    channel_to_hand: list[int] = Field(
        default_factory=lambda: list(DEFAULT_CHANNEL_TO_HAND))
    fps: int | None = None
    hardware_id: str = "esp-glove-right"
    # 与校准相关的参数
    param_inst_calib: dict[str, Any] | None = None
    param_shape_calib: dict[str, Any] | list[float] | None = None
    param_wrist_calib: dict[str, Any] | None = None

    @model_validator(mode='after')
    def verify(self):
        self.transport = self.transport.strip().lower()
        if self.transport not in {"websocket", "usb_cdc"}:
            raise ValueError("transport must be 'websocket' or 'usb_cdc'")
        self.channel_to_hand = list(validate_channel_to_hand(self.channel_to_hand))
        if self.enabled and not self.fps:
            raise ValueError("fps is required for an enabled hand transport")
        return self

    @property
    def enabled(self) -> bool:
        if self.transport == "usb_cdc":
            return True
        return bool(self.ws_host)

    @property
    def calibration_hardware_id(self) -> str:
        """Bind saved calibration to both the board and channel permutation."""

        if self.transport != "usb_cdc":
            return self.hardware_id
        mapping = "-".join(str(value) for value in self.channel_to_hand)
        return f"{self.hardware_id}|physical-to-hand:{mapping}"


class MoCapNokovConfig(BaseModel):
    server_ip: str | None = None
    fps: int | None = None
    # TODO: 添加 STL 映射
    rigidbody_id_mapping: dict[MoCapHandIDEnum, str | int] | None = Field(
        default_factory=lambda: {MoCapHandIDEnum.LEFT: 'left', MoCapHandIDEnum.RIGHT: 'right', MoCapHandIDEnum.SPIKE: 'spike'})
    record_markers: bool = False

    @model_validator(mode='after')
    def verify(self):
        if self.server_ip:
            if not self.fps:
                raise ValueError("fps is required if server_ip is provided")
        assert MoCapHandIDEnum.SPIKE in self.rigidbody_id_mapping.keys(), "SPIKE rigid body id is required"
        return self

    @property
    def enabled(self) -> bool:
        return bool(self.server_ip)

class MoCapNokovGeneralConfig(BaseModel):
    server_ip: str | None = None
    fps: int | None = None
    rigidbody_id_mapping: dict[str, str] | None = Field(default_factory=lambda: {})
    record_markers: bool = False

    @model_validator(mode='after')
    def verify(self):
        if self.server_ip:
            if not self.fps:
                raise ValueError("fps is required if server_ip is provided")
        return self

    @property
    def enabled(self) -> bool:
        return bool(self.server_ip)


class MoCapViveConfig(BaseModel):
    tracker_serial_mapping: dict[str, str] = Field(
        default_factory=lambda: {}
    )
    fps: int = 60

    @model_validator(mode='after')
    def verify(self):
        if self.tracker_serial_mapping:
            if not self.fps:
                raise ValueError("fps is required if tracker_serial_mapping is provided")
        return self

    @property
    def enabled(self) -> bool:
        return bool(self.tracker_serial_mapping)


class MoCapShadowConfig(BaseModel):
    shadow_offset: list[float] | None = Field(default_factory=lambda : [0.0] * 24)
    shadow_amplitude: list[float] | None = Field(default_factory=lambda : [1.0] * 24)
    endpoint: str | None = None


class MoCapPressureSensorConfig(BaseModel):
    """ESP32 WebSocket 触觉压力传感器配置。"""
    ws_host: str | None = None
    ws_port: int = 18891
    fps: int = 50
    enabled: bool = False
    # 着色增益: 满红对应多大压力（ADC 码）。越小越灵敏。
    color_vmax: float | None = None
    # 降噪链路参数，见 pc/tactile_preprocess.py（与 hand 界面共用同一份实现）。
    # None = 用代码里的默认值。
    base_gate: float | None = None            # 固定噪声门限，等价 hand 的 --gate
    dynamic_noise_ratio: float | None = None  # 动态门限 = 峰值 × 此值
    temporal_smooth: float | None = None      # 时域平滑系数
    spatial_filter: bool | None = None        # 孤立点消除
    crosstalk_ratio: float | None = None      # 区域内串扰抑制，见 pc/hand_mapping.py

    @property
    def is_enabled(self) -> bool:
        return self.enabled and bool(self.ws_host)


class MoCapConfig(BaseModel):
    # 通用配置
    _loaded: bool = False
    ui_lang: str = "zh"          # 界面语言 "zh" / "en"，见 modules/i18n.py
    use_rfu: bool = False
    use_cuda: bool = True
    debug: bool = False
    show_tracker: bool = False
    enable_control: bool = False
    enable_api: bool = False
    api_port: int = 8000

    # 子配置
    wrist_tracker_cfg: MoCapNokovConfig | MoCapNokovGeneralConfig | MoCapViveConfig | None = None
    left_hand_config: MoCapHandConfig | None = None
    right_hand_config: MoCapHandConfig | None = None
    shadow_config: MoCapShadowConfig = Field(default_factory=MoCapShadowConfig)
    pressure_sensor_cfg: MoCapPressureSensorConfig | None = None

    # 数据存储
    data_store: str = './'

    # 来源
    local_path: str = './config.json'

    @classmethod
    def read_from_disk(cls, local_path: str) -> tuple[Any, Exception | None]:
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8-sig') as f:
                    config = json.load(f)
                res = MoCapConfig(**config)
                res.local_path = local_path
                logger.info(f"config file loaded from {local_path}")
                return res, None
            except Exception as e:
                return None, e
        else:
            logger.warning("config file not found, creating new one")
            config = cls(
                wrist_tracker_cfg=MoCapNokovGeneralConfig(),
                left_hand_config=MoCapHandConfig(),
                right_hand_config=MoCapHandConfig(),
                shadow_config=MoCapShadowConfig(),
                local_path=local_path,
            )
            config.save_to_disk(local_path)
            return config, None

    def save_to_disk(self, path: str = None) -> Exception | None:
        if path is None:
            path = self.local_path
        path = os.fspath(path)
        directory = os.path.dirname(os.path.abspath(path))
        prefix = f".{os.path.basename(path)}."
        fd, temporary_path = tempfile.mkstemp(
            prefix=prefix, suffix=".tmp", dir=directory, text=True)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.model_dump(), f, indent=4)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(path):
                os.chmod(temporary_path, os.stat(path).st_mode & 0o777)
            os.replace(temporary_path, path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise
        logger.info(f"config dumped to {path}")
        return None

class MoCapGUIEvents(BaseModel):
    # 通用控制标志
    mocap_exit_requested: bool = False
    mocap_save_config_triggered: bool = False

    # 与根部校准相关的标志
    mocap_calibrate_root_triggered: bool = False
    mocap_calibrate_root_type: MoCapCalibrateRootType = MoCapCalibrateRootType.HORIZONTAL
    mocap_calibrate_root_target: MoCapHandIDEnum = MoCapHandIDEnum.UNKNOWN

    # 与安装校准相关的标志
    mocap_calibrate_installation_triggered: bool = False
    mocap_calibrate_installation_type: MoCapCalibrateInstallationType = MoCapCalibrateInstallationType.POSE_0
    mocap_calibrate_installation_target: MoCapHandIDEnum = MoCapHandIDEnum.UNKNOWN

    # 与形状校准相关的标志
    mocap_calibrate_shape_triggered: bool = False
    mocap_calibrate_shape_type: MoCapCalibrateShapeType = MoCapCalibrateShapeType.INDEX
    mocap_calibrate_shape_target: MoCapHandIDEnum = MoCapHandIDEnum.UNKNOWN
    mocap_calibrate_shape_keypoint_triggered: bool = False
    mocap_calibrate_shape_keypoint_type: MoCapCalibrateShapeKeyPointType = None
    mocap_calibrate_shape_keypoint_target: MoCapHandIDEnum = MoCapHandIDEnum.UNKNOWN

    # 与腕部校准相关的标志
    mocap_calibrate_wrist_triggered: bool = False
    mocap_calibrate_wrist_type: MoCapCalibrateWristType = MoCapCalibrateWristType.THUMB
    mocap_calibrate_wrist_target: MoCapHandIDEnum = MoCapHandIDEnum.UNKNOWN

    # 与 rfu.ctrl 相关的标志
    mocap_rfu_ctrl_launch_triggered: bool = False
    mocap_rfu_ctrl_launch_target: MoCapHandIDEnum = MoCapHandIDEnum.UNKNOWN
    mocap_rfu_ctrl_termination_triggered: bool = False

    # 与 rfu.retarget 相关的标志
    mocap_rfu_retarget_open_triggered: bool = False
    mocap_rfu_retarget_open_target: MoCapHandIDEnum = MoCapHandIDEnum.UNKNOWN
    mocap_rfu_retarget_next_triggered: bool = False
    mocap_rfu_retarget_save_triggered: bool = False
    mocap_rfu_retarget_close_triggered: bool = False

    # 与录制相关的标志
    mocap_start_recording_triggered: bool = False
    mocap_stop_recording_triggered: bool = False

    def __hash__(self):
        return hash(tuple(self.model_dump().values()))

class ImuMsg:
    sys_ticks: int
    imu_rotation: R
    seq: int

    def __init__(self, sys_ticks: int, imu_rotation: R, seq=0,
                 valid_mask: np.ndarray | list[bool] | None = None,
                 sensor_age_s: np.ndarray | list[float] | None = None,
                 rejected_mask: np.ndarray | list[bool] | None = None,
                 calibration_status: np.ndarray | list[int] | None = None):
        """
        :param sys_ticks: 系统滴答数，int 类型
        :param imu_rotation: 旋转对象，形状为 (16,)
        """
        self.sys_ticks = sys_ticks
        self.imu_rotation = imu_rotation
        self.seq = seq
        self.valid_mask = self._coerce_vector(
            valid_mask, np.ones(16, dtype=bool), bool, "valid_mask")
        self.sensor_age_s = self._coerce_vector(
            sensor_age_s, np.zeros(16, dtype=float), float, "sensor_age_s")
        self.rejected_mask = self._coerce_vector(
            rejected_mask, np.zeros(16, dtype=bool), bool, "rejected_mask")
        self.calibration_status = self._coerce_vector(
            calibration_status, np.full(16, 0xFF, dtype=np.uint8), np.uint8,
            "calibration_status")
        if len(self.imu_rotation) != 16:
            raise ValueError(f"imu_rotation must contain 16 rotations, got {len(self.imu_rotation)}")

    @staticmethod
    def _coerce_vector(value, default, dtype, name):
        array = default if value is None else np.asarray(value, dtype=dtype)
        if array.shape != (16,):
            raise ValueError(f"{name} must have shape (16,), got {array.shape}")
        return array.copy()

    def calibration_ready(self, max_sensor_age_s: float = 0.10) -> bool:
        # 0xFF is the backward-compatible "not supplied" value used by tests
        # and non-hardware producers. Real 2.1 firmware always sends 0..255.
        supplied = self.calibration_status != 0xFF
        status = self.calibration_status.astype(np.uint8)
        bno_ready = (
            (((status >> 6) & 3) >= 1)
            & (((status >> 4) & 3) >= 2)
            & (((status >> 2) & 3) >= 1)
            & ((status & 3) >= 1)
        )
        # 固件每 5s 广播读一次 BNO055 CALIB_STAT，实测 16 路恒为 0x00 ——
        # 固件没有实际上报该字段(读超时清零/融合模式不上报)。此时"未知≠未校准"，
        # 与 receiver 的容忍逻辑一致，不能据此逐帧拒掉标定采集。
        # 只有至少一路 IMU 上报了非零状态时，才对"已上报但未达标"的通道严格把关。
        if np.any(supplied & (status != 0)):
            bno_gate = np.all((~supplied) | bno_ready)
        else:
            bno_gate = True
        return bool(
            # valid_mask means "received a fresh UART response in this exact
            # wire frame".  The receiver keeps the last accepted quaternion
            # for a missed response, so freshness is governed by sensor_age_s
            # instead.  Requiring all 16 bits here threw away roughly 40% of
            # otherwise safe static-calibration frames on the real glove.
            np.all(np.isfinite(self.sensor_age_s))
            and np.all(self.sensor_age_s <= max_sensor_age_s)
            and not np.any(self.rejected_mask)
            and bno_gate
        )

    @property
    def bno_not_ready_ids(self) -> list[int]:
        status = self.calibration_status.astype(np.uint8)
        supplied = status != 0xFF
        # 与 calibration_ready 的容忍逻辑一致: 没有任何 IMU 上报非零标定状态
        # 时(固件恒 0x00/未提供), 未知≠未校准, 不报任何通道为未就绪。
        if not np.any(supplied & (status != 0)):
            return []
        ready = (
            (((status >> 6) & 3) >= 1)
            & (((status >> 4) & 3) >= 2)
            & (((status >> 2) & 3) >= 1)
            & ((status & 3) >= 1)
        )
        return [int(i) for i in np.flatnonzero(supplied & ~ready)]

    def __repr__(self):
        return f"<ImuMsg: {self.seq}>"

class  WristTrackerMsg:
    timestamp_us: int
    pos: np.ndarray
    rot: R
    markers: np.ndarray
    seq: int

    def __init__(self, timestamp_us, pos, rot, markers, seq=0):
        """
        :param timestamp_us: 时间戳（微秒），int 类型
        :param pos: 位置数组，形状为 (3,)
        :param rot: 旋转对象，形状为 (1,)
        """
        self.timestamp_us = timestamp_us
        self.pos = pos
        self.rot = rot
        self.markers = markers
        self.seq = seq

    def to_dict(self):
        return {
            'timestamp_us': self.timestamp_us,
            'pos': self.pos.tolist(),
            'rot': self.rot.as_quat().tolist(),
            'markers': self.markers.tolist() if self.markers is not None else None,
            'seq': self.seq,
        }

    def __repr__(self):
        return f"<WristTrackerMsg pos={self.pos}, rot={self.rot.as_quat().tolist()}, seq={self.seq}, markers={self.markers}>"


class PackedWristTrackerMsg:
    timestamp_us: int
    tracked_objects: dict[str, WristTrackerMsg] | None
    seq: int

    def __init__(self, timestamp_us, seq=0, **tracked_objects):
        self.timestamp_us = timestamp_us
        self.tracked_objects = tracked_objects
        self.seq = seq

    def to_dict(self):
        return {
            'timestamp_us': self.timestamp_us,
            'tracked_objects': {k: v.to_dict() if v is not None else None for k, v in self.tracked_objects.items()},
            'seq': self.seq,
        }
