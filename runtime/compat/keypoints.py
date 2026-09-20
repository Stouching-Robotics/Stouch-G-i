"""runtime.compat.keypoints — 公开的 21 个手部关键点接口（纯源码，不加密）。

这是明文程序与**加密的 FK 解算核心**之间的唯一闸口：
  - :func:`create_solver` 构造 16 IMU → 21 关键点的解算器（内部调用加密核心）；
  - :class:`KeypointSolver` 对外公开与解算器一致的接口（``neutral_joints`` /
    ``solve``），消费方无需感知核心是否加密。

本模块自身不包含任何解算算法。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from common.paths import SDK_ROOT
from common.resources import hand_geometry_path

# 运行期几何路径（数据非算法）。开发树解析到 $PROJ，frozen 单文件解析到 _MEIPASS
# （datas 已把 assets/hand_geometry 放到 _MEIPASS/assets/hand_geometry）。
_PROJECT_ROOT = SDK_ROOT
if getattr(sys, "frozen", False):
    _PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", _PROJECT_ROOT))
DEFAULT_GEOMETRY_PATH = (
    hand_geometry_path("hand_measured_runtime_v1.json"))


@dataclass
class KeypointsFrame:
    """一帧 21 个手部关键点（世界坐标，米）。

    ``joints`` 为 (21, 3) float32；``status`` 为解算状态（kinematics 类型、
    geometry_profile、contact、safety_intervened 等）。
    """

    joints: np.ndarray  # (21, 3) float32
    status: dict[str, Any]


class KeypointSolver:
    """21 点解算器门面。接口与底层解算器逐字对齐，消费方零改动。"""

    def __init__(self, solver: Any, backend: dict[str, Any]):
        self._solver = solver
        self.backend = dict(backend)

    # -- 与底层求解器一致的公开接口（Glove21SolverLite / UncalibratedGlove21Solver）
    def neutral_joints(self) -> np.ndarray:
        """中性姿态 21 点（(21, 3) float32）。"""
        return self._solver.neutral_joints()

    @property
    def neutral_imu_quaternions(self) -> np.ndarray:
        """中性姿态 16 路 IMU 四元数（(16, 4)）。"""
        return self._solver.neutral_imu_quaternions

    @property
    def side(self) -> str:
        return self._solver.side

    def solve(
        self,
        imu_rotation,
        uninitialized_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """输入 scipy Rotation (16,)，输出 ``(joints(21,3) float32, status dict)``。"""
        return self._solver.solve(
            imu_rotation, uninitialized_mask=uninitialized_mask)

    # -- SDK 便捷：直接吃 ImuFrame -----------------------------------------
    def solve_frame(self, imu_frame) -> KeypointsFrame:
        """输入 :class:`runtime.compat.imu.ImuFrame`，输出 :class:`KeypointsFrame`。"""
        rotation = imu_frame.to_rotation()
        if imu_frame.sensor_age_s is not None and imu_frame.sensor_age_s.size:
            mask = ~np.isfinite(imu_frame.sensor_age_s)
        else:
            mask = ~imu_frame.valid_mask
        joints, status = self.solve(rotation, uninitialized_mask=mask)
        return KeypointsFrame(joints=joints, status=status)


def create_solver(
    hand_config,
    side: str,
    geometry_path,
    allow_uncalibrated: bool = False,
) -> tuple[KeypointSolver, dict[str, Any]]:
    """构造 21 点解算器（内部调用加密的 FK 核心）。

    ``hand_config`` 需含 ``param_inst_calib``（标定数据）。标定不可用且
    ``allow_uncalibrated=True`` 时回退到无标定诊断预览。
    返回 ``(KeypointSolver, backend_dict)``。
    """
    if allow_uncalibrated:
        return uncalibrated_solver(side, geometry_path)
    from algorithm.runtime_backend import create_runtime_solver
    solver, backend = create_runtime_solver(hand_config, side, geometry_path)
    return KeypointSolver(solver, backend), backend


def uncalibrated_solver(
    side: str,
    geometry_path,
) -> tuple[KeypointSolver, dict[str, Any]]:
    """无标定诊断回退：首帧捕获 base 的 MANO 中性预览，仅用于诊断。"""
    from algorithm.lite.uncalibrated import UncalibratedGlove21Solver
    solver = UncalibratedGlove21Solver(side)
    backend = {
        "backend": "uncalibrated_mano",
        "local_only": True,
        "pinky_extra_length_mm": 2.0,
        "hand_model_required": True,
    }
    return KeypointSolver(solver, backend), backend


def backend_metadata(backend: dict[str, Any], geometry_path) -> dict[str, Any]:
    """会话元数据用的后端描述（与 pc.runtime_backend 同签名）。"""
    from algorithm.runtime_backend import backend_metadata as describe
    return describe(backend, geometry_path)
