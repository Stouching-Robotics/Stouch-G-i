"""Held-out open-pose verification from the reference Friston calibration."""

import numpy as np
from scipy.spatial.transform import Rotation

from common.errors import CalibrationQualityError
from .friston_motion import CHAINS, MotionRefit
from common.calibration_workflow import CONTACT_CAPTURES as _CONTACT_IDS, INSTALLATION_STEPS as _INSTALL_IDS

_FINISH_IDS = (*_CONTACT_IDS, "contact_compute", "fist_capture", "verify_open")


def _verify_open_samples(samples, solver, profile):
    refit = MotionRefit(profile, solver.rig.side)
    raw = np.asarray([solver.local_pose(sample)[0] for sample in samples])
    corrected = np.asarray([refit.apply(pose)[0][1:] for pose in raw])

    def joint_errors(values, target):
        target = np.broadcast_to(target, values.shape)
        rotation = Rotation.from_rotvec(values.reshape(-1, 3))
        reference = Rotation.from_rotvec(target.reshape(-1, 3))
        return np.rad2deg((rotation * reference.inv()).magnitude().reshape(values.shape[:2]))

    neutral = np.asarray(profile["neutral_input_pose"])
    target = np.asarray(profile["neutral_target_pose"])
    input_errors = joint_errors(raw[:, 1:], neutral[1:])
    output_errors = joint_errors(corrected, target[1:])
    med_in, med_out = np.median(input_errors, axis=0), np.median(output_errors, axis=0)
    labels = ("食指", "中指", "小指", "无名指", "拇指")
    segments = ("根部", "中节", "末节")

    def details(values, limit):
        return "；".join(f"{labels[i // 3]}{segments[i % 3]} (joint_{i + 1}): {values[i]:.1f}°"
                         for i in np.flatnonzero(values > limit))

    if np.any(med_in > 40.):
        raise CalibrationQualityError(
            f"返回平展输入与原参考相差过大（中位数容差40°）：{details(med_in, 40.)}；\n"
            "先恢复安装标定时相同的手型和手腕朝向，手动重试“返回平展验证”。\n"
            "仍失败：重采“安装-平展”并重新计算安装，然后重做四次对指、握拳和返回平展。",
            retry_steps=(*_INSTALL_IDS, *_FINISH_IDS),
        )
    if np.any(med_out > 60.):
        failed = (med_out[CHAINS - 1] > 60.).any(axis=1)
        retry = tuple(step for i, step in enumerate(_CONTACT_IDS) if failed[-1] or failed[i])
        contact_label = ("食指", "中指", "无名指", "小指")[_CONTACT_IDS.index(retry[0])]
        raise CalibrationQualityError(
            f"平展校正后的手型超差（中位数上限60°）：{details(med_out, 60.)}；\n"
            "输入参考在容差内，先保持同一平展姿势手动重试一次。\n"
            f"仍失败：只重采受影响的对指（{contact_label}等），保留其他采样并重新计算，再返回平展验证。\n"
            "若最初安装平展姿势不正确，应从“安装-平展”重做所有后续步骤。",
            retry_steps=(*retry, "contact_compute", "verify_open"),
        )

    roots = CHAINS[:, 0]
    tangents = np.asarray(profile["root_tangent_axes"])
    normals = solver.rig.anatomical_basis()[roots, :, 1]
    normals = normals - np.sum(normals * tangents, axis=1, keepdims=True) * tangents
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths < 1e-9):
        raise ValueError("Friston root tangent and normal are degenerate")
    normals = normals / lengths
    basis = np.stack((tangents, normals, np.cross(tangents, normals)), axis=-1)
    corrected_rot = Rotation.from_rotvec(corrected[:, roots - 1].reshape(-1, 3))
    actual = corrected_rot.as_matrix().reshape(-1, 5, 3, 3) @ basis
    reference = Rotation.from_rotvec(target[roots]).as_matrix() @ basis
    axes = np.rad2deg(np.arccos(np.clip(np.sum(actual * reference, axis=-2), -1., 1.)))
    return {
        "passed": True, "count": len(samples), "limit_deg": 60, "input_limit_deg": 40, "output_limit_deg": 60,
        "input_joint_p95_deg": np.percentile(input_errors, 95, axis=0).tolist(),
        "output_joint_p95_deg": np.percentile(output_errors, 95, axis=0).tolist(),
        "input_joint_median_deg": med_in.tolist(), "output_joint_median_deg": med_out.tolist(),
        "output_root_tangent_normal_out_p95_deg": np.percentile(axes, 95, axis=0).tolist(),
        "output_root_tangent_normal_out_median_deg": np.median(axes, axis=0).tolist(),
        "input_reference_warning_joint_ids": (np.flatnonzero(med_in > 30.) + 1).tolist(),
        "decision_statistic": "per-joint median across captured frames", "policy": "held-out-return-open-v3",
        "used_for_fit": False, "threshold_basis": "engineering model-space budget, not physical accuracy certification",
    }
