"""Wrist root frame from a measured pitch axis and world-vertical prior."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from algorithm.utils import correct_imu_quaternion_axes

ROOT_METHOD_FULL = "multi-axis-v1"
ROOT_METHOD_SIMPLIFIED = "level-up-world-vertical-v1"
PRIOR_SOURCE = "receiver-world-vertical-assumption"
PRIOR_COORDINATE_VERSION = "correct-imu-quaternion-axes-v1"


def processed_world_up():
    raw = R.from_euler("z", 90, degrees=True).as_quat()
    axis = R.from_quat(correct_imu_quaternion_axes(raw)).as_rotvec()
    return axis / np.linalg.norm(axis)


def solve_world_vertical_pitch_frame(positive_z_axis):
    z = np.asarray(positive_z_axis, dtype=float)
    if z.shape != (3,) or not np.isfinite(z).all():
        raise ValueError("pitch axis must be a finite 3-vector")
    norm = np.linalg.norm(z)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("pitch axis is degenerate")
    z = z / norm
    up = processed_world_up()
    projected = z - up * np.dot(z, up)
    horizontal = np.linalg.norm(projected)
    if horizontal < 0.1:
        raise ValueError("pitch axis is parallel to world vertical")
    projected /= horizontal
    residual = float(np.degrees(np.arccos(np.clip(np.dot(z, projected), -1., 1.))))
    if residual > 30. + 1e-10:
        raise ValueError("pitch axis conflicts with world vertical prior")
    return np.column_stack((np.cross(up, projected), up, projected)), residual


def _wrist(value):
    q = np.asarray(value, dtype=float)
    if q.shape == (16, 4):
        q = q[0]
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("wrist quaternion must be finite and have shape (4,) or (16,4)")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("wrist quaternion is degenerate")
    return q / norm


def solve_level_up_root(base_xyzw, up_xyzw):
    base = R.from_quat(_wrist(base_xyzw))
    pose = R.from_quat(_wrist(up_xyzw))
    rotvec = (pose * base.inv()).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if not np.radians(20.) <= angle <= np.radians(160.):
        raise ValueError("root up rotation must be between 20 and 160 degrees")
    frame, residual = solve_world_vertical_pitch_frame(-rotvec / angle)
    return frame, dict(method=ROOT_METHOD_SIMPLIFIED, measured_axis_rank=1,
        used_poses=["z_rot_pose_1"], measured_angles_deg={"z_rot_pose_1": float(np.degrees(angle))},
        max_residual_deg=residual, raw_prior_residual_deg=residual,
        prior_source=PRIOR_SOURCE, prior_coordinate_version=PRIOR_COORDINATE_VERSION,
        independent_axes_validated=False)
