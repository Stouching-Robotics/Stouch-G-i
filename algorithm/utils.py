from __future__ import annotations

import queue
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from common.datamodels import WristTrackerMsg
from common.resources import hand_assets_dir

# Heavy dependencies (torch / manotorch / open3d / polyscope / statemachine /
# tqdm) are imported lazily inside the functions that use them, so the
# lightweight glove_lite runtime can import the pure-numpy helpers below
# without pulling those packages in.


# 单例装饰器
def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)

        return instances[cls]

    return get_instance


# IMU 体坐标系 -> MANO 解剖坐标系的基变换 (2026-08-06 实测标定)
#
# IMU 实际安装约定:
#   +Y = MCP 指向 TIP (指尖方向)
#   +Z = 垂直手背向上 (背侧法向)
#   +X = Y × Z
#
# MANO 解剖坐标系 (manotorch AxisLayerFK, 见 axislayer.py 的 b/u/l axis):
#   X = b_axis (twist), 沿骨骼指向父关节, 即近端方向 = -Y_imu
#   Y = u_axis (spread), 外展轴            = +Z_imu
#   Z = l_axis (bend),  屈伸轴, 正向为屈曲  = -X_imu
#     (实测: 食指下弯 59° 时 rotvec ≈ (-0.85, -0.57, -0.15), 主轴 -X;
#      MANO 侧 compose(bend=+0.6) 使指尖朝掌心运动, 故正 bend = 屈曲)
#
# ------------------------------------------------------------------
# 本矩阵与 calibration.py 的根坐标系列序是一对耦合参数，必须同时修改。
#
# 当前组合 (= fsglove 上游约定):
#   calibration.py `_get_global_installation_error` 列序 = (Z x p, Z, p)
#   本矩阵 = [[-1,0,0],[0,0,1],[0,1,0]]
#
# 此前组合:
#   列序 = (p x Z, p, Z)，本矩阵 = [[0,-1,0],[0,0,1],[-1,0,0]]
#   其列含义为 X_imu -> (0,0,-1)  Y_imu -> (-1,0,0)  Z_imu -> (0,1,0)
#
# 两套根坐标系相差 180° 旋转 P = [[-1,0,0],[0,0,1],[0,1,0]]，
# 而 P 正好等于本矩阵在 fsglove 约定下应取的值。只回退其中一处，
# 屈曲会被映射成 twist，表现为手指弯曲方向反转。
#
# det = +1, 是纯旋转。
_imu_axis_correction_matrix_np = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
])
_imu_axis_correction_det = round(np.linalg.det(_imu_axis_correction_matrix_np))


def correct_imu_quaternion_axes(quaternion_xyzw: np.ndarray) -> np.ndarray:
    corrected_quaternion = np.empty(4, dtype=float)
    corrected_quaternion[:3] = _imu_axis_correction_det * (_imu_axis_correction_matrix_np @ quaternion_xyzw[:3])
    corrected_quaternion[3] = quaternion_xyzw[3]
    return corrected_quaternion


def correct_mano_axis_mapping(rotvec):
    return rotvec


def convert_optitrack_messages_to_pos_with_quat(optitrack_message: list[dict], tracked_objects_names) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    pos_dict = {obj: [] for obj in tracked_objects_names}
    quat_dict = {obj: [] for obj in tracked_objects_names}
    for objs_msg in optitrack_message:
        for obj in objs_msg.keys():
            if obj not in tracked_objects_names:
                continue
            obj_frame = objs_msg[obj]
            pos_ = obj_frame.position
            pos = np.asarray([pos_.x, pos_.y, pos_.z])
            quat_ = obj_frame.rotation
            quat = np.asarray([quat_.w, quat_.x, quat_.y, quat_.z])
            pos_dict[obj].append(pos)
            quat_dict[obj].append(quat)
    return pos_dict, quat_dict


def get_average_rotation(rotations: list[R]):
    """
    Rotations 是一个 3D numpy 数组的列表，每个数组是一批旋转
    """
    rot_seq_np = np.stack([r.as_quat() for r in rotations], axis=0)
    if rot_seq_np.ndim == 3:
        ref_quat = rot_seq_np[0:1, :, :]
        same_hemisphere = np.sum(rot_seq_np * ref_quat, axis=2, keepdims=True) >= 0
        rot_seq_np = np.where(same_hemisphere, rot_seq_np, -rot_seq_np)
    else:
        ref_quat = rot_seq_np[0:1, :]
        same_hemisphere = np.sum(rot_seq_np * ref_quat, axis=1, keepdims=True) >= 0
        rot_seq_np = np.where(same_hemisphere, rot_seq_np, -rot_seq_np)

    rot_mean = np.mean(rot_seq_np, axis=0)
    if len(rot_mean.shape) > 1:
        norm = np.linalg.norm(rot_mean, axis=1)
        if np.any(norm <= 1e-8):
            raise ValueError("Cannot average rotations: quaternion mean is near zero")
        rot_mean /= norm[:, np.newaxis]
    else:
        norm = np.linalg.norm(rot_mean)
        if norm <= 1e-8:
            raise ValueError("Cannot average rotations: quaternion mean is near zero")
        rot_mean /= norm
    return R.from_quat(rot_mean)


def _get_thumb_and_target_points(points: torch.Tensor, pose_category: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    pose_category = pose_category.to(points.device).long().view(-1)
    thumb_point = points[:, 4, :]
    oppose_fingertip_index = pose_category * 4
    idx = torch.arange(points.shape[0], device=points.device)
    oppose_fingertip_points = points[idx, oppose_fingertip_index]
    return thumb_point, oppose_fingertip_points


def get_loss(loss_f: nn.Module, points: torch.Tensor, pose_category: torch.Tensor) -> torch.Tensor:
    """
    :param points: (N, 21, 3) N 个批次的 MANO 手部 21 个关键点
    :param pose_category: (N, 1) N 个类别
    :param loss_f: 损失函数


    :return loss: 损失值
    """
    thumb_point, oppose_fingertip_points = _get_thumb_and_target_points(points, pose_category)
    return loss_f(thumb_point, oppose_fingertip_points)


def get_direction_loss(points: torch.Tensor, pose_category: torch.Tensor, reference_vectors: torch.Tensor) -> torch.Tensor:
    import torch

    thumb_point, oppose_fingertip_points = _get_thumb_and_target_points(points, pose_category)
    cur_vec = oppose_fingertip_points - thumb_point
    ref_vec = reference_vectors.to(points.device)
    return torch.relu(-(cur_vec * ref_vec).sum(dim=1)).mean()


def get_optimized_shape_params(
        joints: torch.Tensor,
        finger_category: torch.Tensor,
        hand_category: str,
        iter_num: int = 500,
        learning_rate: float = 0.01,
        regular_term_weight: float = 1e-5,
        direction_term_weight: float = 0.1,
        output_queue: queue.Queue = None,
        compute_device: torch.device | None = None,
        initial_shape_params: torch.Tensor | np.ndarray | None = None,
        show_progress: bool = True,
) -> torch.Tensor:
    """
    ``contact_loss`` is MSE in metres squared (normally 1e-4..1e-3), so the
    beta prior must stay on the same numerical scale.  Larger historic values
    such as 1e-2 overwhelm the contact gradient and pin beta near zero.

    :param joints: (N, 16, 3) N 个批次的 16 个关节旋转向量（MANO 模型）
    :param finger_category: (N, 1) N 个类别。2 代表食指，3 代表中指，4 代表无名指，5 代表小指
    :param hand_category: "left" 或 "right"
    :param iter_num: 迭代次数
    :param learning_rate: 学习率
    :param regular_term_weight: 正则项权重，值越大越抑制手型参数变化
    :param output_queue: 用于存放结果的队列
    :param compute_device: 计算设备

    :return shape_params: (10, ) 优化后的形状参数
    """
    assert hand_category in ['left', 'right']
    assert joints.shape[0] == finger_category.shape[0]

    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from manotorch.manolayer import ManoLayer, MANOOutput

    if compute_device is None:
        compute_device = torch.device("cpu")

    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side=hand_category,
        center_idx=None,
        flat_hand_mean=True,
        mano_assets_root=str(hand_assets_dir()),
    ).to(compute_device)
    joints = joints.view(joints.shape[0], -1).to(compute_device)
    finger_category = finger_category.to(compute_device).long()
    if initial_shape_params is None:
        shape_params_init = torch.zeros((10,), dtype=torch.float32, device=compute_device)
    else:
        shape_params_init = torch.as_tensor(initial_shape_params, dtype=torch.float32, device=compute_device)
        if shape_params_init.shape != (10,):
            raise ValueError(f"initial_shape_params must have shape (10,), got {tuple(shape_params_init.shape)}")
    shape_params = shape_params_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([shape_params], lr=learning_rate)

    loss_f = nn.MSELoss()
    with torch.no_grad():
        reference_results: MANOOutput = mano_layer(joints, shape_params.detach().repeat((joints.shape[0], 1)))
        ref_thumb, ref_target = _get_thumb_and_target_points(reference_results.joints, finger_category)
        reference_vectors = ref_target - ref_thumb

    begin_loss = 0.0
    final_loss = 0.0
    with tqdm(range(iter_num), disable=not show_progress) as pbar:
        for _ in range(iter_num):
            mano_results: MANOOutput = mano_layer(joints, shape_params.repeat((joints.shape[0], 1)))
            verts = mano_results.joints
            optimizer.zero_grad()
            contact_loss = get_loss(loss_f, verts, finger_category)
            direction_loss = get_direction_loss(verts, finger_category, reference_vectors)
            loss = (
                contact_loss
                + direction_term_weight * direction_loss
                + regular_term_weight * torch.linalg.norm(shape_params)
            )
            if _ == 0:
                begin_loss = loss.item()
            final_loss = loss.item()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                shape_params.clamp_(-3.0, 3.0)

            pbar.set_description(f'Loss: {loss.detach().cpu().numpy()}')
            pbar.update()
    result = shape_params.detach().cpu()
    if show_progress:
        print(f"After {iter_num} iterations, the loss dropped from {begin_loss} to {final_loss}")
    if output_queue is not None:
        output_queue.put_nowait(result.numpy())
    return result


def evaluate_shape_contact(
        joints: torch.Tensor,
        finger_category: torch.Tensor,
        hand_category: str,
        shape_params: torch.Tensor | np.ndarray,
        compute_device: torch.device | None = None,
) -> dict[str, object]:
    """Return fingertip opposition residuals for a proposed MANO shape."""
    import torch
    from manotorch.manolayer import ManoLayer

    if compute_device is None:
        compute_device = torch.device("cpu")

    mano_layer = ManoLayer(
        rot_mode="axisang", use_pca=False, side=hand_category,
        center_idx=None, flat_hand_mean=True,
        mano_assets_root=str(hand_assets_dir()),
    ).to(compute_device)
    pose = joints.reshape(joints.shape[0], -1).to(compute_device)
    shape = torch.as_tensor(shape_params, dtype=torch.float32, device=compute_device)
    with torch.no_grad():
        output = mano_layer(pose, shape.repeat((pose.shape[0], 1)))
        thumb, target = _get_thumb_and_target_points(output.joints, finger_category)
        distance_m = torch.linalg.norm(thumb - target, dim=1).detach().cpu().numpy()
    category_np = finger_category.detach().cpu().numpy().reshape(-1)
    finger_names = {2: "index", 3: "middle", 4: "ring", 5: "little"}
    per_finger = {}
    for category, name in finger_names.items():
        values = distance_m[category_np == category]
        if values.size:
            per_finger[name] = {
                "mean_mm": float(np.mean(values) * 1000.0),
                "p95_mm": float(np.percentile(values, 95) * 1000.0),
                "max_mm": float(np.max(values) * 1000.0),
                "sample_count": int(values.size),
            }
    return {
        "mean_contact_error_mm": float(np.mean(distance_m) * 1000.0),
        "p95_contact_error_mm": float(np.percentile(distance_m, 95) * 1000.0),
        "max_contact_error_mm": float(np.max(distance_m) * 1000.0),
        "per_finger": per_finger,
    }

def mano_rotvec_to_rfu_euler(mano_pose) -> np.ndarray:
    euler = np.zeros((16, 3))  # 存储结果
    r_matrix = R.from_rotvec(mano_pose.detach().cpu().numpy()).as_matrix()
    for i in range(16):
        rotation_matrix = r_matrix[i]  # 转换为旋转矩阵
        # 提取欧拉角 xyz
        # TODO: 优化
        euler[i, 0] = -np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])  # 绕x轴的角度
        euler[i, 1] = np.arctan2(-rotation_matrix[0, 1], np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 2] ** 2))  # 绕y轴的角度
        euler[i, 2] = np.arctan2(rotation_matrix[0, 0], rotation_matrix[1, 1])
    return euler


def progress_statemachine(m: StateMachine, target_state=None):
    """
    如果 target_state 为 None，则推进到一个稳定状态
    """
    if target_state is not None:
        target_value = getattr(target_state, "value", target_state)
        while getattr(m.current_state, "value", m.current_state) != target_value:
            m.loop()
            time.sleep(0.1)
        return

    last_state = None
    while m.current_state != last_state:
        last_state = m.current_state
        m.loop()
        time.sleep(0.1)


class CoordinateFrame:
    def __init__(self, key: str, scale=0.05):
        """
        用于 Polyscope 可视化的简便坐标系
        """
        self.key = key
        self.T = np.eye(4)
        self.scale = scale
        self.init_nodes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]) * self.scale
        self.nodes = np.copy(self.init_nodes)

        self.nodes_transposed_init = np.ones(shape=(4, 4))
        self.nodes_transposed_init[:3, :] = self.nodes.T

    def set_transform(self, T):
        """
        :param T: 4x4 变换矩阵
        """
        self.T = T
        nodes = self.T @ self.nodes_transposed_init
        self.nodes = nodes[:3, :].T

    def register_model(self):
        """
        将坐标系模型注册到 Polyscope，此函数只需调用一次
        """
        import polyscope as ps

        ps.register_curve_network(f"x_axis_{self.key}", nodes=self.nodes[[0, 1], :], edges=np.array([[0, 1]]), color=(1, 0, 0))
        ps.register_curve_network(f"y_axis_{self.key}", nodes=self.nodes[[0, 2], :], edges=np.array([[0, 1]]), color=(0, 1, 0))
        ps.register_curve_network(f"z_axis_{self.key}", nodes=self.nodes[[0, 3], :], edges=np.array([[0, 1]]), color=(0, 0, 1))

    def update_model(self, msg: WristTrackerMsg):
        """
        在 user_callback 中调用此函数以更新模型
        """
        import polyscope as ps

        if msg is not None:
            T = np.eye(4)
            T[:3, :3] = msg.rot.as_matrix()
            T[:3, 3] = msg.pos
            self.set_transform(T)
        ps.get_curve_network(f"x_axis_{self.key}").update_node_positions(self.nodes[[0, 1], :])
        ps.get_curve_network(f"y_axis_{self.key}").update_node_positions(self.nodes[[0, 2], :])
        ps.get_curve_network(f"z_axis_{self.key}").update_node_positions(self.nodes[[0, 3], :])


_constant_frame_mano_to_world_rot = R.from_euler(seq='xyz', angles=[np.pi / 2, 0, -np.pi]).inv()


def frame_mano_to_world(rotation: R) -> R:
    """
    将 MANO 旋转转换到世界坐标系，使得休息姿态下的 MANO 模型显示为：
    - 中指指向 +X
    - 掌心指向 -Z
    """
    return _constant_frame_mano_to_world_rot * rotation

def get_nws_points(point_cloud: o3d.geometry.PointCloud):
    import open3d as o3d

    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    render_option = vis1.get_render_option()
    render_option.background_color = (0.2, 0.2, 0.2)
    vis1.add_geometry(point_cloud)
    vis1.update_renderer()
    vis1.run()  # 用户选取点
    vis1.destroy_window()

    pts_sel = vis1.get_picked_points()
    print(pts_sel)


def get_optimized_shape_params_from_keypoints(
        key_pts: torch.Tensor,
        vert_ids,
        hand_category: str,
        iter_num: int = 500,
        learning_rate: float = 0.01,
        regular_term_weight: float = 0.01,
        compute_device: torch.device | None = None
) -> torch.Tensor:
    """
    :param finger_category: (N, 1) N 个类别。2 代表食指，3 代表中指，4 代表无名指，5 代表小指
    :param hand_category: "left" 或 "right"
    :param iter_num: 迭代次数
    :param learning_rate: 学习率
    :param regular_term_weight: 正则项权重，值越大优化越激进
    :param output_queue: 用于存放结果的队列
    :param compute_device: 计算设备

    :return shape_params: (10, ) 优化后的形状参数
    """
    assert hand_category in ['left', 'right']

    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from manotorch.manolayer import ManoLayer, MANOOutput

    if compute_device is None:
        compute_device = torch.device("cpu")

    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side=hand_category,
        center_idx=None,
        flat_hand_mean=True,
        mano_assets_root=str(hand_assets_dir()),
    ).to(compute_device)
    wrist_pos = torch.zeros((3,), requires_grad=True, device=compute_device)
    wrist_rot = torch.zeros((3,), requires_grad=True, device=compute_device)
    zero_tensor = torch.zeros((45,), requires_grad=True, device=compute_device)
    # Deterministic start: repeated calibration of identical keypoints must
    # not land in a different local optimum merely because global RNG moved.
    shape_params = torch.zeros((10,), requires_grad=True, device=compute_device)
    optimizer = torch.optim.Adam([wrist_pos, wrist_rot, shape_params], lr=learning_rate)

    loss_f = nn.MSELoss()

    key_pts = key_pts.to(compute_device)
    hand_root = torch.mean(key_pts[-2: ], dim = 0)
    key_pts = torch.concat([key_pts[: -2], hand_root.unsqueeze(0)], dim=0)

    begin_loss = 0.0
    final_loss = 0.0
    with tqdm(range(iter_num)) as pbar:
        for _ in range(iter_num):
            rot_params = torch.concat([wrist_rot, zero_tensor], dim=0)
            mano_results: MANOOutput = mano_layer(rot_params.unsqueeze(0), shape_params.unsqueeze(0))
            verts = mano_results.verts.squeeze(0)
            mano_key_pts = verts[vert_ids.to(compute_device)]
            mano_root = torch.mean(verts[-2: ], dim=0)
            mano_key_pts = torch.concat([mano_key_pts[:-2], mano_root.unsqueeze(0)], dim=0)

            optimizer.zero_grad()
            loss = loss_f(mano_key_pts, key_pts) + regular_term_weight * torch.linalg.norm(shape_params)
            if _ == 0:
                begin_loss = loss.item()
            final_loss = loss.item()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                shape_params.clamp_(-3.0, 3.0)

            pbar.set_description(f'Loss: {loss.detach().cpu().numpy()}')
            pbar.update()
    result = shape_params.detach().cpu()
    print(f"After {iter_num} iterations, the loss dropped from {begin_loss} to {final_loss}")
    return result
