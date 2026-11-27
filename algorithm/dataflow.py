import threading
import time
from collections.abc import Sequence

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from common.datamodels import PackedWristTrackerMsg
from common.ring_buffer import RingBuffer
from algorithm.utils import get_average_rotation


class CalibrationDataError(ValueError):
    """Raised when live sensor data is unsafe to use for calibration."""


def input_synchronizer(
        kill_switch: threading.Event,
        in_buf_map: dict[str, RingBuffer],
        out_buf: RingBuffer,
        max_sample_age_us: int = 150_000,
):
    """
    同步多个数据流。这是一个简单实现，以固定刷新率收集每个流的最新状态并推送到输出环形缓冲区。
    """
    fps = out_buf.fps

    last_t = time.time()

    while True:
        if kill_switch.is_set():
            break

        # 各接收器使用本机到达时间；超出同步窗口的旧流会在下方置空，
        # 不会把旧 Tracker 位姿和新 IMU 姿态拼成同一帧。
        curr_frame = dict()
        timestamps_us = []
        now_us = int(time.time() * 1_000_000)
        for k, in_buf in in_buf_map.items():
            data, index, err = in_buf.peek()
            curr_frame[k] = data

            timestamp_us = getattr(data, 'timestamp_us', getattr(data, 'sys_ticks', None))
            if timestamp_us is not None:
                timestamp_us = int(timestamp_us)
                timestamps_us.append(timestamp_us)
                # This gate also covers a single enabled stream.  Pairwise
                # synchronization alone cannot detect that its receiver has
                # stopped and would otherwise republish one old frame forever.
                if now_us - timestamp_us > max_sample_age_us:
                    curr_frame[k] = None

        # Do not combine arbitrarily old tracker/IMU samples.  The GUI can
        # safely keep its last displayed pose until streams are aligned again.
        if len(timestamps_us) >= 2 and max(timestamps_us) - min(timestamps_us) > 50_000:
            for key, data in list(curr_frame.items()):
                timestamp_us = getattr(data, 'timestamp_us', getattr(data, 'sys_ticks', None))
                if timestamp_us is not None and max(timestamps_us) - int(timestamp_us) > 50_000:
                    curr_frame[key] = None

        out_buf.push(curr_frame)

        # 刷新率控制
        now = time.time()
        last_t += 1 / fps
        if now < last_t:
            time.sleep(last_t - now)
        else:
            last_t = now


def sample_tracker_pose_frame_from_msg_buffer(in_buf: RingBuffer, count: int = 50, timeout=None) -> list[PackedWristTrackerMsg]:
    """
    :return: PackedWristTrackerMsg 列表
    """
    frame, idx, err = in_buf.peek()
    frames = []
    next_idx = max(idx + 1, 1)
    start_t = time.time()
    while len(frames) < count:
        frame, _, err = in_buf.peek(next_idx)
        if timeout and time.time() - start_t > timeout:
            raise TimeoutError("Timeout, not enough frames collected")
        if err is None:
            next_idx += 1
            if frame is not None:
                frames.append(frame)
        else:
            time.sleep(0.1)

    logger.info("pose frames collected, length: {}".format(len(frames)))
    return frames


def sample_imu_quat_from_msg_buffer(in_buf: RingBuffer, count: int, timeout=None,
                                    max_sensor_age_s: float = 0.10) -> list[R]:
    """
    从缓冲区连续取 count 帧 IMU 姿态。取的是调用之后新到的帧，
    因此至少需要 count / 实际帧率 秒。

    :return: Rotation 对象列表
    """
    _, idx, _ = in_buf.peek()
    # RingBuffer.head 从 1 开始，索引 0 那一格在写满一圈之前永远是 None。
    # 缓冲区为空时 peek() 会返回 idx=0，直接用会死等，这里跳到 1。
    next_idx = max(idx + 1, 1)
    start_counter = in_buf.get_counter()
    imu_rotations = []
    rejected_frames = 0
    rejected_imu_ids: set[int] = set()
    overruns = 0
    start_t = time.monotonic()
    def check_deadline():
        elapsed = time.monotonic() - start_t
        if timeout and elapsed > timeout:
            arrived = in_buf.get_counter() - start_counter
            raise TimeoutError(
                f"Calibration capture timed out: required {count} frames but "
                f"received {len(imu_rotations)}. The buffer received {arrived} "
                f"new frames in {elapsed:.1f}s "
                f"({arrived / max(elapsed, 1e-6):.1f} fps). "
                f"Sampling window restarted {overruns} times after scheduling overruns. "
                f"The stream is stopped, delayed or too slow; check the connection and host load."
            )
    while len(imu_rotations) < count:
        check_deadline()
        frame, actual_idx, err = in_buf.peek(next_idx)
        if err is None and frame is not None:
            if actual_idx < next_idx:
                time.sleep(0.01)
                continue
            if actual_idx > next_idx:
                # An overwritten cursor is snapped to the newest frame by
                # RingBuffer.peek. Resume there, never count it repeatedly.
                imu_rotations.clear()
                overruns += 1
            next_idx = actual_idx + 1
            if not frame.calibration_ready(max_sensor_age_s=max_sensor_age_s):
                rejected_frames += 1
                invalid = (
                    ~np.isfinite(frame.sensor_age_s)
                    | (frame.sensor_age_s > max_sensor_age_s)
                    | frame.rejected_mask
                )
                rejected_imu_ids.update(int(i) for i in np.flatnonzero(invalid))
                rejected_imu_ids.update(frame.bno_not_ready_ids)
                continue
            imu_rotations.append(frame.imu_rotation)
        else:
            time.sleep(0.01)
    check_deadline()
    logger.info(
        "imu_quat frames collected, length: {}, rejected: {}, invalid ids: {}".format(
            len(imu_rotations), rejected_frames, sorted(rejected_imu_ids)))
    return imu_rotations


def validate_static_imu_rotations(
        rotations: list[R], max_rms_motion_deg: float = 6.0,
        max_peak_motion_deg: float = 16.0,
        imu_ids: Sequence[int] | None = None) -> R:
    """Validate a nominally static 16-IMU sequence and return its mean.

    ``imu_ids`` optionally restricts the stillness gate to a subset of IMU
    indices (e.g. the contact finger's 3 + the thumb's 3 sensors used by fine
    calibration); ``None`` validates all 16.  The returned mean rotation is
    always computed over all 16 sensors.
    """
    if not rotations:
        raise CalibrationDataError("The calibration pose sequence is empty")
    res = get_average_rotation(rotations)
    deviations_deg = np.asarray([
        np.degrees((rotation * res.inv()).magnitude())
        for rotation in rotations
    ])
    rms_per_imu = np.sqrt(np.mean(deviations_deg ** 2, axis=0))
    peak_per_imu = np.max(deviations_deg, axis=0)
    if imu_ids is None:
        checked = np.arange(deviations_deg.shape[1])
    else:
        checked = np.asarray(list(imu_ids), dtype=int)
    bad = np.flatnonzero(
        (rms_per_imu[checked] > max_rms_motion_deg)
        | (peak_per_imu[checked] > max_peak_motion_deg))
    if len(bad):
        detail = ", ".join(
            f"imu_{checked[idx]}:rms={rms_per_imu[checked[idx]]:.1f}deg,"
            f"peak={peak_per_imu[checked[idx]]:.1f}deg"
            for idx in bad)
        raise CalibrationDataError(
            f"The hand moved during calibration ({detail}); the capture was rejected")
    return res


def get_averaged_hand_pose_input(in_buf: RingBuffer, n: int = 50, timeout=2,
                                 max_rms_motion_deg: float = 6.0,
                                 max_peak_motion_deg: float = 16.0,
                                 imu_ids: Sequence[int] | None = None) -> R:
    """
    从IMU四元数获取平均手部姿态

    :return R: 平均旋转，形状为 (16,)
    """
    rotations = sample_imu_quat_from_msg_buffer(in_buf, n, timeout)
    return validate_static_imu_rotations(
        rotations, max_rms_motion_deg=max_rms_motion_deg,
        max_peak_motion_deg=max_peak_motion_deg, imu_ids=imu_ids)


def get_seq_hand_pose_input(
        in_buf: RingBuffer, n: int = 400, timeout=6,
        max_rms_motion_deg: float = 6.0,
        max_peak_motion_deg: float = 16.0,
        imu_ids: Sequence[int] | None = None) -> list[R]:
    """
    从IMU四元数获取手部姿态序列

    :return list[R]: 旋转列表
    """
    rot_seq = sample_imu_quat_from_msg_buffer(in_buf, n, timeout)
    # Fine shape calibration is also a static light-touch capture.  Previously
    # only root/installation captures had a motion gate, so transition frames
    # silently became a large fingertip residual during shape optimization.
    # The stillness gate only checks the relevant IMUs (contact finger + thumb).
    validate_static_imu_rotations(
        rot_seq, max_rms_motion_deg=max_rms_motion_deg,
        max_peak_motion_deg=max_peak_motion_deg, imu_ids=imu_ids)
    return rot_seq

def get_averaged_tracker_position_input(in_buf: RingBuffer, hand_id, n: int = 50, timeout=2) -> np.ndarray | None:
    """
    从腕部追踪器获取平均位置

    :return np.ndarray: 平均位置
    """
    tracker_frames = sample_tracker_pose_frame_from_msg_buffer(in_buf, n, timeout)
    try:
        positions = np.array([frame.tracked_objects[hand_id.value].pos for frame in tracker_frames])
    except (KeyError, TypeError, AttributeError) as exc:
        raise CalibrationDataError(f"Tracker object {hand_id.value} is missing") from exc
    if positions.shape != (n, 3) or np.any(~np.isfinite(positions)):
        raise CalibrationDataError("Tracker position data is invalid")
    mean = np.mean(positions, axis=0)
    deviation_mm = np.linalg.norm(positions - mean, axis=1) * 1000.0
    if float(np.sqrt(np.mean(deviation_mm ** 2))) > 3.0 or float(np.max(deviation_mm)) > 8.0:
        raise CalibrationDataError(
            f"The tracker moved during capture (RMS "
            f"{np.sqrt(np.mean(deviation_mm ** 2)):.1f}mm, "
            f"peak {np.max(deviation_mm):.1f}mm)")
    return mean

def get_averaged_tracker_rotation_input(in_buf: RingBuffer, hand_id, n: int = 50, timeout=2) -> np.ndarray | None:
    """
    从腕部追踪器获取平均旋转

    :return np.ndarray: 平均旋转，形状为 (1,)
    """
    tracker_frames = sample_tracker_pose_frame_from_msg_buffer(in_buf, n, timeout)
    try:
        rotations = [frame.tracked_objects[hand_id.value].rot for frame in tracker_frames]
    except (KeyError, TypeError, AttributeError) as exc:
        raise CalibrationDataError(f"Tracker object {hand_id.value} is missing") from exc
    mean = get_average_rotation(rotations)
    deviation_deg = np.degrees(np.asarray([(rotation * mean.inv()).magnitude() for rotation in rotations]))
    if float(np.sqrt(np.mean(deviation_deg ** 2))) > 3.0 or float(np.max(deviation_deg)) > 8.0:
        raise CalibrationDataError(
            f"The tracker rotated during capture (RMS "
            f"{np.sqrt(np.mean(deviation_deg ** 2)):.1f} deg, "
            f"peak {np.max(deviation_deg):.1f} deg)")
    return mean
