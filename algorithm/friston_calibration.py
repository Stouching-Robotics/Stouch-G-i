"""Friston wearing calibration behind the existing step and capture callbacks."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from threading import Event, RLock, Thread
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from algorithm.dataflow import get_seq_hand_pose_input
from algorithm.friston_motion import FINGERS, FistPreset, SAMPLES_SCHEMA, build_motion_profile
from algorithm.friston_open_validation import _verify_open_samples
from algorithm.friston_rig import FristonImuFit, FristonRig, RIG_SHA256
from algorithm.friston_rotations import mean_imu_quaternions, solve_multi_axis_root
from algorithm.hand.pressure_stream import UsbPressureSensorStream
from algorithm.hand_usb_receiver import EXPECTED_IMU_IDS, hand_usb_receiver
from common.calibration_root import ROOT_METHOD_FULL, ROOT_METHOD_SIMPLIFIED
from common.calibration_validation import (
    RETURN_OPEN_OMIT_POLICY, RETURN_OPEN_SKIP_POLICY, is_return_open_skipped,
)
from common.calibration_workflow import CONTACT_CAPTURES, INSTALLATION_STEPS, ROOT_CAPTURES, step_id
from common.enums import (
    MoCapCalibrateRootType as Root, MoCapCalibrateInstallationType as Installation,
    MoCapCalibrateShapeType as Shape, MoCapCalibrateFistType as Fist,
)
from common.errors import CalibrationQualityError
from common.ring_buffer import RingBuffer


_ROOT_LABELS = ("horizontal", "vertical_up", "vertical_down", "roll", "yaw")
_ROOT_KEYS = ("base_pose", "z_rot_pose_1", "z_rot_pose_2", "roll_pose", "yaw_pose")


class FristonCalibrationHand:
    """Keep transport, captures and derived mounting tied to one glove session."""

    def __init__(self, hand_config, side):
        self._config = hand_config
        self.side = side
        self._rig = FristonRig(side)
        self._lock = RLock()
        self._kill_switch = Event()
        self._thread = None
        self._stream_epoch = 0
        self._diagnostics_state = {"sink": None}
        self.buffer = RingBuffer(fps=self.fps)
        self.usb_pressure_sensor = UsbPressureSensorStream(fps=self.fps)
        self.missing_imu_ids = EXPECTED_IMU_IDS.copy()
        self.firmware_version = None
        self.calibration_result_revision = 0
        self.last_calibration_ok = False
        self.last_calibration_message = ""
        self.last_calibration_retry_steps = ()
        self.set_calibration_workflow("simplified")

    @property
    def fps(self):
        return self._config.fps or 80

    @property
    def mano_state(self):
        """Compatibility name used by the existing calibration controller."""
        return self

    def start(self):
        config = self._config
        self._thread = Thread(
            target=hand_usb_receiver,
            args=(
                self._kill_switch, config.serial_port, config.usb_vid, config.usb_pid,
                config.channel_to_hand, self.buffer, self._update_missing_imu_ids,
                self.usb_pressure_sensor.buffer, self._update_firmware_version, self._diagnostics_state,
            ),
            kwargs={"on_stream_reset": self._on_stream_reset},
            name="calibration-usb-receiver", daemon=True,
        )
        self._thread.start()

    def shutdown(self):
        self._kill_switch.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _on_stream_reset(self):
        # The receiver must be able to update this while a capture is waiting.
        self._stream_epoch += 1
        self.missing_imu_ids = EXPECTED_IMU_IDS.copy()

    def _update_missing_imu_ids(self, missing):
        self.missing_imu_ids = list(missing)

    def _update_firmware_version(self, version):
        self.firmware_version = str(version)

    def set_calibration_diagnostic_sink(self, sink):
        if sink is not None and not callable(sink):
            raise TypeError("Calibration diagnostics sink must be callable")
        self._diagnostics_state["sink"] = sink

    def set_calibration_workflow(self, mode, *, include_return_open=False):
        if mode not in ("simplified", "full"):
            raise ValueError("Calibration workflow must be full or simplified")
        with self._lock:
            self.calibration_workflow = mode
            self._include_return_open = bool(include_return_open)
            self._reset_root()

    def _reset_root(self):
        self._capture_epoch = self._stream_epoch
        self._roots = {}
        self.calib_root_pose_ready = dict.fromkeys(_ROOT_LABELS, False)
        for label in _ROOT_LABELS:
            setattr(self, f"calib_root_{label}_pose", Rotation.identity(16))
        self._invalidate_root()
        return "Cleared Friston root and wearing captures"

    def _invalidate_root(self):
        self._root_id = None
        self._root_quality = None
        self._fit = None
        self._invalidate_installation()

    def _invalidate_installation(self, *, keep_open=False):
        if not keep_open:
            self._open_samples = None
        self._installation_id = None
        self._contacts = {}
        self._fist_samples = None
        self._invalidate_fine()

    def _invalidate_fine(self):
        self._profile = None
        self._shape_id = None
        self._fist_ready = False
        self._return_samples = None

    def _invalidate_return(self):
        self._return_samples = None
        if self._profile is not None:
            self._profile["quality"]["return_open_validation"] = {"passed": False, "status": "required"}

    def _require_stream_epoch(self):
        if self._capture_epoch != self._stream_epoch:
            raise CalibrationQualityError(
                "Glove stream restarted. Start a new Friston calibration and recapture the poses.",
                retry_steps=("root_level",),
            )

    def _run(self, action, retry_steps=(), *, check_epoch=True):
        with self._lock:
            try:
                if check_epoch:
                    self._require_stream_epoch()
                message = action()
                self._require_stream_epoch()
            except (KeyError, ValueError, RuntimeError, TimeoutError, np.linalg.LinAlgError) as exc:
                self.last_calibration_ok = False
                self.last_calibration_message = str(exc)
                self.last_calibration_retry_steps = getattr(exc, "retry_steps", ()) or retry_steps
            else:
                self.last_calibration_ok = True
                self.last_calibration_message = message
                self.last_calibration_retry_steps = ()
            self.calibration_result_revision += 1

    def _capture(self, count, timeout, *, wrist_only=False):
        if self.missing_imu_ids:
            raise ValueError("All 16 IMUs must be ready before capture")
        rotations = get_seq_hand_pose_input(
            self.buffer, n=count, timeout=timeout, imu_ids=(0,) if wrist_only else None,
        )
        self._require_stream_epoch()
        return np.stack([rotation.as_quat() for rotation in rotations])

    def _capture_root(self, trigger):
        label = trigger.value
        if self.calibration_workflow == "simplified" and trigger not in (Root.HORIZONTAL, Root.VERTICAL_UP):
            raise ValueError("Select full calibration for additional root poses")
        if trigger == Root.HORIZONTAL:
            self._reset_root()
        self._roots.pop(label, None)
        self.calib_root_pose_ready[label] = False
        self._invalidate_root()
        wrist_only = not (self.calibration_workflow == "simplified" and trigger == Root.HORIZONTAL)
        samples = self._capture(30, 5, wrist_only=wrist_only)
        self._roots[label] = samples
        setattr(self, f"calib_root_{label}_pose", Rotation.from_quat(mean_imu_quaternions(samples)))
        self.calib_root_pose_ready[label] = True
        return f"Captured Friston root pose {label}"

    def _compute_root(self):
        self._invalidate_root()
        simplified = self.calibration_workflow == "simplified"
        required = _ROOT_LABELS[:2] if simplified else ("horizontal", "vertical_up", "vertical_down", "yaw")
        missing = [label for label in required if label not in self._roots]
        if missing:
            raise ValueError(f"Missing root calibration poses: {', '.join(missing)}")
        means = {label: mean_imu_quaternions(samples) for label, samples in self._roots.items()}
        base = means["horizontal"]
        if simplified:
            fit = FristonImuFit(self._rig, base, means["vertical_up"])
            quality = fit.root_quality
            open_samples = self._roots["horizontal"]
            self._mount(fit, open_samples)
            self._open_samples = open_samples
            self._installation_id = uuid4().hex
        else:
            root = solve_multi_axis_root(
                base, *(means.get(label) for label in _ROOT_LABELS[1:]),
                side=self.side, yaw_capture=self._yaw_direction,
            )
            fit = FristonImuFit(self._rig, base, root_frame=root["frame"])
            quality = {
                **root["quality"], "method": "multi-axis-kabsch",
                "independent_axes_validated": root["independent_axes_validated"],
            }
        self._fit = fit
        self._root_quality = quality
        self._root_id = uuid4().hex
        result = f"Friston root calculated, wrist axis residual {quality['max_residual_deg']:.1f} deg"
        return result + (" using the world-vertical prior" if simplified else "")

    @property
    def _yaw_direction(self):
        return "left" if self.side == "right" else "right"

    def update_root_calibrator(self, trigger):
        if trigger == Root.RESET:
            self._run(self._reset_root, check_epoch=False)
        elif trigger == Root.CALC:
            if self.calibration_workflow == "simplified":
                retries = (*ROOT_CAPTURES[:2], "root_compute")
            else:
                retries = (*ROOT_CAPTURES, "root_compute")
            self._run(self._compute_root, retries)
        else:
            self._run(lambda: self._capture_root(trigger), (step_id("root", trigger),),
                      check_epoch=trigger != Root.HORIZONTAL)

    @staticmethod
    def _mount(fit, samples):
        neutral = np.broadcast_to(fit.rig.reference_angles[0], (len(samples), 21))
        fit.set_mounting(fit.estimate_mounting(samples, neutral))

    def _capture_installation(self):
        if self._fit is None:
            raise ValueError("Complete Friston root calibration first")
        self._invalidate_installation()
        self._open_samples = self._capture(30, 5)
        return "Captured the Friston installation open pose"

    def _compute_installation(self):
        self._invalidate_installation(keep_open=True)
        if self._fit is None or self._open_samples is None:
            raise ValueError("Complete root and capture installation open first")
        self._mount(self._fit, self._open_samples)
        self._installation_id = uuid4().hex
        return "Friston static mounting calculated from the open pose"

    def update_installation_calibrator(self, trigger):
        if trigger == Installation.POSE_0:
            self._run(self._capture_installation, ("installation_open",))
        elif trigger == Installation.CALC:
            self._run(self._compute_installation, INSTALLATION_STEPS)
        elif trigger == Installation.RESET:
            def reset():
                self._invalidate_installation()
                return "Cleared Friston installation and dependent captures"
            self._run(reset)
        else:
            def unsupported():
                raise ValueError("Friston installation uses the open-hand POSE_0 capture")
            self._run(unsupported, INSTALLATION_STEPS)

    def _require_installation(self):
        if self._installation_id is None:
            raise ValueError("Complete Friston root and installation calibration first")

    def _capture_contact(self, finger):
        self._require_installation()
        self._invalidate_fine()
        self._contacts.pop(finger, None)
        self._contacts[finger] = self._capture(90, 4)
        return f"Captured Friston thumb-to-{finger} contact"

    @property
    def _dependency(self):
        return f"{self._root_id}:{self._installation_id}"

    def _compute_profile(self):
        self._invalidate_fine()
        self._require_installation()
        missing = tuple(f"contact_{finger}" for finger in FINGERS if finger not in self._contacts)
        if missing:
            raise CalibrationQualityError("Capture the missing Friston fingertip contacts", retry_steps=missing)
        fitted = build_motion_profile(self._fit, self._open_samples, self._contacts, self._dependency)
        quality = fitted["quality"]
        gaps = {finger: quality["captured"][finger]["p95_gap_mm"] for finger in FINGERS}
        if not quality["capture_gate_passed"]:
            failed = tuple(f"contact_{finger}" for finger, gap in gaps.items() if not np.isfinite(gap) or gap > 12.)
            raise CalibrationQualityError(
                "Friston fingertip contact fit exceeded the 12 mm capture tolerance",
                retry_steps=(*(failed or CONTACT_CAPTURES), "contact_compute"),
            )
        profile = fitted["profile"]
        profile["calibration_policy"] = "installation-neutral-verified-v1"
        profile["quality"]["return_open_validation"] = {"passed": False, "status": "required"}
        self._profile = profile
        self._shape_id = uuid4().hex
        if self._fist_samples is not None:
            self._compute_fist()
        return f"Friston contact fit passed, worst p95 gap {max(gaps.values()):.1f} mm"

    def _sample_payload(self):
        return {
            "schema": SAMPLES_SCHEMA, "side": self.side, "depends_on": self._dependency,
            "neutral_xyzw": self._open_samples, "contacts_xyzw": self._contacts,
            "fist_xyzw": self._fist_samples,
        }

    def _reset_fist(self):
        self._fist_samples = None
        self._fist_ready = False
        self._invalidate_return()
        return "Cleared the Friston fist capture"

    def _capture_fist(self):
        self._reset_fist()
        self._require_installation()
        self._fist_samples = self._capture(150, 10)
        return "Captured the Friston fist"

    def _compute_fist(self):
        self._fist_ready = False
        self._invalidate_return()
        if self._profile is None or self._fist_samples is None:
            raise ValueError("Finish contact calibration and capture a fist first")
        try:
            FistPreset(self._sample_payload(), self._rig, self._dependency)
        except ValueError as exc:
            raise CalibrationQualityError(f"Friston fist solve failed: {exc}", retry_steps=("fist_capture",)) from exc
        self._fist_ready = True
        if self.calibration_workflow == "simplified" and not self._include_return_open:
            self._skip_open(RETURN_OPEN_OMIT_POLICY)
        return "Calculated the Friston fist preset"

    def update_fist_calibrator(self, trigger):
        actions = {Fist.CAPTURE: self._capture_fist, Fist.CALC: self._compute_fist, Fist.RESET: self._reset_fist}
        self._run(actions[trigger], ("fist_capture",))

    def _require_fist(self):
        if self._profile is None or not self._fist_ready:
            raise ValueError("Finish Friston contact and fist calibration first")

    def _skip_open(self, policy):
        self._require_fist()
        self._invalidate_return()
        self._profile["quality"]["return_open_validation"] = {
            "passed": False, "status": "skipped", "policy": policy, "used_for_fit": False, "count": 0,
        }
        return "Skipped the optional return-open verification"

    def skip_return_open_validation(self, *, workflow_omission=False):
        policy = RETURN_OPEN_OMIT_POLICY if workflow_omission else RETURN_OPEN_SKIP_POLICY
        self._run(lambda: self._skip_open(policy), ("verify_open",))

    def _verify_open(self):
        self._require_fist()
        self._invalidate_return()
        samples = self._capture(90, 8)
        validation = _verify_open_samples(samples, self._fit, self._profile)
        self._profile["quality"]["return_open_validation"] = validation
        self._return_samples = samples
        return "Friston return-open verification passed"

    def update_shape_calibrator(self, trigger):
        if trigger == Shape.CALC:
            self._run(self._compute_profile, (*CONTACT_CAPTURES, "contact_compute"))
        elif trigger == Shape.VERIFY_OPEN:
            self._run(self._verify_open, ("verify_open",))
        elif trigger == Shape.RESET:
            def reset():
                self._contacts.clear()
                self._reset_fist()
                self._invalidate_fine()
                return "Cleared Friston fine calibration"
            self._run(reset)
        else:
            self._run(lambda: self._capture_contact(trigger.value), (step_id("shape", trigger),))

    def recompute_calibration(self):
        def recompute():
            message = self._compute_profile()
            if self._fist_samples is None:
                raise CalibrationQualityError("Capture the Friston fist before completing recompute",
                                              retry_steps=("fist_capture",))
            return message
        self._run(recompute, (*CONTACT_CAPTURES, "contact_compute"))

    def build_payload(self, usb_serial):
        with self._lock:
            self._require_stream_epoch()
            self._require_fist()
            quality = self._profile["quality"]
            validation = quality.get("return_open_validation", {})
            if validation.get("passed") is not True and not is_return_open_skipped(validation):
                raise ValueError("Complete or explicitly skip return-open verification before saving")
            poses = {
                key: mean_imu_quaternions(self._roots[label]).tolist() if label in self._roots else None
                for key, label in zip(_ROOT_KEYS, _ROOT_LABELS)
            }
            samples = self._sample_payload()
            for name in ("neutral_xyzw", "fist_xyzw"):
                samples[name] = samples[name].tolist()
            samples["contacts_xyzw"] = {finger: sample.tolist() for finger, sample in self._contacts.items()}
            if validation.get("passed"):
                samples["return_open_xyzw"] = self._return_samples.tolist()
            config = self._config
            return {
                "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                "tool": "imu_calibrate_cli", "side": self.side, "hardware_id": config.hardware_id,
                "calibration_hardware_id": config.calibration_hardware_id,
                "channel_to_hand": list(config.channel_to_hand), "usb_serial": usb_serial,
                "usb_vid": config.usb_vid, "usb_pid": config.usb_pid, "fps": self.fps,
                "param_inst_calib": {
                    "side": self.side,
                    "metadata": {
                        "root_method": (ROOT_METHOD_SIMPLIFIED if self.calibration_workflow == "simplified"
                                        else ROOT_METHOD_FULL),
                        "root_yaw_capture": self._yaw_direction,
                        "root_calibration_id": self._root_id,
                        "installation_calibration_id": self._installation_id,
                    },
                    "calibration_quat_set": poses,
                },
                "friston": {"geometry_sha256": RIG_SHA256, "mounting_xyzw": self._fit.mounting.tolist()},
                "param_shape_calib": {
                    "depends_on": self._dependency, "shape_calibration_id": self._shape_id,
                    "params": [0.] * 10, "sensor_motion_refit": copy.deepcopy(self._profile),
                    "motion_refit_samples": samples,
                },
                "quality": {
                    "root": copy.deepcopy(self._root_quality),
                    "installation": {"mode": "static_pose_0_only", "local_inst_calibrated": True},
                    "shape": {},
                    "friston": {
                        "geometry_sha256": RIG_SHA256, "capture_gate_passed": quality["capture_gate_passed"],
                        "captured": copy.deepcopy(quality["captured"]),
                    },
                },
            }
