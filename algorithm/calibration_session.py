"""Scripted session adapter over the Friston wearing calibration workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common.paths import SDK_ROOT
from common.errors import CalibrationError, CalibrationQualityError
from common.calibration_workflow import invalidated_step_ids
from common.types import (
    CalibrationProgress, CalibrationStep, CalibrationStepResult, DeviceHealth)
from common.enums import (
    MoCapCalibrateInstallationType,
    MoCapCalibrateRootType,
    MoCapCalibrateShapeType,
    MoCapCalibrateFistType,
)
from algorithm.imu_calibrate_cli import (
    ImuCalibrationCLI, compute_motion_rms)


PROJECT_ROOT = SDK_ROOT

def _step_specs(side: str, workflow: str = "simplified", include_return_open: bool = False) -> tuple:
    """Root wizard steps with a per-side yaw direction for a palm-facing grip.

    LEFT 手 yaw 保持"水平右转"（指尖朝身体中线）；RIGHT 手取反为"水平左转"
    （指尖同样朝内），两手指尖朝内、掌心相对。ROLL 翻掌方向由
    Raw2MANOCalibrator 按侧自动镜像，步骤文案不需要区分左右。
    """
    yaw_title = ("Root - Yaw Left" if str(side).lower() == "right"
                 else "Root - Yaw Right")
    specs = (
        (CalibrationStep("root_level", "Root - Level", "capture", "root"),
         MoCapCalibrateRootType.HORIZONTAL),
        (CalibrationStep("root_up", "Root - Fingertips Up", "capture", "root"),
         MoCapCalibrateRootType.VERTICAL_UP),
        (CalibrationStep("root_down", "Root - Fingertips Down", "capture", "root"),
         MoCapCalibrateRootType.VERTICAL_DOWN),
        (CalibrationStep("root_roll", "Root - Roll (Thumb Up)", "capture", "root"),
         MoCapCalibrateRootType.ROLL),
        (CalibrationStep("root_yaw", yaw_title, "capture", "root"),
         MoCapCalibrateRootType.YAW),
        (CalibrationStep("root_compute", "Compute Root Calibration", "compute", "root"),
         MoCapCalibrateRootType.CALC),
        (CalibrationStep("installation_open", "Installation - Open Hand", "capture",
                         "installation"),
         MoCapCalibrateInstallationType.POSE_0),
        (CalibrationStep("installation_compute", "Compute Installation Calibration",
                         "compute", "installation"),
         MoCapCalibrateInstallationType.CALC),
        (CalibrationStep("contact_index", "Contact - Thumb to Index", "capture", "contact"),
         MoCapCalibrateShapeType.INDEX),
        (CalibrationStep("contact_middle", "Contact - Thumb to Middle", "capture", "contact"),
         MoCapCalibrateShapeType.MIDDLE),
        (CalibrationStep("contact_ring", "Contact - Thumb to Ring", "capture", "contact"),
         MoCapCalibrateShapeType.RING),
        (CalibrationStep("contact_little", "Contact - Thumb to Little", "capture", "contact"),
         MoCapCalibrateShapeType.LITTLE),
        (CalibrationStep("contact_compute", "Compute Fine Calibration", "compute", "contact"),
         MoCapCalibrateShapeType.CALC),
        (CalibrationStep("fist_capture", "Fist - Capture and Compute", "capture", "fist"),
         MoCapCalibrateFistType.CAPTURE),
        (CalibrationStep("verify_open", "Return-Open Verification", "capture", "contact"),
         MoCapCalibrateShapeType.VERIFY_OPEN),
    )
    if workflow == "full":
        return specs
    if workflow != "simplified":
        raise ValueError("calibration workflow must be full or simplified")
    selected = tuple(specs[index] for index in (0, 1, 5, 8, 9, 10, 11, 12, 13))
    selected = (*selected[:2],
                (CalibrationStep("root_compute", "Compute Root and Installation Calibration",
                                 "compute", "root"), MoCapCalibrateRootType.CALC),
                *selected[3:])
    return selected + ((specs[-1],) if include_return_open else ())


class ImuCalibratorEngine:
    def __init__(self, side: str, serial_port: str | None = None,
                 registry: Path | str | None = None,
                 config: Path | str | None = None,
                 startup_timeout: float = 30.0, *,
                 workflow: str = "simplified", include_return_open: bool = False):
        self.side = str(side).lower()
        if self.side not in ("left", "right"):
            raise ValueError("side must be left or right")
        _step_specs(self.side, workflow, include_return_open)
        self.workflow = workflow
        self.include_return_open = bool(include_return_open)
        self.registry = Path(registry or PROJECT_ROOT / "config" / "glove_devices.json")
        self.config = Path(config or PROJECT_ROOT / "config" / (
            "config_left.json" if self.side == "left" else "config.json"))
        args = argparse.Namespace(
            side=self.side,
            config=self.config,
            registry=self.registry,
            serial_port=serial_port,
            fps=None,
            startup_timeout=float(startup_timeout),
            out=None,
            force=False,
            check_stream=False,
            flow="all",
            calibration_workflow=self.workflow,
            include_return_open=self.include_return_open,
        )
        self._backend = ImuCalibrationCLI(args)
        self._connected = False
        self._completed: list[str] = []
        self._skipped: list[str] = []

    @property
    def steps(self) -> tuple[CalibrationStep, ...]:
        return tuple(spec[0] for spec in self._specs())

    def _specs(self):
        return _step_specs(self.side, getattr(self, "workflow", "full"),
                           getattr(self, "include_return_open", False))

    def connect(self) -> DeviceHealth:
        if not self._connected:
            try:
                self._backend.boot()
            except Exception as exc:
                self._backend.shutdown()
                raise CalibrationError(str(exc)) from exc
            self._connected = True
        return self.health()

    def start_new(self) -> CalibrationProgress:
        # A failed reconnect/reset still begins a new attempt: old completion
        # must not authorize saving the previous session's calibration.
        self._completed.clear()
        self._skipped = []
        if not self._connected:
            self.connect()
        self._backend.hand.update_root_calibrator(MoCapCalibrateRootType.RESET)
        return self.progress()

    def _find_step(self, step_id: str):
        for index, (step, trigger) in enumerate(self._specs()):
            if step.id == step_id:
                return index, step, trigger
        raise ValueError(f"unknown calibration step: {step_id}")

    def run_step(self, step_id: str) -> CalibrationStepResult:
        # Earlier SDK clients explicitly ran this after fist_capture. The
        # guided capture now includes the solve, so acknowledge it once
        # without another capture, solve, or completion marker.
        if step_id == "fist_compute":
            if not {"fist_capture", "contact_compute"}.issubset(self._completed):
                raise CalibrationError("Run fist_capture; it now includes fist calculation")
            return CalibrationStepResult("fist_compute", True,
                                         "Fist calculation was included in fist_capture")
        index, step, trigger = self._find_step(step_id)
        invalidated = invalidated_step_ids(step.id, getattr(self, 'workflow', 'full'))
        self._completed = [value for value in self._completed
                           if value not in invalidated]
        self._skipped = [value for value in getattr(self, "_skipped", ())
                         if value in self._completed]
        if not self._connected:
            self.connect()
        compute_only = trigger in (MoCapCalibrateRootType.CALC,
                                   MoCapCalibrateInstallationType.CALC,
                                   MoCapCalibrateShapeType.CALC)
        if not compute_only and self._backend.hand.missing_imu_ids:
            raise CalibrationError("all 16 IMUs must be ready")
        state = self._backend.hand.mano_state
        def command(callback, value):
            before = state.calibration_result_revision
            callback(value)
            if state.calibration_result_revision <= before:
                raise CalibrationError("calibration command did not complete")
            if not state.last_calibration_ok:
                raise CalibrationQualityError(str(state.last_calibration_message),
                    retry_steps=getattr(state, 'last_calibration_retry_steps', ()))

        try:
            if step.stage == "root":
                command(self._backend.hand.update_root_calibrator, trigger)
            elif step.stage == "installation":
                command(self._backend.hand.update_installation_calibrator, trigger)
            elif step.stage == "fist":
                command(self._backend.hand.update_fist_calibrator, trigger)
                command(self._backend.hand.update_fist_calibrator, MoCapCalibrateFistType.CALC)
            else:
                command(self._backend.hand.update_shape_calibrator, trigger)
        except Exception as exc:
            raise CalibrationQualityError(str(exc),
                retry_steps=getattr(exc, 'retry_steps', ()) or
                getattr(state, 'last_calibration_retry_steps', ())) from exc
        result = CalibrationStepResult(
            step_id=step.id,
            ok=bool(state.last_calibration_ok),
            message=str(state.last_calibration_message),
        )
        if not result.ok:
            raise CalibrationQualityError(result.message)
        self._completed = [s.id for s in self.steps if s.id in set(self._completed) | {step.id}]
        return result

    def retry_step(self, step_id: str) -> CalibrationStepResult:
        return self.run_step(step_id)

    def set_workflow(self, workflow: str, *, include_return_open: bool = False):
        _step_specs(self.side, workflow, include_return_open)
        if self._connected:
            self._backend.hand.set_calibration_workflow(workflow, include_return_open=include_return_open)
            self._backend.hand.update_root_calibrator(MoCapCalibrateRootType.RESET)
        self.workflow, self.include_return_open = workflow, bool(include_return_open)
        self._backend.args.calibration_workflow = workflow
        self._backend.args.include_return_open = bool(include_return_open)
        self._completed.clear()
        self._skipped = []
        return self.progress()

    def recompute(self) -> CalibrationStepResult:
        if not self._connected:
            raise CalibrationError('Connect the glove before recomputing')
        self._completed = [x for x in self._completed if x not in ('contact_compute', 'verify_open')]
        self._skipped = []
        state = self._backend.hand.mano_state
        before = state.calibration_result_revision
        self._backend.hand.recompute_calibration()
        if state.calibration_result_revision <= before or not state.last_calibration_ok:
            raise CalibrationQualityError(str(state.last_calibration_message),
                retry_steps=getattr(state, 'last_calibration_retry_steps', ()))
        self._completed = [s.id for s in self.steps
                           if s.id in set(self._completed) | {'contact_compute', 'fist_capture'}]
        return CalibrationStepResult('contact_compute', True, str(state.last_calibration_message))

    def skip_return_open(self) -> CalibrationStepResult:
        """Explicitly finish the optional final check without marking it passed."""
        if (getattr(self, "workflow", "full") == "simplified"
                and not getattr(self, "include_return_open", False)):
            raise CalibrationError("This simplified workflow already omits return-open verification")
        index, step, _ = self._find_step("verify_open")
        self._completed = [value for value in self._completed
                           if self._find_step(value)[0] < index]
        self._skipped = []
        required = {item.id for item in self.steps[:index]}
        if not required.issubset(self._completed):
            raise CalibrationError("Complete all preceding steps before skipping return-open verification")
        if not self._connected:
            raise CalibrationError("Connect the glove before skipping return-open verification")
        state = self._backend.hand.mano_state
        before = state.calibration_result_revision
        self._backend.hand.skip_return_open_validation()
        if state.calibration_result_revision <= before:
            raise CalibrationError("calibration skip command did not complete")
        if not state.last_calibration_ok:
            raise CalibrationQualityError(str(state.last_calibration_message))
        self._completed.append(step.id)
        self._skipped.append(step.id)
        return CalibrationStepResult(step.id, True, str(state.last_calibration_message))

    def progress(self) -> CalibrationProgress:
        total = len(self._specs())
        next_index = next((i for i, spec in enumerate(self._specs())
                           if spec[0].id not in self._completed), total - 1)
        return CalibrationProgress(
            current_index=next_index,
            total_steps=total,
            completed_step_ids=tuple(self._completed),
            complete=len(self._completed) == total,
            skipped_step_ids=tuple(getattr(self, "_skipped", ())),
        )

    def health(self) -> DeviceHealth:
        if not self._connected or self._backend.hand is None:
            return DeviceHealth(False, 0, (), 0.0, "not connected")
        missing = tuple(self._backend.hand.missing_imu_ids)
        counter = self._backend.hand.buffer.get_counter()
        return DeviceHealth(
            connected=True,
            ready_imu_count=16 - len(missing),
            missing_imus=missing,
            stream_fps=float(self._backend.hand_config.fps or 0),
            message=f"buffer frames={counter}",
        )

    def motion_rms(self) -> tuple[float, float]:
        if not self._connected:
            return 0.0, 0.0
        current = self.steps[self.progress().current_index]
        shared = (getattr(self, "workflow", "full") == "simplified"
                  and current.id == "root_level")
        return compute_motion_rms(self._backend.hand.buffer, 30,
                                  imu_ids=(0,) if current.stage == "root" and not shared else None)

    def save(self, filename: str, overwrite: bool = False,
             directory: Path | str | None = None) -> Path:
        if not self.progress().complete:
            raise CalibrationError("complete every calibration step before saving")
        value = str(filename or "").strip()
        if not value or "/" in value or "\\" in value:
            raise ValueError("filename must be a JSON file name without a path")
        if not value.lower().endswith(".json"):
            value += ".json"
        folder = Path(directory or PROJECT_ROOT / "calibration")
        output = folder / value
        if output.exists() and not overwrite:
            raise CalibrationError(f"calibration already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self._backend.build_payload()
        payload["tool"] = "runtime.ImuCalibrator"
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        return output

    def close(self) -> None:
        self._backend.shutdown()
        self._connected = False
