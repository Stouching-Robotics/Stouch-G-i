"""Readonly force request queue coordination."""
import queue,time
REQUEST_TIMEOUT_S=7.0
_RESULT_POLL_S=0.2
def request_once(command_queue, result_queue, request, stop_event,
                 timeout_s: float = REQUEST_TIMEOUT_S) -> dict:
    """Put one request and wait for its answer on the matching result queue."""

    side = request.get("side")
    base = {"side": side, "op": request.get("op"),
            "set_index": request.get("set_index")}
    try:
        command_queue.put_nowait(request)
    except queue.Full:
        return {**base, "ok": False,
                "message": "a command is already in flight for this hand"}
    except (OSError, EOFError, ValueError) as exc:
        return {**base, "ok": False, "message": f"command queue unusable: {exc}"}

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return {**base, "ok": False, "message": "session is shutting down"}
        try:
            payload = result_queue.get(timeout=_RESULT_POLL_S)
        except queue.Empty:
            continue
        except (OSError, EOFError, ValueError) as exc:
            return {**base, "ok": False, "message": f"result queue unusable: {exc}"}
        if payload.get("side") != side:
            continue  # not this hand's answer; keep looking
        if payload.get("op") != request.get("op"):
            continue
        if request.get("op") == "set_force_baseline" and any(
                payload.get(key) != request.get(key)
                for key in ("connection_id", "baseline_id")):
            continue  # an answer to an older baseline update
        return payload
    return {**base, "ok": False,
            "message": (f"no answer within {timeout_s:.0f} s; the solver "
                        f"process may be busy on another command")}

