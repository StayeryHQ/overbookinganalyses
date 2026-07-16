# dash_app/backend/jobs.py
# Tiny file-backed job runner for the app's long tasks (data update, retrain,
# artifact builds).
#
# Why not Dash background callbacks: their progress/running state lives inside
# the callback context and dies when the user navigates away — the UI then shows
# a stuck loading bar forever while the work keeps running invisibly. Here every
# job writes its state to Data/jobs/<name>.json; any page polls that file with a
# dcc.Interval and renders real progress, success or error — across page changes
# AND app restarts. One job per name at a time; jobs are daemon threads.
#
# Deliberate limitation: threads cannot be killed, so there is NO cancel. A job
# runs to completion (or error) and says so. Honest > pretend-cancel.

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import src

_LOCK = threading.Lock()
_THREADS: dict[str, threading.Thread] = {}
_CANCEL: dict[str, threading.Event] = {}


class JobCancelled(Exception):
    """Raised inside a job (via the progress checkpoint) when the user cancels."""


def _cancel_path(name: str) -> Path:
    return _job_path(name).with_suffix(".cancel")


def cancel(name: str) -> bool:
    """Request COOPERATIVE cancellation. The job stops at its next progress checkpoint
    and is marked 'cancelled' WITHOUT writing its result — so whatever data existed
    before stays untouched. Threads can't be force-killed, so a step already running
    (e.g. a model fit or a single SHAP compute) finishes that step before the cancel is
    noticed.

    Cancellation is FILE-based (Data/jobs/<name>.cancel holds the running job's start
    timestamp), so it works even when the cancel click and the job thread live in
    different worker processes — an in-memory Event alone would not cross processes.
    """
    st = read(name)
    if st.get("status") != "running":
        return False
    try:
        _cancel_path(name).write_text(str(st.get("started", "")))
    except Exception:  # noqa: BLE001
        pass
    ev = _CANCEL.get(name)   # also flip the in-process Event for the common single-worker case
    if ev is not None:
        ev.set()
    return True


def _cancel_requested(name: str, started, ev: "threading.Event | None") -> bool:
    """True if THIS job (identified by its start timestamp) has been asked to cancel."""
    if ev is not None and ev.is_set():
        return True
    try:
        p = _cancel_path(name)
        return p.exists() and p.read_text().strip() == str(started)
    except Exception:  # noqa: BLE001
        return False


def _job_path(name: str) -> Path:
    d = src.data_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"


def _write(name: str, state: dict) -> None:
    try:
        _job_path(name).write_text(json.dumps(state))
    except Exception:  # noqa: BLE001 — a status write must never kill the job
        pass


def read(name: str) -> dict:
    """Current state of a job: {status: idle|running|done|error, progress,
    message, started, finished, result, error}. Detects orphaned jobs: a file
    that says 'running' without a live thread (app restart / crash) is flipped
    to a loud error instead of showing an eternal loading bar."""
    p = _job_path(name)
    if not p.exists():
        return {"status": "idle"}
    try:
        state = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {"status": "idle"}
    if state.get("status") == "running":
        t = _THREADS.get(name)
        if t is None or not t.is_alive():
            state["status"] = "error"
            state["error"] = ("The app was restarted (or the worker died) while this "
                              "job was running. Start it again.")
            state["finished"] = time.time()
            _write(name, state)
    return state


def running(name: str) -> bool:
    return read(name).get("status") == "running"


def start(name: str, fn: Callable, *args, **kwargs) -> bool:
    """Run fn(progress, *args, **kwargs) in a daemon thread; progress(msg, frac)
    streams into the status file. Returns False if `name` is already running."""
    with _LOCK:
        if running(name):
            return False
        cancel_ev = threading.Event()
        _CANCEL[name] = cancel_ev
        state = {"status": "running", "progress": 0.0, "message": "starting…",
                 "started": time.time(), "finished": None, "result": None, "error": None}
        _write(name, state)

        def progress(msg: str, frac: float) -> None:
            # Cooperative cancel checkpoint: raise BEFORE doing/reporting more work, so
            # the job aborts before it writes anything and the previous data survives.
            if _cancel_requested(name, state["started"], cancel_ev):
                raise JobCancelled()
            state["message"] = str(msg)
            state["progress"] = max(0.0, min(1.0, float(frac)))
            _write(name, state)

        def run() -> None:
            try:
                result = fn(progress, *args, **kwargs)
                state.update(status="done", progress=1.0, message="done",
                             finished=time.time(), result=result)
            except JobCancelled:
                state.update(status="cancelled", progress=0.0, finished=time.time(),
                             message="Cancelled — previous data kept.", result=None)
            except Exception as e:  # noqa: BLE001 — the whole point: fail LOUDLY
                state.update(status="error", finished=time.time(),
                             error=f"{type(e).__name__}: {str(e)[:600]}",
                             trace=traceback.format_exc()[-2000:])
            _write(name, state)

        t = threading.Thread(target=run, daemon=True, name=f"job-{name}")
        _THREADS[name] = t
        t.start()
        return True
