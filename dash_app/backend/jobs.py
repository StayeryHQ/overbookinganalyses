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
        state = {"status": "running", "progress": 0.0, "message": "starting…",
                 "started": time.time(), "finished": None, "result": None, "error": None}
        _write(name, state)

        def progress(msg: str, frac: float) -> None:
            state["message"] = str(msg)
            state["progress"] = max(0.0, min(1.0, float(frac)))
            _write(name, state)

        def run() -> None:
            try:
                result = fn(progress, *args, **kwargs)
                state.update(status="done", progress=1.0, message="done",
                             finished=time.time(), result=result)
            except Exception as e:  # noqa: BLE001 — the whole point: fail LOUDLY
                state.update(status="error", finished=time.time(),
                             error=f"{type(e).__name__}: {str(e)[:600]}",
                             trace=traceback.format_exc()[-2000:])
            _write(name, state)

        t = threading.Thread(target=run, daemon=True, name=f"job-{name}")
        _THREADS[name] = t
        t.start()
        return True
