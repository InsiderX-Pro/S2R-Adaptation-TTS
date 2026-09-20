"""Persistent JSON-lines adapter for an isolated pyannote environment.

This parent-side module intentionally uses only the Python standard library.
The pyannote imports live in :mod:`pyannote_worker`, which is launched with the
configured interpreter and kept alive across recordings so the model is loaded
only once.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, TextIO
import uuid


DEFAULT_PYANNOTE_PYTHON = os.environ.get("PYANNOTE_PYTHON", sys.executable)
DEFAULT_PYANNOTE_MODEL = os.environ.get("PYANNOTE_MODEL_DIR", "models/pyannote-speaker-diarization")

_STARTUP_REQUEST_ID = "__startup__"
_EOF = object()


@dataclass(frozen=True)
class PyannoteConfig:
    python_executable: str = DEFAULT_PYANNOTE_PYTHON
    model_dir: str = DEFAULT_PYANNOTE_MODEL
    device: str = "cuda"
    startup_timeout_s: float = 120.0
    request_timeout_s: float = 3600.0
    shutdown_timeout_s: float = 10.0


def _worker_script_path() -> Path:
    """Return the worker entrypoint; split out so tests can substitute a fake."""

    return Path(__file__).with_name("pyannote_worker.py")


def _positive_timeout(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


class PyannoteDiarizer:
    """Keep one offline pyannote worker alive and diarize recordings serially."""

    def __init__(self, config: PyannoteConfig) -> None:
        if not isinstance(config, PyannoteConfig):
            raise TypeError("config must be a PyannoteConfig")
        self.config = config
        self._worker: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[object] = queue.Queue()
        self._pending: dict[str, dict[str, Any]] = {}
        self._stderr_tail: deque[str] = deque(maxlen=100)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.RLock()

    @property
    def worker_pid(self) -> int | None:
        worker = self._worker
        return worker.pid if worker is not None and worker.poll() is None else None

    def _read_stdout(self, stream: TextIO) -> None:
        try:
            for line in stream:
                self._stdout_queue.put(line)
        finally:
            self._stdout_queue.put(_EOF)

    def _read_stderr(self, stream: TextIO) -> None:
        try:
            for line in stream:
                text = line.rstrip("\r\n")
                if text:
                    self._stderr_tail.append(text)
        except (OSError, ValueError):
            return

    def _stderr_context(self) -> str:
        if not self._stderr_tail:
            return ""
        return "; stderr_tail=" + " | ".join(self._stderr_tail)[-4000:]

    def _wait_response(self, request_id: str, timeout_s: float) -> dict[str, Any]:
        pending = self._pending.pop(request_id, None)
        if pending is not None:
            return pending
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"pyannote worker timed out waiting for {request_id}")
            try:
                item = self._stdout_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"pyannote worker timed out waiting for {request_id}") from exc
            if item is _EOF:
                returncode = self._worker.poll() if self._worker is not None else None
                raise RuntimeError(
                    f"pyannote worker exited before replying (returncode={returncode})"
                    f"{self._stderr_context()}"
                )
            raw_line = str(item).strip()
            if not raw_line:
                continue
            try:
                response = json.loads(raw_line)
            except json.JSONDecodeError:
                # Third-party progress output occasionally reaches stdout.  Only
                # valid protocol objects participate in request matching.
                continue
            if not isinstance(response, dict):
                continue
            response_id = str(response.get("request_id") or "")
            if not response_id:
                continue
            if response_id == request_id:
                return response
            self._pending[response_id] = response

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        self._dispose_worker()

        executable = Path(self.config.python_executable).expanduser().resolve()
        model_dir = Path(self.config.model_dir).expanduser().resolve()
        worker_script = _worker_script_path().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"pyannote Python executable does not exist: {executable}")
        if not model_dir.is_dir():
            raise FileNotFoundError(f"pyannote model snapshot does not exist: {model_dir}")
        if not worker_script.is_file():
            raise FileNotFoundError(f"pyannote worker script does not exist: {worker_script}")

        startup_timeout = _positive_timeout(self.config.startup_timeout_s, name="startup_timeout_s")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        command = [
            str(executable),
            str(worker_script),
            "--model-dir",
            str(model_dir),
            "--device",
            str(self.config.device),
        ]
        worker = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        self._worker = worker
        self._stdout_queue = queue.Queue()
        self._pending.clear()
        self._stderr_tail.clear()
        assert worker.stdout is not None and worker.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(worker.stdout,),
            name="pyannote-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(worker.stderr,),
            name="pyannote-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            startup = self._wait_response(_STARTUP_REQUEST_ID, startup_timeout)
            if startup.get("status") != "ready":
                raise RuntimeError(str(startup.get("error") or "pyannote worker failed during startup"))
        except Exception:
            self._terminate_worker()
            raise

    def _send(self, payload: dict[str, Any]) -> None:
        worker = self._worker
        if worker is None or worker.poll() is not None or worker.stdin is None:
            returncode = worker.poll() if worker is not None else None
            raise RuntimeError(
                f"pyannote worker is not running (returncode={returncode}){self._stderr_context()}"
            )
        try:
            worker.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            worker.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError(f"failed to write to pyannote worker{self._stderr_context()}") from exc

    def _terminate_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5.0)
        self._dispose_worker()

    def _dispose_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            for stream in (worker.stdin, worker.stdout, worker.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self._pending.clear()

    def diarize(self, audio_path: str | Path) -> dict[str, Any]:
        """Diarize one local audio file and return a structured result.

        Operational failures are returned as ``status="error"`` so a batch can
        continue and route the recording to review.  Calls are serialized because
        one JSON-lines worker owns one GPU pipeline.
        """

        started = time.monotonic()
        source = Path(audio_path).expanduser().resolve()
        base: dict[str, Any] = {
            "audio_path": str(source),
            "turns": [],
            "exclusive_turns": [],
            "model_dir": str(Path(self.config.model_dir).expanduser().resolve()),
            "device": str(self.config.device),
        }
        try:
            if not source.is_file():
                raise FileNotFoundError(f"audio file does not exist: {source}")
            request_timeout = _positive_timeout(self.config.request_timeout_s, name="request_timeout_s")
            with self._lock:
                self._start_worker()
                request_id = f"diarize-{uuid.uuid4().hex}"
                self._send(
                    {
                        "request_id": request_id,
                        "op": "diarize",
                        "audio_path": str(source),
                    }
                )
                try:
                    response = self._wait_response(request_id, request_timeout)
                except TimeoutError:
                    # A timed-out GPU call cannot be safely reused for the next
                    # recording; terminate it and lazily start a fresh worker.
                    self._terminate_worker()
                    raise
            status = str(response.get("status") or "error")
            if status != "ok":
                raise RuntimeError(str(response.get("error") or "pyannote worker returned an error"))
            turns = response.get("turns")
            if not isinstance(turns, list):
                raise RuntimeError("pyannote worker returned non-list turns")
            exclusive_turns = response.get("exclusive_turns", [])
            if not isinstance(exclusive_turns, list):
                raise RuntimeError("pyannote worker returned non-list exclusive_turns")
            result = dict(response)
            result.pop("request_id", None)
            result.update(base)
            result["status"] = "ok"
            result["turns"] = turns
            result["exclusive_turns"] = exclusive_turns
            result["elapsed_s"] = round(time.monotonic() - started, 3)
            return result
        except Exception as exc:
            return {
                **base,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.monotonic() - started, 3),
            }

    def close(self) -> None:
        """Gracefully stop the worker; safe to call more than once."""

        with self._lock:
            worker = self._worker
            if worker is None:
                return
            shutdown_timeout = _positive_timeout(self.config.shutdown_timeout_s, name="shutdown_timeout_s")
            if worker.poll() is None:
                try:
                    self._send(
                        {
                            "request_id": f"close-{uuid.uuid4().hex}",
                            "op": "close",
                        }
                    )
                except RuntimeError:
                    pass
                if worker.stdin is not None:
                    try:
                        worker.stdin.close()
                    except OSError:
                        pass
                try:
                    worker.wait(timeout=shutdown_timeout)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    try:
                        worker.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait(timeout=5.0)
            self._dispose_worker()

    def __enter__(self) -> "PyannoteDiarizer":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "DEFAULT_PYANNOTE_MODEL",
    "DEFAULT_PYANNOTE_PYTHON",
    "PyannoteConfig",
    "PyannoteDiarizer",
]
