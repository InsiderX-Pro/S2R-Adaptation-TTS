from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "ominivoice_data_pipeline"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PARENT = _load_module("ominivoice_pyannote_parent_under_test", "pyannote_diarization.py")
WORKER = _load_module("ominivoice_pyannote_worker_under_test", "pyannote_worker.py")


FAKE_WORKER = textwrap.dedent(
    r"""
    import argparse
    import json
    import os
    from pathlib import Path
    import sys
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("non-json startup noise", flush=True)
    if Path(args.model_dir).name == "startup_error":
        print(json.dumps({
            "request_id": "__startup__",
            "status": "error",
            "error": "fake model load failure",
        }), flush=True)
        raise SystemExit(2)
    print(json.dumps({
        "request_id": "__startup__",
        "status": "ready",
        "worker_pid": os.getpid(),
    }), flush=True)

    for line in sys.stdin:
        request = json.loads(line)
        request_id = request.get("request_id", "")
        operation = request.get("op")
        if operation == "close":
            print(json.dumps({"request_id": request_id, "status": "closed"}), flush=True)
            break
        audio_path = str(request.get("audio_path") or "")
        if Path(audio_path).name == "timeout.wav":
            time.sleep(10)
        if Path(audio_path).name == "error.wav":
            response = {
                "request_id": request_id,
                "status": "error",
                "turns": [],
                "exclusive_turns": [],
                "error": "fake inference failure",
            }
        else:
            response = {
                "request_id": request_id,
                "status": "ok",
                "audio_path": audio_path,
                "turns": [{
                    "start_ms": 0,
                    "end_ms": 1000,
                    "speaker_id": "SPEAKER_00",
                    "confidence_source": "unavailable",
                    "overlap": False,
                }],
                "exclusive_turns": [{
                    "start_ms": 0,
                    "end_ms": 1000,
                    "speaker_id": "SPEAKER_00",
                    "confidence_source": "unavailable",
                    "overlap": False,
                }],
                "exclusive_source": "pyannote_exclusive",
                "speaker_count": 1,
                "worker_pid": os.getpid(),
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)
    """
)


class FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(self, rows: list[tuple[FakeSegment, str, str]]) -> None:
        self.rows = rows

    def itertracks(self, *, yield_label: bool = False):
        if not yield_label:
            raise AssertionError("worker must request labels")
        yield from self.rows


class WrappedResult:
    def __init__(self, annotation: FakeAnnotation, exclusive: FakeAnnotation | None = None) -> None:
        self.speaker_diarization = annotation
        if exclusive is not None:
            self.exclusive_speaker_diarization = exclusive


class PyannoteWorkerHelperTests(unittest.TestCase):
    def test_annotation_to_turns_supports_wrapped_pyannote_4_result(self) -> None:
        annotation = FakeAnnotation(
            [
                (FakeSegment(3.0, 4.0), "track-3", "B"),
                (FakeSegment(0.0, 2.0), "track-1", "A"),
                (FakeSegment(1.5, 3.0), "track-2", "B"),
                (FakeSegment(4.0, 4.0), "invalid", "C"),
            ]
        )

        turns = WORKER.annotation_to_turns(WrappedResult(annotation))

        self.assertEqual(
            [(turn["start_ms"], turn["end_ms"], turn["speaker_id"]) for turn in turns],
            [(0, 2_000, "A"), (1_500, 3_000, "B"), (3_000, 4_000, "B")],
        )
        self.assertTrue(turns[0]["overlap"])
        self.assertTrue(turns[1]["overlap"])
        self.assertFalse(turns[2]["overlap"])
        self.assertTrue(all("confidence" not in turn for turn in turns))
        self.assertTrue(all(turn["confidence_source"] == "unavailable" for turn in turns))
        self.assertEqual(turns[0]["start_time"], turns[0]["start_ms"])
        json.dumps(turns)

    def test_annotation_to_turns_accepts_direct_annotation(self) -> None:
        annotation = FakeAnnotation(
            [
                (FakeSegment(0.0, 1.0), "one", "A"),
                (FakeSegment(0.5, 1.5), "two", "A"),
            ]
        )

        turns = WORKER.annotation_to_turns(annotation)

        self.assertEqual(len(turns), 2)
        self.assertFalse(turns[0]["overlap"])
        self.assertFalse(turns[1]["overlap"])

    def test_annotation_to_turns_rejects_non_annotation(self) -> None:
        with self.assertRaisesRegex(TypeError, "itertracks"):
            WORKER.annotation_to_turns(object())

    def test_diarization_to_timelines_uses_native_exclusive_annotation(self) -> None:
        regular = FakeAnnotation(
            [
                (FakeSegment(0.0, 2.0), "regular-a", "A"),
                (FakeSegment(1.5, 3.0), "regular-b", "B"),
            ]
        )
        exclusive = FakeAnnotation(
            [
                (FakeSegment(0.0, 1.75), "exclusive-a", "A"),
                (FakeSegment(1.75, 3.0), "exclusive-b", "B"),
            ]
        )

        timelines = WORKER.diarization_to_timelines(WrappedResult(regular, exclusive))

        self.assertEqual(timelines["exclusive_source"], "pyannote_exclusive")
        self.assertTrue(any(turn["overlap"] for turn in timelines["turns"]))
        self.assertEqual(
            [(turn["start_ms"], turn["end_ms"], turn["speaker_id"]) for turn in timelines["exclusive_turns"]],
            [(0, 1_750, "A"), (1_750, 3_000, "B")],
        )
        self.assertTrue(all(not turn["overlap"] for turn in timelines["exclusive_turns"]))

    def test_diarization_to_timelines_derives_exclusive_for_direct_annotation(self) -> None:
        annotation = FakeAnnotation(
            [
                (FakeSegment(0.0, 2.0), "regular-a", "A"),
                (FakeSegment(1.5, 3.0), "regular-b", "B"),
            ]
        )

        timelines = WORKER.diarization_to_timelines(annotation)

        self.assertEqual(timelines["exclusive_source"], "derived_from_regular")
        self.assertEqual(
            [(turn["start_ms"], turn["end_ms"], turn["speaker_id"]) for turn in timelines["exclusive_turns"]],
            [(0, 2_000, "A"), (2_000, 3_000, "B")],
        )


class PyannoteAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.worker_script = self.root / "fake_pyannote_worker.py"
        self.worker_script.write_text(FAKE_WORKER, encoding="utf-8")

    def _config(self, **overrides):
        values = {
            "python_executable": sys.executable,
            "model_dir": str(self.model_dir),
            "device": "cpu",
            "startup_timeout_s": 5.0,
            "request_timeout_s": 5.0,
            "shutdown_timeout_s": 2.0,
        }
        values.update(overrides)
        return PARENT.PyannoteConfig(**values)

    def _audio(self, name: str) -> Path:
        path = self.root / name
        path.write_bytes(b"fake wav content")
        return path

    def test_config_defaults_are_configurable(self) -> None:
        config = PARENT.PyannoteConfig()
        self.assertEqual(
            config.python_executable,
            PARENT.DEFAULT_PYANNOTE_PYTHON,
        )
        self.assertEqual(
            config.model_dir,
            PARENT.DEFAULT_PYANNOTE_MODEL,
        )
        self.assertEqual(config.device, "cuda")

    def test_persistent_worker_is_reused_for_multiple_recordings(self) -> None:
        first_audio = self._audio("第一段.wav")
        second_audio = self._audio("second.wav")
        with mock.patch.object(PARENT, "_worker_script_path", return_value=self.worker_script):
            diarizer = PARENT.PyannoteDiarizer(self._config())
            self.addCleanup(diarizer.close)

            first = diarizer.diarize(first_audio)
            first_pid = diarizer.worker_pid
            second = diarizer.diarize(second_audio)

            self.assertEqual(first["status"], "ok")
            self.assertEqual(second["status"], "ok")
            self.assertIsNotNone(first_pid)
            self.assertEqual(first["worker_pid"], second["worker_pid"])
            self.assertEqual(diarizer.worker_pid, first_pid)
            self.assertEqual(first["turns"][0]["speaker_id"], "SPEAKER_00")
            self.assertNotIn("confidence", first["turns"][0])
            self.assertEqual(first["turns"][0]["confidence_source"], "unavailable")
            self.assertEqual(first["exclusive_turns"][0]["speaker_id"], "SPEAKER_00")
            self.assertEqual(first["exclusive_source"], "pyannote_exclusive")
            self.assertEqual(first["audio_path"], str(first_audio.resolve()))
            json.dumps(first, ensure_ascii=False)

            diarizer.close()
            self.assertIsNone(diarizer.worker_pid)
            diarizer.close()

    def test_context_manager_closes_worker(self) -> None:
        audio = self._audio("clip.wav")
        with mock.patch.object(PARENT, "_worker_script_path", return_value=self.worker_script):
            with PARENT.PyannoteDiarizer(self._config()) as diarizer:
                self.assertEqual(diarizer.diarize(audio)["status"], "ok")
                self.assertIsNotNone(diarizer.worker_pid)
            self.assertIsNone(diarizer.worker_pid)

    def test_worker_error_is_returned_as_structured_failure(self) -> None:
        audio = self._audio("error.wav")
        with mock.patch.object(PARENT, "_worker_script_path", return_value=self.worker_script):
            with PARENT.PyannoteDiarizer(self._config()) as diarizer:
                result = diarizer.diarize(audio)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["turns"], [])
        self.assertEqual(result["exclusive_turns"], [])
        self.assertIn("fake inference failure", result["error"])

    def test_request_timeout_kills_worker_and_next_request_restarts(self) -> None:
        timeout_audio = self._audio("timeout.wav")
        next_audio = self._audio("next.wav")
        with mock.patch.object(PARENT, "_worker_script_path", return_value=self.worker_script):
            diarizer = PARENT.PyannoteDiarizer(self._config(request_timeout_s=0.1))
            self.addCleanup(diarizer.close)

            timed_out = diarizer.diarize(timeout_audio)
            self.assertEqual(timed_out["status"], "error")
            self.assertIn("TimeoutError", timed_out["error"])
            self.assertIsNone(diarizer.worker_pid)

            recovered = diarizer.diarize(next_audio)
            self.assertEqual(recovered["status"], "ok")
            self.assertIsNotNone(diarizer.worker_pid)

    def test_startup_failure_is_returned_without_hanging(self) -> None:
        failing_model_dir = self.root / "startup_error"
        failing_model_dir.mkdir()
        audio = self._audio("clip.wav")
        with mock.patch.object(PARENT, "_worker_script_path", return_value=self.worker_script):
            diarizer = PARENT.PyannoteDiarizer(
                self._config(model_dir=str(failing_model_dir))
            )
            self.addCleanup(diarizer.close)
            result = diarizer.diarize(audio)

        self.assertEqual(result["status"], "error")
        self.assertIn("fake model load failure", result["error"])
        self.assertIsNone(diarizer.worker_pid)

    def test_missing_audio_fails_before_starting_worker(self) -> None:
        with mock.patch.object(PARENT, "_worker_script_path", return_value=self.worker_script):
            diarizer = PARENT.PyannoteDiarizer(self._config())
            self.addCleanup(diarizer.close)
            result = diarizer.diarize(self.root / "missing.wav")

        self.assertEqual(result["status"], "error")
        self.assertIn("FileNotFoundError", result["error"])
        self.assertIsNone(diarizer.worker_pid)


if __name__ == "__main__":
    unittest.main()
