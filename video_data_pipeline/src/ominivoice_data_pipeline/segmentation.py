"""Local-only audio segmentation and speaker-purity gates.

The public functions accept and return JSON-compatible dictionaries.  Times are
expressed in seconds and intervals use the half-open convention ``[start, end)``.
No model runtime is required: the VAD uses PCM energy, timeline fusion consumes
pre-computed speaker turns, and the final gate fails closed when speaker evidence
is missing or ambiguous.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import wave
from typing import Any


__all__ = [
    "clean_speaker_timeline",
    "energy_vad",
    "find_quiet_boundary_ms",
    "fuse_timelines",
    "final_single_speaker_gate",
    "prepare_speaker_timelines",
]


_EPSILON = 1e-9
_DEFAULT_VAD_CONFIG: dict[str, Any] = {
    "frame_ms": 30.0,
    "hop_ms": 10.0,
    "threshold_dbfs": None,
    "noise_quantile": 0.20,
    "speech_margin_db": 10.0,
    "min_threshold_dbfs": -55.0,
    "max_threshold_dbfs": -25.0,
    "floor_dbfs": -100.0,
    "min_speech_ms": 250.0,
    "min_silence_ms": 300.0,
    "speech_pad_ms": 100.0,
    "merge_gap_ms": 150.0,
    "target_segment_ms": 12_000.0,
    "max_segment_ms": 30_000.0,
    "split_search_ms": 2_000.0,
    "min_split_segment_ms": 1_000.0,
    "split_pause_window_ms": 180.0,
    "split_distance_penalty_db": 1.5,
}

_DEFAULT_FUSION_CONFIG: dict[str, Any] = {
    "boundary_tolerance_ms": 10.0,
    "merge_adjacent_ms": 10.0,
    "cleanup_enabled": True,
    "unknown_boundary_max_ms": 200.0,
    "same_speaker_gap_max_ms": 200.0,
    "unknown_speaker": "UNKNOWN",
    "overlap_speaker": "OVERLAP",
    "unknown_labels": ["", "UNKNOWN", "UNK", "UNASSIGNED", "NONE", "NULL"],
    "overlap_labels": ["OVERLAP", "OVL"],
}

_DEFAULT_TIMELINE_CLEANUP_CONFIG: dict[str, Any] = {
    "boundary_tolerance_ms": 10.0,
    "merge_adjacent_ms": 10.0,
    "unknown_boundary_max_ms": 200.0,
    "same_speaker_gap_max_ms": 200.0,
    "unknown_speaker": "UNKNOWN",
    "overlap_speaker": "OVERLAP",
    "unknown_labels": ["", "UNKNOWN", "UNK", "UNASSIGNED", "NONE", "NULL"],
    "overlap_labels": ["OVERLAP", "OVL"],
}

_DEFAULT_GATE_CONFIG: dict[str, Any] = {
    "min_duration_ms": 200.0,
    "min_speaker_ms": 200.0,
    "boundary_collar_ms": 50.0,
    "min_dominant_ratio": 0.90,
    "max_other_speaker_ms": 100.0,
    "max_other_speaker_ratio": 0.02,
    "max_overlap_ratio": 0.02,
    "max_unknown_ratio": 0.08,
    "max_distinct_speakers": 1,
    "unknown_labels": ["", "UNKNOWN", "UNK", "UNASSIGNED", "NONE", "NULL"],
    "overlap_labels": ["OVERLAP", "OVL"],
    # The chronologically last candidate is governed by these stricter rules.
    "tail_min_speaker_ms": 300.0,
    "tail_min_dominant_ratio": 0.92,
    "tail_max_other_speaker_ms": 100.0,
    "tail_max_other_speaker_ratio": 0.02,
    "tail_max_overlap_ratio": 0.0,
    "tail_max_unknown_ratio": 0.08,
    "tail_max_trailing_unknown_ms": 200.0,
    "tail_require_last_voice_dominant": True,
}


def _config(defaults: Mapping[str, Any], value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return dict(defaults)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - set(defaults))
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    result = dict(defaults)
    result.update(value)
    return result


def _number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _ratio(value: Any, *, name: str) -> float:
    result = _number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _first_present(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    raise KeyError(f"one of {list(names)} is required")


def _interval(row: Any, *, index: int, kind: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"{kind}[{index}] must be a JSON object")
    if "start" in row:
        start = _number(row["start"], name=f"{kind}[{index}].start", minimum=0.0)
    elif "start_ms" in row:
        start = _number(row["start_ms"], name=f"{kind}[{index}].start_ms", minimum=0.0) / 1000.0
    elif "start_time" in row:
        start = _number(row["start_time"], name=f"{kind}[{index}].start_time", minimum=0.0) / 1000.0
    else:
        raise KeyError(f"{kind}[{index}] requires start, start_ms, or start_time")
    if "end" in row:
        end = _number(row["end"], name=f"{kind}[{index}].end", minimum=0.0)
    elif "end_ms" in row:
        end = _number(row["end_ms"], name=f"{kind}[{index}].end_ms", minimum=0.0) / 1000.0
    elif "end_time" in row:
        end = _number(row["end_time"], name=f"{kind}[{index}].end_time", minimum=0.0) / 1000.0
    else:
        raise KeyError(f"{kind}[{index}] requires end, end_ms, or end_time")
    if end <= start:
        raise ValueError(f"{kind}[{index}] must have end > start")
    return {
        "start": start,
        "end": end,
        "id": str(row.get("id", row.get("segment_id", f"{kind}-{index:06d}"))),
        "row": dict(row),
        "index": index,
    }


def _dbfs(amplitude: float, floor_dbfs: float) -> float:
    if amplitude <= 0.0:
        return floor_dbfs
    return max(floor_dbfs, 20.0 * math.log10(amplitude))


def _fallback_pcm_levels(fragment: bytes, sample_width: int, channels: int) -> tuple[float, float]:
    frame_width = sample_width * channels
    frame_count = len(fragment) // frame_width
    if frame_count == 0:
        return 0.0, 0.0

    full_scale = 128.0 if sample_width == 1 else float(1 << (sample_width * 8 - 1))
    square_sum = 0.0
    peak = 0.0
    for frame_index in range(frame_count):
        frame_offset = frame_index * frame_width
        channel_sum = 0.0
        for channel in range(channels):
            offset = frame_offset + channel * sample_width
            raw = fragment[offset : offset + sample_width]
            if sample_width == 1:
                sample = raw[0] - 128
            elif sample_width == 3:
                unsigned = int.from_bytes(raw, "little", signed=False)
                sample = unsigned - (1 << 24) if unsigned & (1 << 23) else unsigned
            else:
                sample = int.from_bytes(raw, "little", signed=True)
            channel_sum += sample
        mono = channel_sum / channels
        square_sum += mono * mono
        peak = max(peak, abs(mono))
    return math.sqrt(square_sum / frame_count) / full_scale, peak / full_scale


def _pcm_levels(fragment: bytes, sample_width: int, channels: int) -> tuple[float, float]:
    """Return normalized mono RMS and peak, using a C stdlib helper when available."""

    try:
        import audioop  # type: ignore[import-not-found]
    except ImportError:  # Python 3.13+ removed audioop.
        return _fallback_pcm_levels(fragment, sample_width, channels)

    if channels not in (1, 2):
        return _fallback_pcm_levels(fragment, sample_width, channels)
    mono = fragment
    if channels == 2:
        mono = audioop.tomono(mono, sample_width, 0.5, 0.5)
    if sample_width == 1:
        # RIFF/WAVE 8-bit PCM is unsigned; audioop's width-1 arithmetic is signed.
        mono = audioop.bias(mono, 1, -128)
    full_scale = 128.0 if sample_width == 1 else float(1 << (sample_width * 8 - 1))
    return audioop.rms(mono, sample_width) / full_scale, audioop.max(mono, sample_width) / full_scale


def _read_energy_frames(
    path: Path,
    *,
    frame_ms: float,
    hop_ms: float,
    floor_dbfs: float,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    frames: list[dict[str, float]] = []
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError(f"only uncompressed PCM WAV is supported, got {wav_file.getcomptype()}")
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        total_samples = wav_file.getnframes()
        if sample_rate <= 0 or channels <= 0 or sample_width not in (1, 2, 3, 4):
            raise ValueError("unsupported or invalid WAV metadata")

        frame_samples = max(1, int(round(frame_ms * sample_rate / 1000.0)))
        hop_samples = max(1, int(round(hop_ms * sample_rate / 1000.0)))
        pcm_frame_bytes = channels * sample_width
        read_block_samples = max(sample_rate, frame_samples * 4)
        buffer = bytearray()
        buffer_first_sample = 0
        cursor_samples = 0
        eof = False

        while True:
            while len(buffer) // pcm_frame_bytes < cursor_samples + frame_samples and not eof:
                block = wav_file.readframes(read_block_samples)
                if block:
                    buffer.extend(block)
                else:
                    eof = True

            available_samples = len(buffer) // pcm_frame_bytes
            if cursor_samples >= available_samples:
                break
            end_samples = min(cursor_samples + frame_samples, available_samples)
            byte_start = cursor_samples * pcm_frame_bytes
            byte_end = end_samples * pcm_frame_bytes
            rms, peak = _pcm_levels(bytes(buffer[byte_start:byte_end]), sample_width, channels)
            absolute_start = buffer_first_sample + cursor_samples
            absolute_end = buffer_first_sample + end_samples
            frames.append(
                {
                    "start": absolute_start / sample_rate,
                    "end": absolute_end / sample_rate,
                    "rms_dbfs": _dbfs(rms, floor_dbfs),
                    "peak_dbfs": _dbfs(peak, floor_dbfs),
                }
            )
            cursor_samples += hop_samples

            # Compact the streaming buffer without retaining the whole recording.
            if cursor_samples >= read_block_samples:
                del buffer[: cursor_samples * pcm_frame_bytes]
                buffer_first_sample += cursor_samples
                cursor_samples = 0

        metadata = {
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "sample_count": total_samples,
            "duration": total_samples / sample_rate,
        }
    return frames, metadata


def find_quiet_boundary_ms(
    audio_path: str | Path,
    *,
    target_ms: int,
    lower_ms: int,
    upper_ms: int,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    pause_window_ms: float = 180.0,
    distance_penalty_db: float = 1.5,
    floor_dbfs: float = -96.0,
) -> dict[str, Any]:
    """Find a sustained low-energy pause near a requested split point.

    Only the bounded search window is read. The result is suitable for splitting
    long ASR spans that do not contain genuine word-level timestamps. Candidate
    positions are scored over a context window in the linear power domain so an
    isolated low-energy phone is not mistaken for a sentence boundary.
    """

    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    if isinstance(target_ms, bool) or isinstance(lower_ms, bool) or isinstance(upper_ms, bool):
        raise TypeError("boundary times must be integer milliseconds")
    target_ms = int(target_ms)
    lower_ms = int(lower_ms)
    upper_ms = int(upper_ms)
    if lower_ms < 0 or upper_ms <= lower_ms:
        raise ValueError("quiet-boundary search requires 0 <= lower_ms < upper_ms")
    if not lower_ms <= target_ms <= upper_ms:
        raise ValueError("target_ms must fall inside the search window")
    if frame_ms <= 0.0 or hop_ms <= 0.0 or hop_ms > frame_ms:
        raise ValueError("quiet-boundary frame/hop values are invalid")
    if pause_window_ms <= 0.0 or distance_penalty_db < 0.0:
        raise ValueError("pause window must be positive and distance penalty non-negative")

    candidates: list[dict[str, float]] = []
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError("quiet-boundary search requires uncompressed PCM WAV")
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        total_samples = wav_file.getnframes()
        if sample_rate <= 0 or channels <= 0 or sample_width not in (1, 2, 3, 4):
            raise ValueError("unsupported or invalid WAV metadata")
        duration_ms = int(round(total_samples * 1000.0 / sample_rate))
        bounded_lower = max(0, min(lower_ms, duration_ms))
        bounded_upper = max(bounded_lower, min(upper_ms, duration_ms))
        if bounded_upper <= bounded_lower:
            raise ValueError("quiet-boundary search window is outside the audio")
        frame_samples = max(1, int(round(frame_ms * sample_rate / 1000.0)))
        hop_samples = max(1, int(round(hop_ms * sample_rate / 1000.0)))
        lower_sample = int(round(bounded_lower * sample_rate / 1000.0))
        upper_sample = int(round(bounded_upper * sample_rate / 1000.0))
        wav_file.setpos(lower_sample)
        fragment = wav_file.readframes(min(total_samples - lower_sample, upper_sample - lower_sample + frame_samples))
        frame_width = channels * sample_width
        available_samples = len(fragment) // frame_width
        cursor = 0
        while cursor < available_samples:
            frame_end = min(cursor + frame_samples, available_samples)
            frame_bytes = fragment[cursor * frame_width : frame_end * frame_width]
            rms, _ = _pcm_levels(frame_bytes, sample_width, channels)
            position_ms = int(round((lower_sample + cursor) * 1000.0 / sample_rate))
            if position_ms > bounded_upper:
                break
            candidates.append({"position_ms": position_ms, "rms_dbfs": _dbfs(rms, floor_dbfs)})
            cursor += hop_samples
    if not candidates:
        return {
            "split_ms": target_ms,
            "rms_dbfs": None,
            "method": "target_fallback",
            "target_ms": target_ms,
            "search_start_ms": lower_ms,
            "search_end_ms": upper_ms,
        }
    half_window_ms = pause_window_ms / 2.0

    def score(frame: Mapping[str, float]) -> tuple[float, float]:
        position_ms = float(frame["position_ms"])
        context = [
            float(other["rms_dbfs"])
            for other in candidates
            if abs(float(other["position_ms"]) - position_ms) <= half_window_ms
        ]
        mean_power = sum(10.0 ** (value / 10.0) for value in context) / max(1, len(context))
        sustained_dbfs = 10.0 * math.log10(max(mean_power, 1e-12))
        normalized_distance = abs(position_ms - target_ms) / max(1.0, upper_ms - lower_ms)
        return sustained_dbfs + distance_penalty_db * normalized_distance, abs(position_ms - target_ms)

    quietest = min(candidates, key=score)
    quietest_score, _ = score(quietest)
    return {
        "split_ms": int(quietest["position_ms"]),
        "rms_dbfs": float(quietest["rms_dbfs"]),
        "sustained_score_dbfs": round(float(quietest_score), 3),
        "pause_window_ms": float(pause_window_ms),
        "distance_penalty_db": float(distance_penalty_db),
        "method": "sustained_pause_rms",
        "target_ms": target_ms,
        "search_start_ms": lower_ms,
        "search_end_ms": upper_ms,
    }


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bridge_short_silences(flags: list[bool], max_gap_frames: int) -> list[bool]:
    if max_gap_frames <= 0:
        return flags[:]
    result = flags[:]
    index = 0
    while index < len(flags):
        if flags[index]:
            index += 1
            continue
        end = index
        while end < len(flags) and not flags[end]:
            end += 1
        enclosed = index > 0 and end < len(flags) and flags[index - 1] and flags[end]
        if enclosed and end - index <= max_gap_frames:
            result[index:end] = [True] * (end - index)
        index = end
    return result


def _merge_intervals(intervals: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    result = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = result[-1]
        if start - previous_end <= max_gap + _EPSILON:
            result[-1] = (previous_start, max(previous_end, end))
        else:
            result.append((start, end))
    return result


def _split_long_interval(
    start: float,
    end: float,
    *,
    target_duration: float,
    max_duration: float,
    search_radius: float,
    min_piece: float,
    pause_window: float,
    distance_penalty_db: float,
    frames: Sequence[Mapping[str, float]],
) -> list[tuple[float, float]]:
    if max_duration <= 0.0 or end - start <= max_duration + _EPSILON:
        return [(start, end)]
    if target_duration <= 0.0 or target_duration > max_duration:
        raise ValueError("target split duration must be positive and no greater than max duration")
    if pause_window <= 0.0 or distance_penalty_db < 0.0:
        raise ValueError("pause window must be positive and distance penalty non-negative")

    frame_positions = [float(frame["start"]) for frame in frames]
    power_prefix = [0.0]
    for frame in frames:
        power_prefix.append(power_prefix[-1] + 10.0 ** (float(frame["rms_dbfs"]) / 10.0))

    def sustained_dbfs(position: float) -> float:
        half_window = pause_window / 2.0
        left = bisect_left(frame_positions, position - half_window)
        right = bisect_right(frame_positions, position + half_window)
        if right <= left:
            return 0.0
        # Average in the linear power domain so a single very quiet frame inside
        # a voiced phone cannot masquerade as a genuine inter-word pause.
        mean_power = (power_prefix[right] - power_prefix[left]) / (right - left)
        return 10.0 * math.log10(max(mean_power, 1e-12))

    result: list[tuple[float, float]] = []
    cursor = start
    while end - cursor > max_duration + _EPSILON:
        target = cursor + target_duration
        lower = max(cursor + min_piece, target - search_radius)
        upper = min(cursor + max_duration, end - min_piece, target + search_radius)
        candidates = [
            frame
            for frame in frames
            if lower <= float(frame["start"]) <= upper
        ]
        if candidates:
            quietest = min(
                candidates,
                key=lambda frame: (
                    sustained_dbfs(float(frame["start"]))
                    + distance_penalty_db
                    * abs(float(frame["start"]) - target)
                    / max(search_radius, _EPSILON),
                    abs(float(frame["start"]) - target),
                ),
            )
            split = float(quietest["start"])
        else:
            split = min(target, end - min_piece)
        if split <= cursor + _EPSILON:
            split = min(target, end)
        result.append((cursor, split))
        cursor = split
    if end - cursor > _EPSILON:
        result.append((cursor, end))
    return result


def energy_vad(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run adaptive energy VAD and return coarse speech intervals.

    Input::

        {"audio_path": "/path/to/pcm.wav", "config": {...}}

    ``threshold_dbfs`` may be fixed.  When it is null, the threshold is the
    configured lower energy quantile plus ``speech_margin_db``, clamped between
    ``min_threshold_dbfs`` and ``max_threshold_dbfs``.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a JSON object")
    if "audio_path" not in payload:
        raise KeyError("audio_path is required")
    audio_path = Path(str(payload["audio_path"])).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    config = _config(_DEFAULT_VAD_CONFIG, payload.get("config"), name="VAD config")

    frame_ms = _number(config["frame_ms"], name="frame_ms", minimum=_EPSILON)
    hop_ms = _number(config["hop_ms"], name="hop_ms", minimum=_EPSILON)
    floor_dbfs = _number(config["floor_dbfs"], name="floor_dbfs")
    if hop_ms > frame_ms:
        raise ValueError("hop_ms must be <= frame_ms")
    noise_quantile = _ratio(config["noise_quantile"], name="noise_quantile")
    speech_margin_db = _number(config["speech_margin_db"], name="speech_margin_db", minimum=0.0)
    min_threshold = _number(config["min_threshold_dbfs"], name="min_threshold_dbfs")
    max_threshold = _number(config["max_threshold_dbfs"], name="max_threshold_dbfs")
    if min_threshold > max_threshold:
        raise ValueError("min_threshold_dbfs must be <= max_threshold_dbfs")
    for key in (
        "min_speech_ms",
        "min_silence_ms",
        "speech_pad_ms",
        "merge_gap_ms",
        "target_segment_ms",
        "max_segment_ms",
        "split_search_ms",
        "min_split_segment_ms",
        "split_pause_window_ms",
        "split_distance_penalty_db",
    ):
        config[key] = _number(config[key], name=key, minimum=0.0)

    frames, audio = _read_energy_frames(
        audio_path,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        floor_dbfs=floor_dbfs,
    )
    if not frames:
        threshold = float(config["threshold_dbfs"] or max_threshold)
        noise_floor = floor_dbfs
    else:
        noise_floor = _quantile([frame["rms_dbfs"] for frame in frames], noise_quantile)
        configured_threshold = config["threshold_dbfs"]
        if configured_threshold is None:
            threshold = min(max_threshold, max(min_threshold, noise_floor + speech_margin_db))
        else:
            threshold = _number(configured_threshold, name="threshold_dbfs")

    flags = [frame["rms_dbfs"] >= threshold for frame in frames]
    max_gap_frames = int(math.floor(float(config["min_silence_ms"]) / hop_ms))
    flags = _bridge_short_silences(flags, max_gap_frames)

    intervals: list[tuple[float, float]] = []
    index = 0
    min_speech_seconds = float(config["min_speech_ms"]) / 1000.0
    pad_seconds = float(config["speech_pad_ms"]) / 1000.0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        end_index = index + 1
        while end_index < len(flags) and flags[end_index]:
            end_index += 1
        start = frames[index]["start"]
        end = frames[end_index - 1]["end"]
        if end - start + _EPSILON >= min_speech_seconds:
            intervals.append((max(0.0, start - pad_seconds), min(float(audio["duration"]), end + pad_seconds)))
        index = end_index

    intervals = _merge_intervals(intervals, float(config["merge_gap_ms"]) / 1000.0)
    max_segment_seconds = float(config["max_segment_ms"]) / 1000.0
    target_segment_seconds = min(float(config["target_segment_ms"]) / 1000.0, max_segment_seconds)
    if target_segment_seconds <= 0.0:
        raise ValueError("target_segment_ms and max_segment_ms must be positive")
    split_intervals: list[tuple[float, float]] = []
    for start, end in intervals:
        split_intervals.extend(
            _split_long_interval(
                start,
                end,
                target_duration=target_segment_seconds,
                max_duration=max_segment_seconds,
                search_radius=float(config["split_search_ms"]) / 1000.0,
                min_piece=float(config["min_split_segment_ms"]) / 1000.0,
                pause_window=float(config["split_pause_window_ms"]) / 1000.0,
                distance_penalty_db=float(config["split_distance_penalty_db"]),
                frames=frames,
            )
        )

    segments: list[dict[str, Any]] = []
    for segment_index, (start, end) in enumerate(split_intervals):
        contributing = [
            frame
            for frame in frames
            if frame["end"] > start + _EPSILON and frame["start"] < end - _EPSILON
        ]
        mean_dbfs = (
            sum(float(frame["rms_dbfs"]) for frame in contributing) / len(contributing)
            if contributing
            else floor_dbfs
        )
        peak_dbfs = max((float(frame["peak_dbfs"]) for frame in contributing), default=floor_dbfs)
        segments.append(
            {
                "id": f"vad-{segment_index:06d}",
                "start": round(start, 6),
                "end": round(end, 6),
                "start_ms": int(round(start * 1000.0)),
                "end_ms": int(round(end * 1000.0)),
                "start_time": int(round(start * 1000.0)),
                "end_time": int(round(end * 1000.0)),
                "duration": round(end - start, 6),
                "duration_ms": int(round((end - start) * 1000.0)),
                "source": "energy_vad",
                "mean_dbfs": round(mean_dbfs, 3),
                "peak_dbfs": round(peak_dbfs, 3),
                "is_last": segment_index == len(split_intervals) - 1,
            }
        )

    return {
        "schema_version": "ominivoice.energy-vad.v1",
        "audio_path": str(audio_path.resolve()),
        "audio": audio,
        "config": config,
        "analysis": {
            "frame_count": len(frames),
            "noise_floor_dbfs": round(noise_floor, 3),
            "threshold_dbfs": round(threshold, 3),
            "raw_speech_frame_count": sum(frame["rms_dbfs"] >= threshold for frame in frames),
            "speech_frame_count_after_hangover": sum(flags),
        },
        "segments": segments,
    }


def _label_set(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        return tuple(str(value) for value in values if value is not None)
    raise TypeError("speaker_ids must be a string or JSON array")


def _speaker_turns(rows: Any, *, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    unknown = {str(value).strip().upper() for value in _sequence(config["unknown_labels"], name="unknown_labels")}
    overlap = {str(value).strip().upper() for value in _sequence(config["overlap_labels"], name="overlap_labels")}
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(_sequence(rows, name="speaker_segments")):
        turn = _interval(raw, index=index, kind="speaker_segments")
        row = turn["row"]
        state = str(row.get("speaker_state", row.get("state", ""))).strip().lower()
        if "speaker_ids" in row:
            labels = _label_set(row["speaker_ids"])
        else:
            label = row.get("speaker_id", row.get("speaker", row.get("label")))
            labels = _label_set(label)
        labels = tuple(label for label in labels if label.strip().upper() not in unknown)
        forced_overlap = state == "overlap" or any(label.strip().upper() in overlap for label in labels)
        labels = tuple(label for label in labels if label.strip().upper() not in overlap)
        if state in {"unknown", "unassigned"}:
            labels = ()
        turn["labels"] = tuple(sorted(set(labels)))
        turn["forced_overlap"] = forced_overlap
        result.append(turn)
    result.sort(key=lambda row: (row["start"], row["end"], row["id"]))
    return result


def _coalesce_boundaries(values: Sequence[float], tolerance: float, start: float, end: float) -> list[float]:
    ordered = sorted(max(start, min(end, value)) for value in values)
    groups: list[list[float]] = []
    for value in ordered:
        if not groups or value - groups[-1][-1] > tolerance + _EPSILON:
            groups.append([value])
        else:
            groups[-1].append(value)
    result: list[float] = []
    for group in groups:
        if any(abs(value - start) <= _EPSILON for value in group):
            result.append(start)
        elif any(abs(value - end) <= _EPSILON for value in group):
            result.append(end)
        else:
            result.append(sum(group) / len(group))
    if not result or abs(result[0] - start) > _EPSILON:
        result.insert(0, start)
    if abs(result[-1] - end) > _EPSILON:
        result.append(end)
    return sorted(set(result))


def _atoms_for_interval(
    interval: Mapping[str, Any],
    turns: Sequence[Mapping[str, Any]],
    *,
    boundary_tolerance: float,
) -> list[dict[str, Any]]:
    start = float(interval["start"])
    end = float(interval["end"])
    relevant = [
        turn
        for turn in turns
        if float(turn["end"]) > start + _EPSILON and float(turn["start"]) < end - _EPSILON
    ]
    boundaries = [start, end]
    for turn in relevant:
        boundaries.extend((max(start, float(turn["start"])), min(end, float(turn["end"]))))
    boundaries = _coalesce_boundaries(boundaries, boundary_tolerance, start, end)

    atoms: list[dict[str, Any]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left <= _EPSILON:
            continue
        midpoint = (left + right) / 2.0
        active = [
            turn
            for turn in relevant
            if float(turn["start"]) <= midpoint + _EPSILON and float(turn["end"]) > midpoint + _EPSILON
        ]
        labels = tuple(sorted({label for turn in active for label in turn["labels"]}))
        forced_overlap = any(bool(turn["forced_overlap"]) for turn in active)
        if not active or (not labels and not forced_overlap):
            state = "unknown"
        elif forced_overlap or len(labels) > 1:
            state = "overlap"
        else:
            state = "single"
        atoms.append(
            {
                "start": left,
                "end": right,
                "speaker_state": state,
                "speaker_ids": list(labels),
                "source_speaker_segment_ids": [str(turn["id"]) for turn in active],
            }
        )
    return atoms


def _merge_atoms(atoms: Sequence[Mapping[str, Any]], max_gap: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for atom in atoms:
        current = dict(atom)
        if (
            result
            and current["speaker_state"] == result[-1]["speaker_state"]
            and current["speaker_ids"] == result[-1]["speaker_ids"]
            and float(current["start"]) - float(result[-1]["end"]) <= max_gap + _EPSILON
        ):
            result[-1]["end"] = current["end"]
            result[-1]["source_speaker_segment_ids"] = sorted(
                set(result[-1]["source_speaker_segment_ids"]) | set(current["source_speaker_segment_ids"])
            )
        else:
            result.append(current)
    return result


def _clean_atoms(
    atoms: Sequence[Mapping[str, Any]],
    *,
    unknown_boundary_max: float,
    same_speaker_gap_max: float,
    merge_gap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Clean harmless unknown slivers while leaving overlap atoms untouched."""

    working = _merge_atoms(atoms, 0.0)
    actions: list[dict[str, Any]] = []
    last_index = len(working) - 1
    for index, atom in enumerate(working):
        if atom["speaker_state"] != "unknown":
            continue
        duration = float(atom["end"]) - float(atom["start"])
        left = working[index - 1] if index > 0 else None
        right = working[index + 1] if index < last_index else None
        replacement: str | None = None
        action_type: str | None = None
        if (
            left is not None
            and right is not None
            and left["speaker_state"] == "single"
            and right["speaker_state"] == "single"
            and left["speaker_ids"] == right["speaker_ids"]
            and len(left["speaker_ids"]) == 1
            and duration <= same_speaker_gap_max + _EPSILON
        ):
            replacement = str(left["speaker_ids"][0])
            action_type = "absorb_same_speaker_gap"
        elif duration <= unknown_boundary_max + _EPSILON:
            neighbor = right if left is None else left if right is None else None
            if (
                neighbor is not None
                and neighbor["speaker_state"] == "single"
                and len(neighbor["speaker_ids"]) == 1
            ):
                replacement = str(neighbor["speaker_ids"][0])
                action_type = "absorb_unknown_boundary"
        if replacement is None or action_type is None:
            continue
        atom["speaker_state"] = "single"
        atom["speaker_ids"] = [replacement]
        actions.append(
            {
                "action": action_type,
                "start_ms": int(round(float(atom["start"]) * 1000.0)),
                "end_ms": int(round(float(atom["end"]) * 1000.0)),
                "duration_ms": int(round(duration * 1000.0)),
                "speaker_id": replacement,
            }
        )
    return _merge_atoms(working, merge_gap), actions


def _optional_payload_time(payload: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        if name not in payload:
            continue
        value = _number(payload[name], name=name, minimum=0.0)
        return value / 1000.0 if name.endswith("_ms") or name.endswith("_time") else value
    return None


def _render_timeline_atom(
    atom: Mapping[str, Any],
    *,
    index: int,
    unknown_speaker: str,
    overlap_speaker: str,
) -> dict[str, Any]:
    start = float(atom["start"])
    end = float(atom["end"])
    state = str(atom["speaker_state"])
    labels = [str(label) for label in atom["speaker_ids"]]
    speaker = labels[0] if state == "single" else overlap_speaker if state == "overlap" else unknown_speaker
    return {
        "id": f"speaker-clean-{index:06d}",
        "start": round(start, 6),
        "end": round(end, 6),
        "start_ms": int(round(start * 1000.0)),
        "end_ms": int(round(end * 1000.0)),
        "start_time": int(round(start * 1000.0)),
        "end_time": int(round(end * 1000.0)),
        "duration": round(end - start, 6),
        "duration_ms": int(round((end - start) * 1000.0)),
        "speaker": speaker,
        "speaker_id": labels[0] if state == "single" else None,
        "speaker_ids": labels,
        "speaker_state": state,
        "source_speaker_segment_ids": list(atom["source_speaker_segment_ids"]),
    }


def clean_speaker_timeline(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize raw speaker turns and absorb only bounded unknown slivers.

    Input accepts ``speaker_segments``, ``speaker_timeline``, ``turns`` or
    ``segments``.  Optional timeline bounds may be supplied as
    ``timeline_start[_ms]`` and ``timeline_end[_ms]``.  Short unknown edges and
    short gaps surrounded by the same speaker are absorbed; different-speaker
    boundaries and true overlap are always retained.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a JSON object")
    config = _config(
        _DEFAULT_TIMELINE_CLEANUP_CONFIG,
        payload.get("config"),
        name="timeline cleanup config",
    )
    for key in (
        "boundary_tolerance_ms",
        "merge_adjacent_ms",
        "unknown_boundary_max_ms",
        "same_speaker_gap_max_ms",
    ):
        config[key] = _number(config[key], name=key, minimum=0.0)
    speaker_rows = _first_present(payload, ("speaker_segments", "speaker_timeline", "turns", "segments"))
    turns = _speaker_turns(speaker_rows, config=config)
    start = _optional_payload_time(
        payload,
        ("timeline_start", "timeline_start_ms", "timeline_start_time", "start", "start_ms", "start_time"),
    )
    end = _optional_payload_time(
        payload,
        ("timeline_end", "timeline_end_ms", "timeline_end_time", "end", "end_ms", "end_time"),
    )
    if start is None and turns:
        start = min(float(turn["start"]) for turn in turns)
    if end is None and turns:
        end = max(float(turn["end"]) for turn in turns)
    if start is None and end is None:
        rendered: list[dict[str, Any]] = []
        return {
            "schema_version": "ominivoice.speaker-timeline-cleanup.v1",
            "config": config,
            "segments": rendered,
            "turns": [],
            "cleanup_actions": [],
            "summary": {"input_turn_count": 0, "output_segment_count": 0, "cleanup_action_count": 0},
        }
    if start is None or end is None:
        raise ValueError("both timeline start and end are required when no speaker turns provide the missing bound")
    if end <= start:
        raise ValueError("timeline end must be greater than timeline start")

    interval = {"start": start, "end": end}
    atoms = _atoms_for_interval(
        interval,
        turns,
        boundary_tolerance=float(config["boundary_tolerance_ms"]) / 1000.0,
    )
    cleaned, actions = _clean_atoms(
        atoms,
        unknown_boundary_max=float(config["unknown_boundary_max_ms"]) / 1000.0,
        same_speaker_gap_max=float(config["same_speaker_gap_max_ms"]) / 1000.0,
        merge_gap=float(config["merge_adjacent_ms"]) / 1000.0,
    )
    rendered = [
        _render_timeline_atom(
            atom,
            index=index,
            unknown_speaker=str(config["unknown_speaker"]),
            overlap_speaker=str(config["overlap_speaker"]),
        )
        for index, atom in enumerate(cleaned)
    ]
    for index, row in enumerate(rendered):
        row["is_last"] = index == len(rendered) - 1
    duration_by_state = {"single": 0.0, "overlap": 0.0, "unknown": 0.0}
    for row in rendered:
        duration_by_state[str(row["speaker_state"])] += float(row["duration"])
    return {
        "schema_version": "ominivoice.speaker-timeline-cleanup.v1",
        "config": config,
        "segments": rendered,
        "turns": [dict(row) for row in rendered],
        "cleanup_actions": actions,
        "summary": {
            "input_turn_count": len(turns),
            "output_segment_count": len(rendered),
            "cleanup_action_count": len(actions),
            "duration_by_state": {key: round(value, 6) for key, value in duration_by_state.items()},
        },
    }


def prepare_speaker_timelines(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare distinct assignment and overlap timelines from pyannote output.

    ``exclusive_turns`` is cleaned into ``assignment_timeline`` and is the only
    source intended for unique speaker labels.  Regular ``turns`` is cleaned
    separately into ``regular_timeline``; its true multi-speaker atoms are also
    exposed directly as ``overlap_intervals`` for hard-gate decisions.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a JSON object")
    regular_rows = _first_present(payload, ("regular_turns", "turns"))
    exclusive_rows = _first_present(payload, ("exclusive_turns", "assignment_turns"))
    shared: dict[str, Any] = {}
    if "config" in payload:
        shared["config"] = payload["config"]
    for key in (
        "timeline_start",
        "timeline_start_ms",
        "timeline_start_time",
        "timeline_end",
        "timeline_end_ms",
        "timeline_end_time",
        "start",
        "start_ms",
        "start_time",
        "end",
        "end_ms",
        "end_time",
    ):
        if key in payload:
            shared[key] = payload[key]

    assignment = clean_speaker_timeline({**shared, "turns": exclusive_rows})
    regular = clean_speaker_timeline({**shared, "turns": regular_rows})
    assignment_timeline = [dict(row) for row in assignment["segments"]]
    regular_timeline = [dict(row) for row in regular["segments"]]
    overlap_intervals = [
        dict(row) for row in regular_timeline if row["speaker_state"] == "overlap"
    ]
    assignment_overlap_intervals = [
        dict(row) for row in assignment_timeline if row["speaker_state"] == "overlap"
    ]
    return {
        "schema_version": "ominivoice.prepared-speaker-timelines.v1",
        "config": assignment["config"],
        "assignment_timeline": assignment_timeline,
        "regular_timeline": regular_timeline,
        "overlap_intervals": overlap_intervals,
        "assignment_valid": not assignment_overlap_intervals,
        "assignment_overlap_intervals": assignment_overlap_intervals,
        "cleanup_actions": {
            "assignment": list(assignment["cleanup_actions"]),
            "regular": list(regular["cleanup_actions"]),
        },
        "summary": {
            "assignment_segment_count": len(assignment_timeline),
            "regular_segment_count": len(regular_timeline),
            "overlap_interval_count": len(overlap_intervals),
            "assignment_overlap_interval_count": len(assignment_overlap_intervals),
        },
    }


def fuse_timelines(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Intersect VAD/base segments with a speaker timeline.

    Speaker boundaries split each VAD interval into atomic ``single``, ``overlap``
    or ``unknown`` pieces.  By default bounded unknown edges and same-speaker
    gaps are cleaned, while different speakers and true overlaps stay explicit.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a JSON object")
    config = _config(_DEFAULT_FUSION_CONFIG, payload.get("config"), name="fusion config")
    boundary_tolerance = _number(
        config["boundary_tolerance_ms"], name="boundary_tolerance_ms", minimum=0.0
    ) / 1000.0
    merge_gap = _number(config["merge_adjacent_ms"], name="merge_adjacent_ms", minimum=0.0) / 1000.0
    if not isinstance(config["cleanup_enabled"], bool):
        raise TypeError("cleanup_enabled must be boolean")
    unknown_boundary_max = _number(
        config["unknown_boundary_max_ms"], name="unknown_boundary_max_ms", minimum=0.0
    ) / 1000.0
    same_speaker_gap_max = _number(
        config["same_speaker_gap_max_ms"], name="same_speaker_gap_max_ms", minimum=0.0
    ) / 1000.0
    unknown_speaker = str(config["unknown_speaker"])
    overlap_speaker = str(config["overlap_speaker"])
    base_rows = _first_present(payload, ("vad_segments", "base_segments", "segments"))
    speaker_rows = _first_present(payload, ("speaker_segments", "speaker_timeline", "turns"))
    bases = [
        _interval(row, index=index, kind="vad_segments")
        for index, row in enumerate(_sequence(base_rows, name="vad_segments"))
    ]
    bases.sort(key=lambda row: (row["start"], row["end"], row["id"]))
    turns = _speaker_turns(speaker_rows, config=config)

    fused: list[dict[str, Any]] = []
    cleanup_actions: list[dict[str, Any]] = []
    for base in bases:
        atoms = _merge_atoms(
            _atoms_for_interval(base, turns, boundary_tolerance=boundary_tolerance),
            merge_gap,
        )
        if config["cleanup_enabled"]:
            atoms, base_actions = _clean_atoms(
                atoms,
                unknown_boundary_max=unknown_boundary_max,
                same_speaker_gap_max=same_speaker_gap_max,
                merge_gap=merge_gap,
            )
            for action in base_actions:
                action["vad_segment_id"] = base["id"]
            cleanup_actions.extend(base_actions)
        for atom in atoms:
            state = str(atom["speaker_state"])
            labels = list(atom["speaker_ids"])
            if state == "single":
                speaker = labels[0]
            elif state == "overlap":
                speaker = overlap_speaker
            else:
                speaker = unknown_speaker
            fused.append(
                {
                    "id": f"fused-{len(fused):06d}",
                    "vad_segment_id": base["id"],
                    "start": round(float(atom["start"]), 6),
                    "end": round(float(atom["end"]), 6),
                    "start_ms": int(round(float(atom["start"]) * 1000.0)),
                    "end_ms": int(round(float(atom["end"]) * 1000.0)),
                    "start_time": int(round(float(atom["start"]) * 1000.0)),
                    "end_time": int(round(float(atom["end"]) * 1000.0)),
                    "duration": round(float(atom["end"]) - float(atom["start"]), 6),
                    "duration_ms": int(round((float(atom["end"]) - float(atom["start"])) * 1000.0)),
                    "speaker": speaker,
                    "speaker_ids": labels,
                    "speaker_state": state,
                    "source_speaker_segment_ids": atom["source_speaker_segment_ids"],
                }
            )

    duration_by_state = {"single": 0.0, "overlap": 0.0, "unknown": 0.0}
    for row in fused:
        duration_by_state[row["speaker_state"]] += float(row["duration"])
    for index, row in enumerate(fused):
        row["is_last"] = index == len(fused) - 1
    return {
        "schema_version": "ominivoice.timeline-fusion.v1",
        "config": config,
        "segments": fused,
        "cleanup_actions": cleanup_actions,
        "summary": {
            "vad_segment_count": len(bases),
            "speaker_segment_count": len(turns),
            "fused_segment_count": len(fused),
            "cleanup_action_count": len(cleanup_actions),
            "duration_by_state": {key: round(value, 6) for key, value in duration_by_state.items()},
        },
    }


def _gate_metrics(atoms: Sequence[Mapping[str, Any]], duration: float) -> dict[str, Any]:
    exclusive: dict[str, float] = {}
    overlap_duration = 0.0
    unknown_duration = 0.0
    seen_speakers: set[str] = set()
    for atom in atoms:
        atom_duration = float(atom["end"]) - float(atom["start"])
        labels = [str(label) for label in atom["speaker_ids"]]
        seen_speakers.update(labels)
        if atom["speaker_state"] == "single" and len(labels) == 1:
            exclusive[labels[0]] = exclusive.get(labels[0], 0.0) + atom_duration
        elif atom["speaker_state"] == "overlap":
            overlap_duration += atom_duration
        else:
            unknown_duration += atom_duration

    dominant = max(exclusive, key=lambda label: (exclusive[label], label), default=None)
    dominant_duration = exclusive.get(dominant, 0.0) if dominant is not None else 0.0
    other_duration = sum(value for label, value in exclusive.items() if label != dominant)
    trailing_unknown = 0.0
    for atom in reversed(atoms):
        if atom["speaker_state"] != "unknown":
            break
        trailing_unknown += float(atom["end"]) - float(atom["start"])
    voiced_atoms = [atom for atom in atoms if atom["speaker_state"] != "unknown"]
    last_voice_state = voiced_atoms[-1]["speaker_state"] if voiced_atoms else None
    last_voice_speakers = list(voiced_atoms[-1]["speaker_ids"]) if voiced_atoms else []
    return {
        "dominant_speaker": dominant,
        "distinct_speakers": sorted(seen_speakers),
        "exclusive_duration_by_speaker": dict(sorted(exclusive.items())),
        "dominant_speaker_duration": dominant_duration,
        "dominant_ratio": dominant_duration / duration,
        "other_speaker_duration": other_duration,
        "other_speaker_ratio": other_duration / duration,
        "overlap_duration": overlap_duration,
        "overlap_ratio": overlap_duration / duration,
        "unknown_duration": unknown_duration,
        "unknown_ratio": unknown_duration / duration,
        "trailing_unknown_duration": trailing_unknown,
        "last_voice_state": last_voice_state,
        "last_voice_speakers": last_voice_speakers,
    }


def _tail_index(candidates: Sequence[Mapping[str, Any]]) -> int | None:
    explicit = [index for index, candidate in enumerate(candidates) if candidate["row"].get("is_tail") is True]
    if len(explicit) > 1:
        raise ValueError("at most one candidate segment may set is_tail=true")
    if explicit:
        return explicit[0]
    if not candidates:
        return None
    return max(
        range(len(candidates)),
        key=lambda index: (candidates[index]["end"], candidates[index]["start"], candidates[index]["index"]),
    )


def final_single_speaker_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a hard, fail-closed single-speaker gate to final candidate segments.

    Input requires ``segments`` plus ``speaker_segments`` (or
    ``speaker_timeline``).  The last segment is selected chronologically unless
    exactly one candidate declares ``is_tail: true``.  A tail speaker is never
    inherited from a previous segment: explicit diarization coverage is required,
    and a long unknown suffix fails the tail rule.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a JSON object")
    config = _config(_DEFAULT_GATE_CONFIG, payload.get("config"), name="gate config")
    for key in (
        "min_duration_ms",
        "min_speaker_ms",
        "boundary_collar_ms",
        "max_other_speaker_ms",
        "tail_min_speaker_ms",
        "tail_max_other_speaker_ms",
        "tail_max_trailing_unknown_ms",
    ):
        config[key] = _number(config[key], name=key, minimum=0.0)
    for key in (
        "min_dominant_ratio",
        "max_other_speaker_ratio",
        "max_overlap_ratio",
        "max_unknown_ratio",
        "tail_min_dominant_ratio",
        "tail_max_other_speaker_ratio",
        "tail_max_overlap_ratio",
        "tail_max_unknown_ratio",
    ):
        config[key] = _ratio(config[key], name=key)
    max_speakers = config["max_distinct_speakers"]
    if isinstance(max_speakers, bool) or not isinstance(max_speakers, int) or max_speakers < 1:
        raise ValueError("max_distinct_speakers must be a positive integer")
    if not isinstance(config["tail_require_last_voice_dominant"], bool):
        raise TypeError("tail_require_last_voice_dominant must be boolean")

    candidate_rows = _first_present(payload, ("segments", "candidate_segments"))
    speaker_rows = _first_present(payload, ("speaker_segments", "speaker_timeline", "turns"))
    candidates = [
        _interval(row, index=index, kind="segments")
        for index, row in enumerate(_sequence(candidate_rows, name="segments"))
    ]
    turns = _speaker_turns(speaker_rows, config=config)
    tail_index = _tail_index(candidates)

    annotated: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        is_tail = index == tail_index
        duration = float(candidate["end"]) - float(candidate["start"])
        minimum_speaker_ms = float(config["tail_min_speaker_ms"] if is_tail else config["min_speaker_ms"])
        requested_collar = float(config["boundary_collar_ms"]) / 1000.0
        # Never let a collar alone reduce an otherwise long-enough candidate
        # below its required diarization evidence duration.
        minimum_evaluation = max(minimum_speaker_ms / 1000.0, min(duration, 0.001))
        maximum_collar = max(0.0, (duration - minimum_evaluation) / 2.0)
        applied_collar = min(requested_collar, maximum_collar)
        evaluation = {
            "start": float(candidate["start"]) + applied_collar,
            "end": float(candidate["end"]) - applied_collar,
        }
        evaluation_duration = float(evaluation["end"]) - float(evaluation["start"])
        atoms = _atoms_for_interval(evaluation, turns, boundary_tolerance=0.0)
        metrics = _gate_metrics(atoms, evaluation_duration)
        reasons: list[str] = []
        if duration * 1000.0 + _EPSILON < float(config["min_duration_ms"]):
            reasons.append("duration_too_short")
        if metrics["dominant_speaker"] is None:
            reasons.append("tail_no_explicit_speaker" if is_tail else "no_explicit_speaker")
        elif metrics["dominant_speaker_duration"] * 1000.0 + _EPSILON < minimum_speaker_ms:
            reasons.append("tail_speaker_evidence_too_short" if is_tail else "speaker_evidence_too_short")

        prefix = "tail_" if is_tail else ""
        minimum_dominant = float(config["tail_min_dominant_ratio"] if is_tail else config["min_dominant_ratio"])
        maximum_other_ms = float(
            config["tail_max_other_speaker_ms"] if is_tail else config["max_other_speaker_ms"]
        )
        maximum_other = float(
            config["tail_max_other_speaker_ratio"] if is_tail else config["max_other_speaker_ratio"]
        )
        maximum_overlap = float(config["tail_max_overlap_ratio"] if is_tail else config["max_overlap_ratio"])
        maximum_unknown = float(config["tail_max_unknown_ratio"] if is_tail else config["max_unknown_ratio"])
        significant_other_speakers: list[str] = []
        tolerated_other_speakers: list[str] = []
        for speaker_id in metrics["distinct_speakers"]:
            if speaker_id == metrics["dominant_speaker"]:
                continue
            speaker_duration = float(metrics["exclusive_duration_by_speaker"].get(speaker_id, 0.0))
            speaker_ratio = speaker_duration / evaluation_duration
            if (
                speaker_duration * 1000.0 > maximum_other_ms + _EPSILON
                and speaker_ratio > maximum_other + _EPSILON
            ):
                significant_other_speakers.append(speaker_id)
            else:
                tolerated_other_speakers.append(speaker_id)
        effective_distinct_speakers = (
            (1 if metrics["dominant_speaker"] is not None else 0) + len(significant_other_speakers)
        )
        if effective_distinct_speakers > max_speakers:
            reasons.append("multiple_speakers")
        if metrics["dominant_ratio"] + _EPSILON < minimum_dominant:
            reasons.append(f"{prefix}dominant_coverage_below_min")
        if (
            metrics["other_speaker_duration"] * 1000.0 > maximum_other_ms + _EPSILON
            and metrics["other_speaker_ratio"] > maximum_other + _EPSILON
        ):
            reasons.append(f"{prefix}other_speaker_ratio_exceeded")
        if metrics["overlap_ratio"] > maximum_overlap + _EPSILON:
            reasons.append(f"{prefix}overlap_ratio_exceeded")
        if metrics["unknown_ratio"] > maximum_unknown + _EPSILON:
            reasons.append(f"{prefix}unknown_ratio_exceeded")

        if is_tail:
            if metrics["trailing_unknown_duration"] * 1000.0 > float(config["tail_max_trailing_unknown_ms"]) + _EPSILON:
                reasons.append("tail_trailing_unknown_exceeded")
            if config["tail_require_last_voice_dominant"]:
                last_voice_is_dominant = (
                    metrics["last_voice_state"] == "single"
                    and metrics["last_voice_speakers"] == [metrics["dominant_speaker"]]
                )
                last_voice_is_tolerated = (
                    metrics["last_voice_state"] == "single"
                    and len(metrics["last_voice_speakers"]) == 1
                    and metrics["last_voice_speakers"][0] in tolerated_other_speakers
                )
                if not (last_voice_is_dominant or last_voice_is_tolerated):
                    reasons.append("tail_last_voice_not_dominant")

        reasons = list(dict.fromkeys(reasons))
        row = dict(candidate["row"])
        row.update(
            {
                "id": candidate["id"],
                "start": round(float(candidate["start"]), 6),
                "end": round(float(candidate["end"]), 6),
                "start_ms": int(round(float(candidate["start"]) * 1000.0)),
                "end_ms": int(round(float(candidate["end"]) * 1000.0)),
                "start_time": int(round(float(candidate["start"]) * 1000.0)),
                "end_time": int(round(float(candidate["end"]) * 1000.0)),
                "duration": round(duration, 6),
                "duration_ms": int(round(duration * 1000.0)),
                "speaker_id": metrics["dominant_speaker"],
                "is_tail": is_tail,
                "gate": {
                    "passed": not reasons,
                    "reasons": reasons,
                    "distinct_speakers": metrics["distinct_speakers"],
                    "effective_distinct_speaker_count": effective_distinct_speakers,
                    "significant_other_speakers": significant_other_speakers,
                    "tolerated_other_speakers": tolerated_other_speakers,
                    "evaluation_start_ms": int(round(float(evaluation["start"]) * 1000.0)),
                    "evaluation_end_ms": int(round(float(evaluation["end"]) * 1000.0)),
                    "boundary_collar_ms_applied": round(applied_collar * 1000.0, 3),
                    "dominant_ratio": round(float(metrics["dominant_ratio"]), 6),
                    "dominant_speaker_duration_ms": round(
                        float(metrics["dominant_speaker_duration"]) * 1000.0, 3
                    ),
                    "other_speaker_ratio": round(float(metrics["other_speaker_ratio"]), 6),
                    "other_speaker_duration_ms": round(
                        float(metrics["other_speaker_duration"]) * 1000.0, 3
                    ),
                    "max_other_speaker_ms": maximum_other_ms,
                    "max_other_speaker_ratio": maximum_other,
                    "overlap_ratio": round(float(metrics["overlap_ratio"]), 6),
                    "unknown_ratio": round(float(metrics["unknown_ratio"]), 6),
                    "trailing_unknown_ms": round(float(metrics["trailing_unknown_duration"]) * 1000.0, 3),
                    "last_voice_state": metrics["last_voice_state"],
                    "last_voice_speakers": metrics["last_voice_speakers"],
                },
            }
        )
        annotated.append(row)
        (accepted if not reasons else rejected).append(row)

    return {
        "schema_version": "ominivoice.single-speaker-gate.v1",
        "config": config,
        "segments": annotated,
        "accepted": accepted,
        "rejected": rejected,
        "summary": {
            "total": len(annotated),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "tail_segment_id": annotated[tail_index]["id"] if tail_index is not None else None,
        },
    }
