# Real-video preprocessing reference

Selected from the project's actual `video_data_pipeline` implementation. This
subproject contains the source needed to turn local recordings into accepted
single-speaker audio clips and primary-ASR training labels. It includes no
recordings, transcripts, model files, credentials, run logs or deployment state.

## Selected processing chain

1. `media.py`: probe media, decode/resample with FFmpeg, and export waveform clips.
2. `segmentation.py`: energy VAD, quiet-boundary splitting, overlap/timeline fusion,
   and the final single-speaker gate.
3. `pyannote_diarization.py` / `pyannote_worker.py`: persistent diarization worker
   in a separately configured environment; existing speaker-turn JSON can be reused.
4. `speaker.py` / `dataset.py`: candidate grouping, local speaker attribution,
   duration constraints and manifest fields.
5. `quality.py` / `music_detection.py`: clipping/SNR/text checks and AudioSet music
   detection, retaining the selected implementation's thresholds.
6. `background_music_recovery.py` / `bs_roformer_worker.py`: BS-RoFormer separation
   and post-separation quality checks, with pinned model/config digests.
7. `gemini_asr.py` / `formal_gemini_client/`: primary Gemini 2.5 Flash transcription
   and a separate quality-only observation. The frozen primary prompt and decoding
   parameters are retained. Authentication is explicitly configured through the
   environment or CLI.
8. `pipeline.py` / `cli.py`: resumable execution, retries/exclusion records,
   final waveform export and `training_manifest.jsonl`.

`visual_voice_effect.py` retains the original broadcast-card heuristic as a
separate configurable branch; its geometry is specific to the original visual
layout. `language_profiles.py` retains helpers for Burmese, Lao and Khmer
prompt contracts. The default CLI preserves the Burmese contract. Use a separate
process and a corresponding language contract when adapting it to another language.

Cloud channel jobs, server/GPU launch scripts, migration/backup tools, object-store
publication, cookies and private proxy discovery are intentionally outside this
reference package. Input is a local video/audio file or directory.

## Environment

Install this subproject separately from the weighting package:

```bash
python -m pip install -e './video_data_pipeline[music,dev]'
```

FFmpeg and FFprobe must be on `PATH`. Prepare the pyannote
`speaker-diarization-community-1` model and its Python environment separately.
The separator uses `audio-separator==0.44.5`; an isolated environment is supported
to avoid mixing audio-library requirements. The default config enables music
recovery and requires the corresponding model to be available.

Configure these variables with your own locations, outside this source tree:

| Variable | Value supplied by the caller |
|---|---|
| `PYANNOTE_PYTHON` | Interpreter that can import pyannote |
| `PYANNOTE_MODEL_DIR` | Local pyannote model directory |
| `SEPARATOR_PYTHON` | Interpreter containing audio-separator 0.44.5 |
| `AUDIO_SEPARATOR_MODEL_DIR` | Local separator model directory |
| `GEMINI_CREDENTIALS_JSON` or `GOOGLE_APPLICATION_CREDENTIALS` | Vertex service-account JSON |
| `GEMINI_PROXY_URL` | Optional explicit proxy; omitted means direct connection |

No local ports, credential locations or model-cache locations are discovered by
scanning the caller's machine. Relative model defaults are generic placeholders.
The packaged `default_config.json` and `configs/default.json` contain the same
reference thresholds and no deployment paths.

## Run on your own inputs

The variables below refer to caller-owned input/output locations. Keep generated
data outside the reference repository.

```bash
python -m ominivoice_data_pipeline doctor
python -m ominivoice_data_pipeline run "$VIDEO_INPUT" \
  --output-dir "$DATA_WORK_DIR/video" \
  --pyannote-python "$PYANNOTE_PYTHON" \
  --pyannote-model "$PYANNOTE_MODEL_DIR"
```

For existing acoustic turns, replace the pyannote options with
`--speaker-turns-dir "$SPEAKER_TURNS_DIR"`. Repeat the same command with `--resume`
to reuse completed observations under the same configuration. Running the
pipeline sends clips to the caller's configured Gemini service for ASR/quality
scoring; offline tests use substitutes for those calls.

The generated `training_manifest.jsonl` uses the actual pipeline schema:
`item_id`, `video_id`, local `speaker_id`, `audio_path`, `audio_sha256`,
`duration_ms`, `accepted` and `final_text`. Connect it to the weighting workflow
using the root package's pairing adapter:

```bash
python -m s2r_adaptation.video \
  --input-manifest "$DATA_WORK_DIR/video/training_manifest.jsonl" \
  --output-dir "$DATA_WORK_DIR/pairs" --language my --seed 42
```

The adapter selects a distinct 1–10 second reference from the same recording and
local speaker, scopes speaker labels by recording, keeps `final_text` unchanged,
and reports targets without eligible references. This deterministic reference
policy is provided for reproduction; it does not claim byte-identical recovery
of an old experiment's target/reference assignments.

Run ASR2 on the exact final target audio, preserving the generated `id` and
`audio_sha256`, then continue with the root `s2r_adaptation.manifests` command.

## Tests

```bash
python -m unittest discover -s video_data_pipeline/tests -v
```

The selected original tests cover segmentation, speaker timelines, waveform and
text quality, music recovery, request contracts and pipeline resume/failure
behavior. Workers/ASR/model calls are substituted where needed. They do not
download model checkpoints or require production data.
