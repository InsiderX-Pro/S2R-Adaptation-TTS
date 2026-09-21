# S2R Adaptation TTS · Reproduction Guide

[Project overview](../README.md) · [Model weights](../README.md#model-weights)

Reference source for **From Reliable Text to Real Voices: Trust-Aware Progressive
Adaptation for Low-Resource TTS**. The repository covers real-video preprocessing,
same-speaker reference pairing, dual-ASR reliability scoring and weighted
synthetic-to-real adaptation.

[Project website and audio demos](https://insiderx-pro.github.io/S2R-Adaptation-TTS/)
 · [Latest manuscript](../docs/assets/paper.pdf)
 · [Training guide](TRAINING.md)

The repository remains private while the source release is being prepared.
The public research website is served exclusively from `docs/`; training source
is not included in the website deployment.

## Latest paper results

Adapted CER and SIM-O are mean ± sample SD over seeds 42/17/73.
MOS is mean [approximate 95% crossed-bootstrap CI], with 20 listeners and
30 matched conditions per language (600 ratings per system).

| Cubic S→R | CER (%) ↓ | SIM-O ↑ | H ↑ | MOS ↑ | Independent-ASR CER (%) ↓ |
|---|---|---|---|---|---|
| FireRedTTS3 / Burmese | 16.90 ± 0.26 | 0.6997 ± 0.0016 | 75.97 | 4.12 [3.93, 4.30] | 19.34 |
| FireRedTTS3 / Lao | 13.97 ± 0.36 | 0.6968 ± 0.0045 | 77.00 | 3.97 [3.77, 4.17] | 18.53 |
| OmniVoice / Burmese | 6.85 ± 0.08 | 0.7098 ± 0.0026 | 80.57 | 4.51 [4.35, 4.66] | 9.83 |

OmniVoice Base has 10.35% CER and 0.7278 SIM-O. Cubic improves its CER
by 3.50 percentage points. Numerical rankings do not establish significance;
only the FireRedTTS3/Burmese paired cubic-minus-uniform-0.5 MOS interval
excludes zero. See the manuscript and website for all strategies and controls.

Website content and provenance are documented in [WEBSITE.md](WEBSITE.md).

This is a source-only reference: no running data, real transcripts, model weights,
run reports, machine-specific paths, credentials or server-specific launch scripts are
included. All data, model and output locations are supplied by the caller.

## Source layout

| Location | Purpose |
|---|---|
| `video_data_pipeline/` | Selected actual video-processing implementation and tests |
| `s2r_adaptation/video.py` | Video output → distinct same-recording/speaker target/reference pairs |
| `s2r_adaptation/agreement.py` | Scoring-only text normalization, ASR disagreement and cubic weights |
| `s2r_adaptation/manifests.py` | Validated ASR join and prepared S/R manifests |
| `s2r_adaptation/losses.py` | FireRed and OmniVoice loss reductions by example count |
| `backbones/` | Included FireRed training/core/exporter and OmniVoice 0.1.5 source |
| `s2r_adaptation/firered.py` | FireRed stage configuration, preflight, training and export |
| `s2r_adaptation/codec.py` | Target/reference waveform → paired OmniVoice codec tokens |
| `s2r_adaptation/omni_train.py` | Complete OmniVoice full/LoRA S → R and same-stage resume |
| `s2r_adaptation/omni_lora/` | Recovered native OmniVoice LoRA implementation |
| `s2r_adaptation/configs/` | Eight complete language/stage/backbone templates |
| `scripts/` | Portable S → R orchestration for both backbones |
| `examples/` | Invented transcript records with placeholder audio paths |
| `tests/` | Scoring, pairing, loss/gradient and optional native integration tests |

The full path is:

```text
local video/audio → audio extraction → VAD + diarization → single-speaker clips
  → audio/music/text quality checks + ASR1 → accepted primary-text manifest
  → distinct same-speaker reference pairing
  → ASR2 on the same final target audio → disagreement → cubic weight
  → S adaptation with unit weights → R adaptation with per-example weights
```

## Method contract

```python
d = edit_distance(normalize(asr1_text), normalize(asr2_text)) / len(normalize(asr1_text))
w = max(0.10, (1.0 - min(d, 1.0)) ** 3.0)
loss = sum(w_i * full_sample_loss_i for i in batch) / len(batch)
```

`normalize` applies Unicode NFC and removes whitespace and Unicode categories
P/S/C. Letters, numbers and combining marks are retained. Distance and length
use code points. Case, numerals and Zawgyi are not converted. Normalization is
used only for scoring; ASR1 remains the training text without fusion/correction.

Empty normalized primary labels are excluded. A successful empty ASR2
transcript gives `d=1`; a failed or missing request is an error by default.
Disagreement can exceed one before clipping. The weight floor is applied after
the power: at `d=0.2`, `0.5`, `0.6`, weights are `0.512`, `0.125`, `0.1`.

FireRed weights the whole `flow + 0.1 * stop` loss. OmniVoice computes masked
acoustic-token CE within each example, retains the model's codebook coefficients,
then applies the example weight. Divide by example count, not weight sum.
Uniform 0.5 weighting halves loss and gradients; it need not halve AdamW updates.

## Install and test

Run the commands below from the repository root.

```bash
python -m pip install -r requirements/training.txt
python -m pip install --no-deps -e . -e ./backbones/firered -e ./backbones/omnivoice
python -m pip install -e './video_data_pipeline[music,dev]'
python -m unittest discover -s tests -v
python -m unittest discover -s video_data_pipeline/tests -v
```

Use Python 3.12 for the pinned training environment. Scoring/pairing support Python 3.10+ without third-party packages;
RapidFuzz accelerates large inputs. PyTorch is needed for loss tests/adapters.
Video processing additionally uses FFmpeg, the caller's pyannote and separator
environments, and configured Gemini credentials. See the
[video preprocessing README](../video_data_pipeline/README.md) for the selected
modules, preserved processing settings and environment variables.

All required backbone source is now included. [TRAINING.md](TRAINING.md) provides
installation, immutable public model revisions, complete commands, checkpoint
formats and verification instructions. Tests use tiny native models and synthetic
fixtures; pretrained model files are not bundled.

## Reproduction workflow

The variables below are caller-supplied locations. Set `DATA_WORK_DIR` to a
working directory **outside this source tree**. Example commands create outputs
there; generated outputs are not part of this repository.

### 1. Process real recordings and pair references

```bash
python -m ominivoice_data_pipeline run "$VIDEO_INPUT" \
  --output-dir "$DATA_WORK_DIR/video" \
  --pyannote-python "$PYANNOTE_PYTHON" \
  --pyannote-model "$PYANNOTE_MODEL_DIR"

python -m s2r_adaptation.video \
  --input-manifest "$DATA_WORK_DIR/video/training_manifest.jsonl" \
  --output-dir "$DATA_WORK_DIR/pairs" --language my --seed 42
```

The bridge preserves the video's finalized primary ASR text and scopes local
speaker labels by recording. Each target gets a different 1–10 second reference
from the same recording/speaker; identical audio hashes cannot form a pair.
Unpaired/ineligible targets are reported. This deterministic pairing policy is
a reference implementation, not a claim to recover old experiment assignments.

### 2. Run ASR2 and build cubic weights

Run a fixed ASR2 on the exact final target audio from `pairs/primary.jsonl`.
Record `id`, `audio_sha256`, `hypothesis`, `status="success"` and optionally
`model_id`. The ASR2 model/service runner is an external input; the weighting
step consumes its saved results.

```bash
python -m s2r_adaptation.manifests \
  --primary "$DATA_WORK_DIR/pairs/primary.jsonl" \
  --secondary "$ASR2_RESULTS" --require-audio-sha \
  --gamma 3 --w-min 0.10 --output-dir "$DATA_WORK_DIR/R-data"
```

Primary rows contain stable string `id` and unchanged `text` plus the original
training fields. Secondary IDs may use `id` or `target_id`; duplicate or conflicting
IDs fail. Optional secondary `reference` must equal ASR1 text exactly. Audio
digests must match when both are present; `--require-audio-sha` requires both.
Join by ID, not line order or approximate text. No waveform is modified here.

Each prepared directory contains a generated `train.jsonl`, `excluded.jsonl`
and `report.json` with counts/statistics/checksums. Output directories must be
new and failed preparation publishes no partial manifest.

`--on-asr-error exclude` explicitly records failed/missing requests as exclusions.
`--asr-format archived` is only for verified historical success tables without
row status: it requires `canonical.cer` and recomputes it from the two transcripts.
It cannot retrospectively establish request success.

### 3. Train S, transfer model weights, and train R

The complete entry points, native backbone source, model download revisions,
paired codec preparation, eight configuration templates and checkpoint/export
code are included. Follow [TRAINING.md](TRAINING.md) to install the environment
and set your own data/model locations.

```bash
bash scripts/train_firered_s2r.sh
# Use a separate fresh DATA_WORK_DIR for each backbone/mode/seed:
TRAIN_MODE=full bash scripts/train_omnivoice_s2r.sh
TRAIN_MODE=lora bash scripts/train_omnivoice_s2r.sh
```

FireRed extracts frozen RedAE/CAM++ features on demand, trains S with its native
FSDP implementation, exports a model-only core, and starts R with fresh optimizer,
scheduler and RNG state. The training extension and exporter are supplied in
`backbones/firered`.

OmniVoice encodes distinct target/reference audio with its pinned Higgs codec,
uses reference audio as conditioning and masked target tokens for CE, then applies
the per-example weight. Full and recovered native LoRA training both support S → R
and separate same-stage resume. Codec preparation no longer requires an external
private pipeline. All generated tokens and checkpoints stay outside this tree.

| Language | S updates / learning rate | R updates / learning rate |
|---|---|---|
| Burmese (`my`) | 53,640 / 2e-6 | 16,015 / 1e-6 |
| Lao (`lo`) | 39,749 / 2e-6 | 8,668 / 1e-6 |

Presets use one device, accumulation 3, cosine scheduling and 3% warmup. Paper
seeds are 42/17/73. Every replication needs the corresponding S initialization.
The LoRA option preserves the recovered adapter implementation but is an additional
reference tuning option, not a claim to reproduce the paper's full-tuning scores.

## Scope and release

This implementation reconstructs the paper's weighting path and selects the
surviving real-video processing code for reference. It does not contain paper
run data or claim byte-identical historical source/checkpoint reproduction.
The project owner must select a release license before publication. See
[THIRD_PARTY.md](THIRD_PARTY.md) for dependency/source scope.
