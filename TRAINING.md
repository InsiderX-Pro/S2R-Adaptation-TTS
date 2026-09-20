# Running S → R

Both training paths are included. FireRed uses the recovered native full-FSDP
trainer, frozen RedAE/CAM++ feature extraction and model-only exporter. OmniVoice
uses the included model, paired codec preparation and a single-device training
entry supporting full tuning or the recovered native LoRA implementation.

No private checkout, experiment launcher, GPU lease service or external adapter
file is required. Bring your own training pairs and download the public model
files below. Keep all generated data and checkpoints outside this repository.

## Install

Use Python 3.12 on Linux for training. Select your CUDA device using
`CUDA_VISIBLE_DEVICES`. FireRed full training/export requires a BF16-capable CUDA
device; the OmniVoice entry also supports CPU/float32 for small verification runs.
The default paper recipes use one GPU and three accumulated examples per update.

From this repository's root, in your chosen virtual environment:

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-training.txt
python -m pip install --no-deps -e . -e ./backbones/firered -e ./backbones/omnivoice
```

For CPU tests, use the CPU PyTorch index instead. Video preprocessing has its own
installation instructions in `video_data_pipeline/README.md`; its separator and
diarization workers can use separate environments.

The direct training dependencies are pinned in `requirements-training.txt`.
This is a tested dependency set, not a complete operating-system/CUDA image lock.
`backbones/SOURCES.json` identifies the included source snapshots and original
file digests; Apache license notices are retained beside both backbones.

| Component | Included or retrievable version |
|---|---|
| FireRed core + training extension | `backbones/firered`, upstream reference [1d32ba7](https://github.com/FireRedTeam/FireRedTTS3/tree/1d32ba780da6af37a71bdfd9c68c12003e908a46) with the included custom training/language/routing implementation |
| FireRed model, RedAE, CAM++, text tokenizer | [FireRedTeam/FireRedTTS3 at dcf1bdc](https://huggingface.co/FireRedTeam/FireRedTTS3/tree/dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610) |
| OmniVoice source | Included 0.1.5 snapshot in `backbones/omnivoice`; [upstream project](https://github.com/k2-fsa/OmniVoice) |
| OmniVoice model + Higgs audio codec + text tokenizer | [k2-fsa/OmniVoice at 999c332](https://huggingface.co/k2-fsa/OmniVoice/tree/999c332499c708b116876ff5fe1aa5dd15f422ce) |
| OmniVoice LoRA | Included in `s2r_adaptation/omni_lora`; native implementation, no PEFT package required |

Set `FIRERED_MODEL_DIR` and `OMNIVOICE_MODEL_DIR` to your own model directories:

```bash
python -m s2r_adaptation.assets --backbone firered --output-dir "$FIRERED_MODEL_DIR"
python -m s2r_adaptation.assets --backbone omnivoice --output-dir "$OMNIVOICE_MODEL_DIR"
```

These commands download model weights at the fixed revisions. Add `--list-only`
to inspect the repository ID, revision and file selection without downloading.
The FireRed downloader omits the unrelated Instruct weights. The OmniVoice
download includes `audio_tokenizer/`; no separate unpinned codec download is used
by the paired preparation entry.

## Input contract

Use one JSONL row per target. Each row needs `id`, `audio_path`, unchanged `text`,
`speaker_id`, `video_id`, `language_id` (`my` or `lo`), `duration` in seconds,
`prompt_audio_path`, `prompt_text`, `prompt_duration`, `audio_sha256` and
`prompt_audio_sha256`. References must be distinct same-speaker recordings or
segments. Target duration is 0.4–20 seconds and reference duration is 1–10 seconds.
Use absolute paths for input pairs; locations are caller-specific and belong in
external data manifests. Do not copy those manifests into the release source.

The video pairing command in the root README produces this contract for real
recordings. Synthetic pairs retain the teacher's original input text. The
weighting preparer retains all original fields and adds weight/provenance fields.

ASR2 results need matching `id`, final target `audio_sha256`, `hypothesis` and
`status="success"`. Run ASR2 on the same final waveform used for training.
The training entry does not invoke an ASR service. Transcript normalization is
used for agreement scoring only; it never rewrites target conditioning text.

## One-command workflows

Set `DATA_WORK_DIR` to a **new** directory outside the repository, and set
`SYNTHETIC_PAIRS`, `REAL_PAIRS`, `ASR2_RESULTS` and the relevant model directory.
Set `S2R_LANGUAGE=my` or `lo`; `S2R_SEED` defaults to 42.

```bash
bash scripts/train_firered_s2r.sh
```

This prepares unit-weight S data and cubic-weight R data, configures and trains S,
exports its model tensors, configures R from that export with fresh training
state, trains R, and exports the final R core.

For OmniVoice, use a different fresh `DATA_WORK_DIR`:

```bash
TRAIN_MODE=full bash scripts/train_omnivoice_s2r.sh
# Or select the recovered LoRA option in a separate fresh work directory:
TRAIN_MODE=lora bash scripts/train_omnivoice_s2r.sh
```

The script separately encodes target and reference audio with the fixed Higgs
codec, then trains S and R. `CODEC_DEVICE` defaults to `cuda`; set it to `cpu` if
needed. Prepared tokens store relative paths within their output directory and
checksums for the original audio, token files and source manifest.

LoRA defaults are rank 8, alpha 16, dropout 0.05 and `text_prediction` routing on
Qwen3 attention/MLP projections. This option reuses the recovered adapter, but its
S/R orchestration is reconstructed for this reference. It is an additional tuning
option and does not establish equivalence to the paper's full-tuning results.
S and R must use the same adapter structure and base model. Routed adapters remain
separate from the frozen base; they must not be merged as ordinary global LoRA.

## Individual commands and configuration

Eight complete templates are included under `s2r_adaptation/configs/{firered,omnivoice}`:
`my_S.json`, `my_R.json`, `lo_S.json`, `lo_R.json`. `configure` fills all data/model/output
locations and stage provenance from its arguments; `${...}` values in the templates
are placeholders, not paths to a server. Pass `--template` to supply a modified copy.

| Language | S updates / LR | R updates / LR |
|---|---|---|
| Burmese (`my`) | 53,640 / 2e-6 | 16,015 / 1e-6 |
| Lao (`lo`) | 39,749 / 2e-6 | 8,668 / 1e-6 |

Defaults use AdamW, cosine scheduling, 3% warmup and accumulation 3. Run each paper
seed (42/17/73) with its own corresponding S training. Configuration files can be
edited for debugging or other data; changed budgets/settings are different runs.

After preparing the weighted manifests, FireRed S can be run explicitly:

```bash
python -m s2r_adaptation.firered configure \
  --manifest "$DATA_WORK_DIR/S-data/train.jsonl" --pretrained "$FIRERED_MODEL_DIR" \
  --output-dir "$DATA_WORK_DIR/S-train" --language my --seed 42 \
  --config-out "$DATA_WORK_DIR/S.json"
python -m s2r_adaptation.firered check --config "$DATA_WORK_DIR/S.json"
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m s2r_adaptation.firered train --config "$DATA_WORK_DIR/S.json"
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m s2r_adaptation.firered export --config "$DATA_WORK_DIR/S.json" \
  --output-dir "$DATA_WORK_DIR/S-model"
```

Configure R with its manifest and `--s-core "$DATA_WORK_DIR/S-model"`, using new
R output/config locations. The export contains model tensors/config plus verified
source metadata. Optimizer, scheduler, data offsets and RNG are not imported into R.
FireRed computes frozen features on demand; enable `feature_cache_dir` in a config
to cache them outside the source tree. No pre-existing private feature cache is needed.

The equivalent OmniVoice steps are:

```bash
python -m s2r_adaptation.codec --manifest "$DATA_WORK_DIR/S-data/train.jsonl" \
  --codec-dir "$OMNIVOICE_MODEL_DIR/audio_tokenizer" --device cuda \
  --output-dir "$DATA_WORK_DIR/S-codec"
python -m s2r_adaptation.omni_train configure \
  --manifest "$DATA_WORK_DIR/S-data/train.jsonl" --tokens "$DATA_WORK_DIR/S-codec" \
  --pretrained "$OMNIVOICE_MODEL_DIR" --output-dir "$DATA_WORK_DIR/S-train" \
  --language my --seed 42 --mode full --config-out "$DATA_WORK_DIR/S.json"
python -m s2r_adaptation.omni_train train --config "$DATA_WORK_DIR/S.json"
```

Prepare R codec data from the R manifest, then configure R with new R locations and
`--s-checkpoint "$S_CHECKPOINT"`. The completed S checkpoint path appears in
`S-train/training_summary.json`. No optimizer state crosses this boundary.

The paired processor conditions on unchanged reference/target text and the
reference's acoustic tokens. Only masked target tokens receive CE supervision;
reference, text and padding positions have label `-100`. Per-codebook losses are
averaged within each example, retain native codebook coefficients, and receive
one cubic weight. Loss is averaged by example count, never by the sum of weights.
Sequence packing is disabled. Conditional dropout and masking are recorded in
the complete configuration.

## Resume and checkpoints

Resume an interrupted **same stage** explicitly:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m s2r_adaptation.firered train --config "$STAGE_CONFIG" --resume "$CHECKPOINT"
python -m s2r_adaptation.omni_train train --config "$STAGE_CONFIG" --resume "$CHECKPOINT"
```

Keep the original configuration for resume. OmniVoice restores model/adapter,
AdamW state, scheduler, Python/Torch RNG and shuffled-data cursor. Its checkpoint
directory is published only after every file is written and hashed. A fresh R
stage requires a completed S budget and instead resets all non-model state.

Full OmniVoice weights are in `checkpoint-*/model/`; LoRA tensors and the exact
adapter specification are in `checkpoint-*/adapter/`. For the latter, load the
same downloaded base and use `omni_lora.load_adapter_checkpoint`. Inference with
`text_prediction` routing must set `inference_routing_mask` on each forward call;
the included helper implements the original routing rule. Audio codec/text
tokenizer assets continue to come from the fixed base-model download.

## Verification boundaries

```bash
S2R_FIRERED_CODE="$PWD/backbones/firered" \
S2R_OMNIVOICE_CODE="$PWD/backbones/omnivoice" \
  python -m unittest discover -s tests -v
PYTHONPATH="$PWD/backbones/firered" python -m unittest discover -s backbones/firered/tests -v
python -m unittest discover -s video_data_pipeline/tests -v
```

Tests cover cubic weighting/gradients, pairing, token identity, target-only masking,
native FireRed loss/configuration, full and recovered-LoRA S → R on a tiny native
OmniVoice/Qwen3 model, and bit-exact interrupted/resumed CPU training. Package
installation is also checked outside the source tree, including all eight templates.
These checks use synthetic fixtures. They do not claim a new full-dataset GPU run,
quality-metric reproduction or real-video/API execution.
