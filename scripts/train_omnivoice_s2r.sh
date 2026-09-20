#!/usr/bin/env bash
set -euo pipefail
: "${DATA_WORK_DIR:?Set a fresh directory outside this repository}"
: "${OMNIVOICE_MODEL_DIR:?Download the pinned OmniVoice model first}"
: "${SYNTHETIC_PAIRS:?Provide synthetic target/reference pairs}"
: "${REAL_PAIRS:?Provide real target/reference pairs}"
: "${ASR2_RESULTS:?Provide ASR2 results on the final target audio}"
S2R_LANGUAGE="${S2R_LANGUAGE:-my}"
S2R_SEED="${S2R_SEED:-42}"
TRAIN_MODE="${TRAIN_MODE:-full}"
CODEC_DEVICE="${CODEC_DEVICE:-cuda}"
python -c 'import sys; from s2r_adaptation.backbones import external_output; p=external_output(sys.argv[1]); assert not p.exists(), "DATA_WORK_DIR must be new"' "$DATA_WORK_DIR"

python -m s2r_adaptation.manifests --stage S --primary "$SYNTHETIC_PAIRS" --output-dir "$DATA_WORK_DIR/S-data"
python -m s2r_adaptation.manifests --primary "$REAL_PAIRS" --secondary "$ASR2_RESULTS" \
  --require-audio-sha --gamma 3 --w-min 0.10 --output-dir "$DATA_WORK_DIR/R-data"
for STAGE in S R; do
  python -m s2r_adaptation.codec --manifest "$DATA_WORK_DIR/$STAGE-data/train.jsonl" \
    --codec-dir "$OMNIVOICE_MODEL_DIR/audio_tokenizer" --device "$CODEC_DEVICE" --output-dir "$DATA_WORK_DIR/$STAGE-codec"
done
python -m s2r_adaptation.omni_train configure --manifest "$DATA_WORK_DIR/S-data/train.jsonl" \
  --tokens "$DATA_WORK_DIR/S-codec" --pretrained "$OMNIVOICE_MODEL_DIR" --output-dir "$DATA_WORK_DIR/S-train" \
  --language "$S2R_LANGUAGE" --seed "$S2R_SEED" --mode "$TRAIN_MODE" --config-out "$DATA_WORK_DIR/S.json"
python -m s2r_adaptation.omni_train train --config "$DATA_WORK_DIR/S.json"
S_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$DATA_WORK_DIR/S-train/training_summary.json")"
python -m s2r_adaptation.omni_train configure --manifest "$DATA_WORK_DIR/R-data/train.jsonl" \
  --tokens "$DATA_WORK_DIR/R-codec" --pretrained "$OMNIVOICE_MODEL_DIR" --output-dir "$DATA_WORK_DIR/R-train" \
  --language "$S2R_LANGUAGE" --seed "$S2R_SEED" --mode "$TRAIN_MODE" --s-checkpoint "$S_CHECKPOINT" --config-out "$DATA_WORK_DIR/R.json"
python -m s2r_adaptation.omni_train train --config "$DATA_WORK_DIR/R.json"
