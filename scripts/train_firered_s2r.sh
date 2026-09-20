#!/usr/bin/env bash
set -euo pipefail
: "${DATA_WORK_DIR:?Set a fresh directory outside this repository}"
: "${FIRERED_MODEL_DIR:?Download the pinned FireRed model first}"
: "${SYNTHETIC_PAIRS:?Provide synthetic target/reference pairs}"
: "${REAL_PAIRS:?Provide real target/reference pairs}"
: "${ASR2_RESULTS:?Provide ASR2 results on the final target audio}"
S2R_LANGUAGE="${S2R_LANGUAGE:-my}"
S2R_SEED="${S2R_SEED:-42}"
python -c 'import sys; from s2r_adaptation.backbones import external_output; p=external_output(sys.argv[1]); assert not p.exists(), "DATA_WORK_DIR must be new"' "$DATA_WORK_DIR"

python -m s2r_adaptation.manifests --stage S --primary "$SYNTHETIC_PAIRS" --output-dir "$DATA_WORK_DIR/S-data"
python -m s2r_adaptation.manifests --primary "$REAL_PAIRS" --secondary "$ASR2_RESULTS" \
  --require-audio-sha --gamma 3 --w-min 0.10 --output-dir "$DATA_WORK_DIR/R-data"

python -m s2r_adaptation.firered configure --manifest "$DATA_WORK_DIR/S-data/train.jsonl" \
  --pretrained "$FIRERED_MODEL_DIR" --output-dir "$DATA_WORK_DIR/S-train" \
  --language "$S2R_LANGUAGE" --seed "$S2R_SEED" --config-out "$DATA_WORK_DIR/S.json"
python -m torch.distributed.run --standalone --nproc_per_node=1 -m s2r_adaptation.firered train --config "$DATA_WORK_DIR/S.json"
python -m torch.distributed.run --standalone --nproc_per_node=1 -m s2r_adaptation.firered export \
  --config "$DATA_WORK_DIR/S.json" --output-dir "$DATA_WORK_DIR/S-model"

python -m s2r_adaptation.firered configure --manifest "$DATA_WORK_DIR/R-data/train.jsonl" \
  --pretrained "$FIRERED_MODEL_DIR" --s-core "$DATA_WORK_DIR/S-model" --output-dir "$DATA_WORK_DIR/R-train" \
  --language "$S2R_LANGUAGE" --seed "$S2R_SEED" --config-out "$DATA_WORK_DIR/R.json"
python -m torch.distributed.run --standalone --nproc_per_node=1 -m s2r_adaptation.firered train --config "$DATA_WORK_DIR/R.json"
python -m torch.distributed.run --standalone --nproc_per_node=1 -m s2r_adaptation.firered export \
  --config "$DATA_WORK_DIR/R.json" --output-dir "$DATA_WORK_DIR/R-model"
