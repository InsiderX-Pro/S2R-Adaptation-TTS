# Source and dependency notices

- `backbones/firered` includes the FireRedTTS3 model/core modules and the recovered
  project training extension, language-token changes, native LoRA helpers and
  model-only FSDP exporter. The upstream Apache 2.0 license is retained in that
  directory. Server-specific GPU-resource helpers were removed from the reference.
- `backbones/omnivoice` includes the OmniVoice 0.1.5 model, data, training, utility
  and CLI source snapshot. Existing copyright headers and its Apache 2.0 license
  are retained. The unrelated evaluation suite is excluded.
- `s2r_adaptation/omni_lora` includes the recovered project OmniVoice adapter and
  its numerical helper dependencies. Package-relative imports replace the old
  external-module path. Server launchers, experiment data and path guards are
  excluded. The complete paired S/R trainer and codec bridge are reconstructed
  reference orchestration, not asserted to be byte-identical historical scripts.
- `backbones/SOURCES.json` records source provenance and original file digests.
  `TRAINING.md` links the upstream projects and exact public model revisions.
  `requirements-training.txt` pins the tested direct training dependencies.
- RapidFuzz remains an optional scoring accelerator; a pure-Python exact
  implementation is included. The native OmniVoice adapter does not need PEFT.

The video subproject contains selected actual processing modules, separator
worker, algorithm settings and tests. Environment defaults and connectivity were
made portable. Cloud storage, deployment, migration and backup modules are excluded.

No training data, real transcripts, codec-token arrays, checkpoint weights,
credentials or run records are distributed in this source package. Dataset,
checkpoint, teacher-output and ASR-service rights are separate from code licenses.
The project owner must select a license for the project-specific code before
publication; retained upstream licenses continue to apply to their source.
