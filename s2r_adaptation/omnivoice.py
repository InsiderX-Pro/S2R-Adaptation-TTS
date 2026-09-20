"""A padded-batch adapter for native OmniVoice's masked-token objective.

Use existing precomputed codec tokens. The adapter changes only the loss
reduction and carries one weight per example through processor and collator.
"""
from __future__ import annotations

from .manifests import fields, read_rows, row_id


def install_weighting(stage="R", manifest=None):
    import torch
    from omnivoice.data.processor import OmniVoiceSampleProcessor
    from omnivoice.data.collator import PaddingDataCollator
    from omnivoice.models.omnivoice import OmniVoice
    from .losses import omnivoice_loss

    if getattr(OmniVoice, "_s2r_installed", False):
        raise RuntimeError("install weighting once per training process")
    weights = None
    if manifest:
        from .firered import verify_manifest
        if verify_manifest(manifest)["stage"] != stage:
            raise ValueError("OmniVoice stage and prepared manifest disagree")
        weights = {row_id(row): (fields(row)["text"], fields(row)["pseudo_label_weight"])
                   for row in read_rows(manifest)}
    original_processor = OmniVoiceSampleProcessor.__call__
    original_collator = PaddingDataCollator.__call__
    original_forward = OmniVoice.forward

    def process(self, sample):
        label = sample["label"]
        if weights is not None:
            identity = label.get("id", label.get("target_id", sample.get("__key__")))
            if identity not in weights:
                raise ValueError("codec sample id is missing from prepared weights")
            primary, weight = weights[identity]
            if label["text"] != primary:
                raise ValueError("codec label differs from unchanged primary ASR text")
        elif stage == "S":
            weight = 1.0
        else:
            if "pseudo_label_weight" not in label:
                raise ValueError("R codec label requires pseudo_label_weight or a prepared sidecar")
            weight = label["pseudo_label_weight"]
        result = original_processor(self, sample)
        result["_s2r_weight"] = float(weight)
        return result

    def collate(self, samples):
        batch = original_collator(self, samples)
        batch["sample_weights"] = torch.tensor([sample["_s2r_weight"] for sample in samples], dtype=torch.float32)
        return batch

    def forward(self, input_ids, audio_mask, labels=None, sample_weights=None, **kwargs):
        if labels is None:
            return original_forward(self, input_ids, audio_mask, labels=None, **kwargs)
        if sample_weights is None:
            raise ValueError("weighted training requires the S2R padding collator; packing is unsupported")
        output = original_forward(self, input_ids, audio_mask, labels=None, **kwargs)
        output.loss = omnivoice_loss(output.logits, labels, sample_weights,
                                    self.normalized_audio_codebook_weights)
        output["loss"] = output.loss
        return output

    OmniVoiceSampleProcessor.__call__ = process
    PaddingDataCollator.__call__ = collate
    OmniVoice.forward = forward
    OmniVoice._s2r_installed = True


def main(argv=None):
    from .omni_train import main as train_main
    return train_main(argv)


if __name__ == "__main__":
    main()
