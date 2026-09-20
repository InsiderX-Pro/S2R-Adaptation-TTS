"""CPU forward/backward checks against installed backbones; no model download."""
import os
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn


@unittest.skipUnless(os.environ.get("S2R_FIRERED_CODE"), "set S2R_FIRERED_CODE for native integration")
class NativeFireRedTests(unittest.TestCase):
    def test_native_full_objective_and_all_gradients_scale(self):
        sys.path.insert(0, os.environ["S2R_FIRERED_CODE"])
        from fireredtts3.training.model import FireRedTTS3ForTraining
        class Patches(nn.Module):
            def __init__(self):
                super().__init__()
                self.project = nn.Linear(3, 4)
            def forward(self, value):
                return self.project(value.reshape(1, -1, 2, 3).mean(2))
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(10, 4)
                self.project = nn.Linear(4, 4)
            def forward(self, inputs_embeds, **kwargs):
                return SimpleNamespace(last_hidden_state=self.project(inputs_embeds).tanh())
        class Flow(nn.Module):
            def __init__(self):
                super().__init__()
                self.project = nn.Linear(9, 3)
            def forward(self, x, times): return self.project(x)
        torch.manual_seed(42)
        core = nn.Module()
        core.patch_size, core.history_patches, core.history_length = 2, 1, 2
        core.patch_encoder, core.backbone_llm, core.dit = Patches(), Backbone(), Flow()
        core.spk_proj_llm, core.spk_proj_dit = nn.Linear(3, 4), nn.Linear(3, 2)
        core.dit_head, core.stop_head = nn.Linear(4, 4), nn.Linear(4, 1)
        model = FireRedTTS3ForTraining(core)
        inputs = dict(spk_emb=torch.randn(1, 3), text_tokens=torch.tensor([[1, 2]]),
                      prompt_latents=torch.randn(1, 4, 3), target_latents=torch.randn(1, 6, 3),
                      cfg_dropout_prob=0, stop_positive_weight="auto")
        gradients = []
        outputs = []
        for weight in [1., .125]:
            model.zero_grad(set_to_none=True)
            output = model(**inputs, sample_weight=weight, generator=torch.Generator().manual_seed(123))
            output.loss.backward()
            gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
            outputs.append(output.loss.detach())
        torch.testing.assert_close(outputs[1], outputs[0] * .125)
        self.assertTrue(any("stop_head" in name and g.abs().sum() > 0 for name, g in gradients[0].items()))
        for name in gradients[0]:
            torch.testing.assert_close(gradients[1][name], gradients[0][name] * .125)


@unittest.skipUnless(os.environ.get("S2R_OMNIVOICE_CODE"), "set S2R_OMNIVOICE_CODE for native integration")
class NativeOmniVoiceTests(unittest.TestCase):
    def test_processor_collator_native_forward_and_gradient(self):
        sys.path.insert(0, os.environ["S2R_OMNIVOICE_CODE"])
        from omnivoice.data.processor import OmniVoiceSampleProcessor
        from omnivoice.data.collator import PaddingDataCollator
        from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
        from s2r_adaptation.omnivoice import install_weighting
        from s2r_adaptation.losses import omnivoice_loss
        class Tokenizer:
            pad_token_id = 0
            def __call__(self, text, return_tensors=None):
                return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
        class LLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)
            def get_input_embeddings(self): return self.embed
            def forward(self, inputs_embeds, **kwargs): return (inputs_embeds,)
        config = OmniVoiceConfig(num_audio_codebook=2, audio_vocab_size=6,
                                audio_codebook_weights=[2, 1],
                                llm_config={"model_type": "qwen3", "hidden_size": 4, "vocab_size": 10})
        model = OmniVoice(config, llm=LLM())
        self.addCleanup(setattr, OmniVoiceSampleProcessor, "__call__", OmniVoiceSampleProcessor.__call__)
        self.addCleanup(setattr, PaddingDataCollator, "__call__", PaddingDataCollator.__call__)
        self.addCleanup(setattr, OmniVoice, "forward", OmniVoice.forward)
        self.addCleanup(lambda: delattr(OmniVoice, "_s2r_installed") if hasattr(OmniVoice, "_s2r_installed") else None)
        install_weighting(stage="R")
        processor = OmniVoiceSampleProcessor(Tokenizer(), 2, 5, (0.1, .1), (.5, .5), 0, 0, 0, 0, 0)
        rows = []
        for weight, length in [(.125, 8), (1., 12)]:
            rows.append(processor({"label":{"text":"abc", "pseudo_label_weight":weight},
                                   "audio_tokens":torch.randint(0, 5, (2, length))}))
        batch = PaddingDataCollator(processor, 100)(rows)
        torch.testing.assert_close(batch["sample_weights"], torch.tensor([.125, 1.]))
        output = model(**batch)
        expected = omnivoice_loss(output.logits, batch["labels"], [.125, 1.], [2, 1])
        torch.testing.assert_close(output.loss, expected)
        torch.testing.assert_close(output["loss"], expected)
        output.loss.backward()
        self.assertGreater(model.audio_heads.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__": unittest.main()
