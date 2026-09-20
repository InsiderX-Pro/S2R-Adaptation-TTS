import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from s2r_adaptation.backbones import use_backbone
from s2r_adaptation.codec import prepare_codec
from s2r_adaptation.firered import stage_config, check_config
from s2r_adaptation.manifests import build_manifest, sha256
from s2r_adaptation.omni_data import PairedCodecDataset, PairedProcessor
from s2r_adaptation.omni_train import configuration, train, _create_model, model_fingerprint, checkpoint_metadata


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.primary = self.root / "primary.jsonl"
        self.secondary = self.root / "secondary.jsonl"
        # Synthetic fixtures only; each file has a distinct content digest.
        for name in ("a", "b"):
            (self.root / (name+".wav")).write_bytes(name.encode()*20)
        rows=[]
        secondary=[]
        for name, prompt in [("a", "b"), ("b", "a")]:
            rows.append({"id":name,"audio_path":str(self.root/(name+".wav")),"text":"abcd!",
                "speaker_id":"speaker","video_id":"recording","language_id":"my","duration":2,
                "prompt_audio_path":str(self.root/(prompt+".wav")),"prompt_text":"reference text",
                "prompt_duration":2,"audio_sha256":sha256(self.root/(name+".wav")),
                "prompt_audio_sha256":sha256(self.root/(prompt+".wav"))})
            secondary.append({"id":name,"status":"success","hypothesis":"ab","audio_sha256":rows[-1]["audio_sha256"]})
        self.primary.write_text("".join(json.dumps(r)+"\n" for r in rows))
        self.secondary.write_text("".join(json.dumps(r)+"\n" for r in secondary))
        for stage in ("S", "R"):
            build_manifest(self.primary, self.root/stage, stage=stage,
                           secondary=self.secondary if stage=="R" else None)
            prepare_codec(self.root/stage/"train.jsonl", self.root/(stage+"-codec"),
                lambda path: (torch.arange(12).reshape(2,6)+int(path.stem=="b"))%7,
                codec_identity={"test_only":True}, channels=2,vocab_size=7)

    def create_tiny_base(self):
        use_backbone("omnivoice")
        from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        base=self.root/"base"
        config=OmniVoiceConfig(audio_vocab_size=8,audio_mask_id=7,num_audio_codebook=2,
            audio_codebook_weights=[2,1],llm_config={"model_type":"qwen3","hidden_size":8,
            "intermediate_size":16,"num_hidden_layers":28,"num_attention_heads":2,
            "num_key_value_heads":1,"head_dim":4,"vocab_size":16,"max_position_embeddings":256})
        config._attn_implementation="sdpa"
        model=OmniVoice(config)
        model.save_pretrained(base)
        tokenizer=Tokenizer(WordLevel({"[PAD]":0,"[UNK]":1,"abcd":2,"reference":3},unk_token="[UNK]"))
        tokenizer.pre_tokenizer=Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token="[PAD]",unk_token="[UNK]").save_pretrained(base)
        return base

    def config(self, base, stage, mode="full", initial=None):
        value=configuration(self.root/stage/"train.jsonl",self.root/(stage+"-codec"),base,
            self.root/(mode+"-"+stage),mode=mode,init_from=initial)
        value.update(max_steps=1,save_steps=1,log_steps=1,device="cpu",dtype="float32",
            gradient_checkpointing=False,learning_rate=.001,mask_ratio_range=[1,1],drop_cond_ratio=0)
        value["lora"].update(rank=2,alpha=4,layer_indices=[0,1],target_modules=["q_proj","v_proj"])
        return value

    def test_pair_processor_preserves_reference_and_masks_target_only(self):
        dataset=PairedCodecDataset(self.root/"R-codec",self.root/"R/train.jsonl")
        calls=[]
        class Tokenizer:
            def __call__(self,text,**kwargs):
                calls.append(text)
                return SimpleNamespace(input_ids=torch.tensor([[1,2]]))
        sample=PairedProcessor(Tokenizer(),2,7,mask_ratio_range=(1,1),drop_cond_ratio=0)(dataset[0])
        self.assertTrue((sample["labels"][:,:10]==-100).all())
        self.assertTrue((sample["labels"][:,10:]>=0).all())
        self.assertTrue((sample["input_ids"][:,10:]==7).all())
        self.assertIn("reference text abcd!",calls[1])
        self.assertEqual(sample["sample_weight"],.125)

    def test_codec_rejects_changed_audio_or_stage(self):
        with self.assertRaisesRegex(ValueError,"manifest/stage"):
            PairedCodecDataset(self.root/"S-codec",self.root/"R/train.jsonl")
        (self.root/"a.wav").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"audio bytes"):
            prepare_codec(self.root/"S/train.jsonl",self.root/"invalid",lambda p:torch.zeros(2,2,dtype=torch.long),
                          codec_identity={},channels=2,vocab_size=7)
        self.assertFalse((self.root/"invalid").exists())

    def test_full_s_to_r_loads_model_with_fresh_optimizer(self):
        base=self.create_tiny_base()
        s=train(self.config(base,"S"))
        r_config=self.config(base,"R",initial=s["checkpoint"])
        loaded=_create_model(r_config,s["checkpoint"],model_fingerprint(base))
        from safetensors.torch import load_file
        expected=load_file(str(Path(s["checkpoint"])/"model/model.safetensors"))
        for name,tensor in loaded.state_dict().items():
            torch.testing.assert_close(tensor,expected[name])
        r=train(r_config)
        self.assertEqual((r["initial_step"],r["initial_optimizer_entries"]),(0,0))
        self.assertEqual(checkpoint_metadata(r["checkpoint"])["stage"],"R")
        self.assertEqual(r["global_step"],1)

    def test_recovered_lora_s_to_r_changes_only_adapter(self):
        base=self.create_tiny_base()
        s=train(self.config(base,"S","lora"))
        from safetensors.torch import load_file
        weights=load_file(str(Path(s["checkpoint"])/"adapter/adapter_model.safetensors"))
        self.assertTrue(all("lora_" in name for name in weights))
        self.assertTrue(any("lora_B" in name and tensor.abs().sum()>0 for name,tensor in weights.items()))
        r_config=self.config(base,"R","lora",s["checkpoint"])
        loaded=_create_model(r_config,s["checkpoint"],model_fingerprint(base))
        from s2r_adaptation.omni_lora import adapter_state_dict
        for name,tensor in adapter_state_dict(loaded,loaded._local_peft_spec).items():
            torch.testing.assert_close(tensor,weights[name])
        r=train(r_config)
        self.assertEqual((r["initial_step"],r["initial_optimizer_entries"]),(0,0))
        state=torch.load(Path(r["checkpoint"])/"training_state.pt",weights_only=False,map_location="cpu")
        self.assertTrue(all(int(item["step"])==1 for item in state["optimizer"]["state"].values()))

    def test_firered_bundled_template_has_no_external_extension(self):
        use_backbone("firered")
        from fireredtts3.training.trainer import TrainerConfig
        template=Path(__file__).resolve().parents[1]/"s2r_adaptation/configs/firered/my_S.json"
        value=stage_config(json.loads(template.read_text()),self.root/"S/train.jsonl",self.root/"firered",self.root/"base")
        TrainerConfig(**value).validate()
        result=check_config(value,use_backbone("firered"))
        self.assertEqual(result["native_records"],2)
        self.assertFalse(result["cuda_initialized"])

    def test_same_stage_resume_matches_uninterrupted_training(self):
        base=self.create_tiny_base()
        reference=self.config(base,"S")
        reference.update(max_steps=3,output_dir=str(self.root/"uninterrupted"),
                         mask_ratio_range=[.3,.9],drop_cond_ratio=.25)
        complete=train(reference)
        interrupted={**reference,"output_dir":str(self.root/"interrupted")}
        from s2r_adaptation import omni_train
        save=omni_train._save_checkpoint
        def stop_after_save(*args,**kwargs):
            save(*args,**kwargs)
            raise RuntimeError("simulated interruption")
        with patch.object(omni_train,"_save_checkpoint",side_effect=stop_after_save):
            with self.assertRaisesRegex(RuntimeError,"simulated interruption"):
                train(interrupted)
        checkpoint=self.root/"interrupted/checkpoint-00000001"
        with self.assertRaisesRegex(ValueError,"completed S"):
            train(self.config(base,"R",initial=checkpoint))
        resumed=train(interrupted,resume=checkpoint)
        self.assertEqual(resumed["initial_step"],1)
        self.assertGreater(resumed["initial_optimizer_entries"],0)
        from safetensors.torch import load_file
        left=load_file(str(Path(complete["checkpoint"])/"model/model.safetensors"))
        right=load_file(str(Path(resumed["checkpoint"])/"model/model.safetensors"))
        for name in left:
            torch.testing.assert_close(left[name],right[name],rtol=0,atol=0)

    def test_codec_token_corruption_is_rejected_before_training(self):
        dataset=PairedCodecDataset(self.root/"S-codec",self.root/"S/train.jsonl")
        token_path=dataset.directory/dataset.rows[0]["target_tokens"]
        token_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError,"checksum"):
            dataset[0]


if __name__=="__main__":
    unittest.main()
