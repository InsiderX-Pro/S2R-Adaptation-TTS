"""Download publicly available model files at immutable revisions."""
import argparse
import json
from .backbones import external_output

MODELS = {
    "firered": {"repo_id": "FireRedTeam/FireRedTTS3", "revision": "dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610",
                "allow_patterns": ["fireredtts3_base/*", "redae/*", "campp/*", "text_tokenizer/*", "README.md"]},
    "omnivoice": {"repo_id": "k2-fsa/OmniVoice", "revision": "999c332499c708b116876ff5fe1aa5dd15f422ce",
                  "allow_patterns": ["*.json", "*.safetensors", "*.jinja", "audio_tokenizer/*", "README.md"]},
}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=list(MODELS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--list-only", action="store_true")
    args=parser.parse_args(argv)
    spec=MODELS[args.backbone]
    if args.list_only:
        print(json.dumps(spec, indent=2))
        return
    from huggingface_hub import snapshot_download
    output=external_output(args.output_dir)
    snapshot_download(**spec,local_dir=output)
    (output/"download_revision.json").write_text(json.dumps(spec,indent=2)+"\n",encoding="utf-8")
    print(output)


if __name__=="__main__":
    main()
