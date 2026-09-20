"""Resolve the included backbone sources without a private checkout."""
from pathlib import Path
import sys


def use_backbone(name):
    if name not in {"firered", "omnivoice"}:
        raise ValueError(name)
    directory = Path(__file__).resolve().parents[1] / "backbones" / name
    if directory.is_dir():
        sys.path.insert(0, str(directory))
    return directory


def external_output(path):
    path = Path(path).expanduser().resolve()
    source = Path(__file__).resolve().parents[1]
    if path == source or source in path.parents:
        raise ValueError("Keep generated data/checkpoints outside the source repository")
    return path
