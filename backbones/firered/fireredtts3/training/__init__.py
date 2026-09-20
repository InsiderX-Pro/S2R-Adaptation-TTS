"""Training utilities for FireRedTTS3-Base.

The upstream release exposes inference only.  This package adds a small,
dependency-free training stack around the released Base model without changing
the default inference behaviour.
"""

from .model import FireRedTTS3ForTraining, FireRedTTS3TrainingOutput

__all__ = ["FireRedTTS3ForTraining", "FireRedTTS3TrainingOutput"]
