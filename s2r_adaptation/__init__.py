"""Paper-defined S-to-R reliability weighting, without inference-time ASR."""
from .agreement import disagreement, normalize_for_scoring, reliability_weight

__all__ = ["disagreement", "normalize_for_scoring", "reliability_weight"]
