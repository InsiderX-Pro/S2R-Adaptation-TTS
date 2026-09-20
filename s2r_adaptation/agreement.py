"""Equations (1) and (2). Inputs are Unicode code points, never graphemes."""
from __future__ import annotations

import math
import unicodedata


def normalize_for_scoring(text: str) -> str:
    """NFC; drop whitespace and P/S/C categories; preserve case and L/N/M."""
    if not isinstance(text, str):
        raise TypeError("ASR transcripts must be strings; None is not an empty transcript")
    return "".join(
        char for char in unicodedata.normalize("NFC", text)
        if not char.isspace() and unicodedata.category(char)[0] not in "PSC"
    )


def levenshtein_python(primary: str, secondary: str) -> int:
    """Linear-memory exact Levenshtein fallback, including insertion errors."""
    if len(primary) < len(secondary):
        primary, secondary = secondary, primary
    previous = list(range(len(secondary) + 1))
    for i, left in enumerate(primary, 1):
        current = [i]
        for j, right in enumerate(secondary, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


try:
    from rapidfuzz.distance.Levenshtein import distance as levenshtein
except ImportError:
    levenshtein = levenshtein_python


def disagreement(primary: str, secondary: str) -> float:
    reference = normalize_for_scoring(primary)
    hypothesis = normalize_for_scoring(secondary)
    if not reference:
        raise ValueError("empty normalized primary transcript must be excluded")
    return levenshtein(reference, hypothesis) / max(1, len(reference))


def reliability_weight(d: float, gamma: float = 3.0, w_min: float = 0.10) -> float:
    """max(w_min, (1 - min(d, 1)) ** gamma); d is a ratio, not percent."""
    d, gamma, w_min = float(d), float(gamma), float(w_min)
    if not math.isfinite(d) or d < 0:
        raise ValueError("disagreement must be finite and non-negative")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if not math.isfinite(w_min) or not 0 <= w_min <= 1:
        raise ValueError("w_min must be finite and in [0, 1]")
    return max(w_min, (1.0 - min(d, 1.0)) ** gamma)
