"""Emotion label constants and normalization helpers."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

CANONICAL_EMOTIONS: tuple[str, ...] = ("Angry", "Happy", "Sad", "Neutral")
CANONICAL_EMOTION_SET: frozenset[str] = frozenset(CANONICAL_EMOTIONS)

DEFAULT_SYNONYMS: Mapping[str, str] = {
    "anger": "Angry",
    "angry": "Angry",
    "happiness": "Happy",
    "happy": "Happy",
    "sadness": "Sad",
    "sad": "Sad",
    "neutral": "Neutral",
    "calm": "Neutral",
}


def normalize_emotion_name(value: str | None) -> str:
    """Normalize dataset labels to title-case canonical names when possible."""
    text = (value or "").strip().lower()
    if not text:
        return ""
    return DEFAULT_SYNONYMS.get(text, text.capitalize())


def normalize_prediction_text(
    text: str | None,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
    default: str = "Neutral",
    synonyms: Mapping[str, str] = DEFAULT_SYNONYMS,
) -> str:
    """Map free-form model output to one emotion label.

    The matching order mirrors the notebook/script behavior: exact full-text
    match, last-token match, whole-word search, then fallback.
    """
    if text is None:
        return default

    value = text.strip().lower()
    if not value:
        return default

    label_lookup = {label.lower(): label for label in labels}

    if value in label_lookup:
        return label_lookup[value]
    if value in synonyms and synonyms[value] in labels:
        return synonyms[value]

    tokens = [token for token in re.split(r"\W+", value) if token]
    if tokens:
        last = tokens[-1]
        if last in label_lookup:
            return label_lookup[last]
        if last in synonyms and synonyms[last] in labels:
            return synonyms[last]

    for label_lower, label in label_lookup.items():
        if re.search(rf"\b{re.escape(label_lower)}\b", value):
            return label

    for alias, label in synonyms.items():
        if label in labels and re.search(rf"\b{re.escape(alias)}\b", value):
            return label

    return default
