"""Dataset loading, transcript parsing, and label normalization helpers."""

from mer.data.labels import (
    CANONICAL_EMOTION_SET,
    CANONICAL_EMOTIONS,
    DEFAULT_SYNONYMS,
    normalize_emotion_name,
    normalize_prediction_text,
)

__all__ = [
    "CANONICAL_EMOTION_SET",
    "CANONICAL_EMOTIONS",
    "DEFAULT_SYNONYMS",
    "normalize_emotion_name",
    "normalize_prediction_text",
]
