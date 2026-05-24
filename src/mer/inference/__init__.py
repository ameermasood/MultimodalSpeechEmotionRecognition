"""Inference prompts, prediction helpers, and output parsing."""

from mer.inference.prompts import (
    SYSTEM_PROMPT,
    build_audio_content,
    build_emotion_instruction,
    build_system_user_conversation,
    build_user_only_conversation,
)
from mer.inference.voxtral import (
    add_zero_shot_predictions,
    build_zero_shot_conversation,
    predict_emotion,
    predict_emotion_batch,
)

__all__ = [
    "SYSTEM_PROMPT",
    "add_zero_shot_predictions",
    "build_audio_content",
    "build_emotion_instruction",
    "build_system_user_conversation",
    "build_user_only_conversation",
    "build_zero_shot_conversation",
    "predict_emotion",
    "predict_emotion_batch",
]
