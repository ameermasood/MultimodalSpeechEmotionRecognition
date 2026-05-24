"""Prompt builders for speech emotion recognition inference."""

from __future__ import annotations

import os
from typing import Sequence

from mer.data.labels import CANONICAL_EMOTIONS

SYSTEM_PROMPT = "You are a careful and concise emotion classification assistant."


def build_emotion_instruction(
    use_text: bool = False,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
    mention_transcript_reliability: bool = False,
) -> str:
    """Build the user instruction for audio-only or audio-plus-text inference."""
    label_list = ", ".join(labels)
    instruction = (
        "You are an emotion classifier for speech.\n"
        f"Possible emotions: {label_list}.\n"
        "From the given audio"
    )
    if use_text:
        instruction += " and its transcript"
    instruction += (
        ", classify the SPEAKER's emotion.\n"
        f"Answer with EXACTLY one word from this set: {label_list}.\n"
        "Do not add extra words, punctuation, or explanations."
    )
    if mention_transcript_reliability and use_text:
        instruction += "\nThe transcript may help, but it may be missing or incorrect."
    return instruction


def build_audio_content(audio_path: str) -> dict:
    """Build a processor-friendly audio content block."""
    wav_path = os.path.abspath(audio_path)
    return {
        "type": "audio",
        "audio_url": {"url": wav_path},
        "content": wav_path,
        "path": wav_path,
        "url": wav_path,
    }


def build_user_only_conversation(
    audio_path: str,
    transcript: str = "",
    use_text: bool = False,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
) -> list[dict]:
    """Build the user-only conversation format used by Voxtral scripts."""
    content = [
        build_audio_content(audio_path),
        {"type": "text", "text": build_emotion_instruction(use_text=use_text, labels=labels)},
    ]
    if use_text and transcript.strip():
        content.append({"type": "text", "text": f"Transcript:\n{transcript.strip()}"})
    return [{"role": "user", "content": content}]


def build_system_user_conversation(
    audio_path: str,
    transcript: str = "",
    use_text: bool = False,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
) -> list[dict]:
    """Build the system+user conversation format used by zero-shot notebooks."""
    user = build_user_only_conversation(
        audio_path=audio_path,
        transcript=transcript,
        use_text=use_text,
        labels=labels,
    )[0]
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        user,
    ]
