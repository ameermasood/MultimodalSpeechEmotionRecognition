"""IEMOCAP path and utterance metadata helpers."""

from __future__ import annotations

import os
import re


def infer_gender_from_utt(utt_id: str | None) -> str:
    """Infer speaker gender from an IEMOCAP utterance ID."""
    if not isinstance(utt_id, str) or not utt_id:
        return "unknown"

    session_code = utt_id.split("_")[0]
    if session_code.endswith("F"):
        return "female"
    if session_code.endswith("M"):
        return "male"
    return "unknown"


def infer_session_from_utt(utt_id: str | None) -> str:
    """Extract the session code from an IEMOCAP utterance ID."""
    match = re.match(r"(Ses\d\d)", utt_id or "")
    return match.group(1) if match else "Unknown"


def resolve_iemocap_audio_path(audio_root: str | os.PathLike[str], wav_field: str | os.PathLike[str]) -> str | None:
    """Resolve an IEMOCAP audio path against the audio root."""
    if not wav_field:
        return None

    wav_value = str(wav_field).replace("\\", "/")
    if os.path.isabs(wav_value) and os.path.isfile(wav_value):
        return wav_value

    root = str(audio_root)
    candidate = os.path.join(root, wav_value)
    if os.path.isfile(candidate):
        return candidate

    candidate = os.path.join(root, wav_value.lstrip("/"))
    if os.path.isfile(candidate):
        return candidate

    return None
