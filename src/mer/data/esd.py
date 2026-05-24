"""ESD path, speaker, utterance, and transcript helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

from mer.data.labels import CANONICAL_EMOTIONS

ESD_ENGLISH_SPEAKER_MIN = 11
ESD_ENGLISH_SPEAKER_MAX = 20


def speaker_id_from_esd_path(path: str | os.PathLike[str]) -> str | None:
    """Extract an ESD speaker ID from a standard path or wav basename."""
    value = str(path or "").replace("\\", "/")
    match = re.search(r"downloads/esd/(\d{4})/", value)
    if match:
        return match.group(1)

    basename = os.path.basename(value)
    match = re.match(r"^(\d{4})_", basename)
    return match.group(1) if match else None


def is_english_esd_speaker_path(path: str | os.PathLike[str]) -> bool:
    """Return whether an ESD path belongs to the English speaker range 0011-0020."""
    speaker_id = speaker_id_from_esd_path(path)
    if speaker_id is None:
        return False
    try:
        speaker_num = int(speaker_id)
    except ValueError:
        return False
    return ESD_ENGLISH_SPEAKER_MIN <= speaker_num <= ESD_ENGLISH_SPEAKER_MAX


def utterance_id_from_esd_path(path: str | os.PathLike[str]) -> str | None:
    """Extract an ESD utterance ID like ``0011_000123`` from a wav path."""
    basename = os.path.basename(str(path or ""))
    match = re.match(r"(\d{4}_\d{6})\.wav$", basename)
    return match.group(1) if match else None


def resolve_esd_wav_path(audio_root: str | os.PathLike[str], wav_field: str | os.PathLike[str]) -> str | None:
    """Resolve an ESD wav path against the audio root and its parent."""
    wav_value = str(wav_field or "").replace("\\", "/")
    if not wav_value:
        return None

    if os.path.isabs(wav_value) and os.path.isfile(wav_value):
        return wav_value

    root = str(audio_root)
    candidate = os.path.join(root, wav_value)
    if os.path.isfile(candidate):
        return candidate

    parent = os.path.dirname(root.rstrip("/"))
    candidate = os.path.join(parent, wav_value)
    if os.path.isfile(candidate):
        return candidate

    return None


def esd_transcript_paths(audio_root: str | os.PathLike[str], speaker_id: str) -> list[str]:
    """Return common ESD transcript file locations for a speaker."""
    root = str(audio_root)
    return [
        os.path.join(root, "downloads", "esd", speaker_id, f"{speaker_id}.txt"),
        os.path.join(os.path.dirname(root.rstrip("/")), "downloads", "esd", speaker_id, f"{speaker_id}.txt"),
        os.path.join(root, speaker_id, f"{speaker_id}.txt"),
    ]


def read_esd_transcript(
    audio_root: str | os.PathLike[str],
    wav_path: str | os.PathLike[str],
    emotion_labels: tuple[str, ...] = CANONICAL_EMOTIONS,
) -> str | None:
    """Read the ESD transcript for one wav file.

    ESD transcript lines are expected to start with the utterance ID, followed
    by transcript words and optionally a trailing emotion label.
    """
    speaker_id = speaker_id_from_esd_path(wav_path)
    utterance_id = utterance_id_from_esd_path(wav_path)
    if not speaker_id or not utterance_id:
        return None

    transcript_path = next((p for p in esd_transcript_paths(audio_root, speaker_id) if os.path.isfile(p)), None)
    if transcript_path is None:
        return None

    utterance_re = re.compile(rf"^{re.escape(utterance_id)}\b")
    emotion_labels_lower = {label.lower() for label in emotion_labels}

    try:
        with Path(transcript_path).open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or not utterance_re.match(line):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    return None

                last = parts[-1].strip().lower()
                content_parts = parts[1:-1] if last in emotion_labels_lower and len(parts) >= 3 else parts[1:]
                transcript = " ".join(content_parts).strip()
                return transcript or None
    except OSError:
        return None

    return None
