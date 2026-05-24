"""ESD dataset loading and split helpers for training scripts."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from mer.data.esd import is_english_esd_speaker_path, read_esd_transcript, resolve_esd_wav_path
from mer.data.labels import CANONICAL_EMOTIONS, normalize_emotion_name


def esd_train_jsonl_path(meta_dir: str | os.PathLike[str], fold: int) -> str:
    """Return the conventional ESD training JSONL path for one fold."""
    return os.path.join(str(meta_dir), "esd", f"fold_{fold}", f"esd_train_fold_{fold}.jsonl")


def load_esd_training_records(
    meta_dir: str | os.PathLike[str],
    audio_root: str | os.PathLike[str],
    fold: int,
    labels: tuple[str, ...] = CANONICAL_EMOTIONS,
    include_transcripts: bool = False,
    english_only: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Load ESD training records from metadata and resolve local audio paths."""
    label_set = set(labels)
    jsonl_path = esd_train_jsonl_path(meta_dir, fold)
    records: list[dict[str, str]] = []
    stats = {"rows": 0, "missing_audio": 0, "missing_transcript": 0, "kept": 0}

    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["rows"] += 1
            row: dict[str, Any] = json.loads(line)
            emotion = normalize_emotion_name(row.get("emo", row.get("label", "")))
            wav_rel = str(row.get("wav") or "").replace("\\", "/")

            if emotion not in label_set:
                continue
            if english_only and not is_english_esd_speaker_path(wav_rel):
                continue

            wav_abs = resolve_esd_wav_path(audio_root, wav_rel)
            if wav_abs is None:
                stats["missing_audio"] += 1
                continue

            record = {"audio_path": wav_abs, "label": emotion}
            if include_transcripts:
                transcript = read_esd_transcript(audio_root, wav_abs, emotion_labels=labels) or ""
                if not transcript.strip():
                    stats["missing_transcript"] += 1
                record["transcript"] = transcript
            records.append(record)

    stats["kept"] = len(records)
    return records, stats


def split_balanced_train_val(
    records: list[dict[str, str]],
    val_per_class: int,
    seed: int,
    labels: tuple[str, ...] = CANONICAL_EMOTIONS,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a per-class validation split and class-balanced training split."""
    rng = random.Random(seed)
    by_label = {label: [] for label in labels}
    for record in records:
        label = record.get("label", "")
        if label in by_label:
            by_label[label].append(record)

    train_records: list[dict[str, str]] = []
    val_records: list[dict[str, str]] = []
    for label in labels:
        items = list(by_label[label])
        rng.shuffle(items)
        val_count = min(int(val_per_class), len(items))
        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    if not train_records:
        raise RuntimeError("Train set became empty after carving validation. Reduce val_per_class.")

    train_counts = Counter(record["label"] for record in train_records)
    train_labels = [label for label in labels if train_counts.get(label, 0) > 0]
    min_count = min(train_counts[label] for label in train_labels)

    balanced_train: list[dict[str, str]] = []
    by_train_label = {label: [] for label in labels}
    for record in train_records:
        by_train_label[record["label"]].append(record)
    for label in train_labels:
        balanced_train.extend(by_train_label[label][:min_count])

    rng.shuffle(balanced_train)
    return balanced_train, val_records


def transcript_pool_from_records(records: list[dict[str, str]]) -> list[str]:
    """Collect non-empty transcripts for stochastic transcript corruption."""
    return [record["transcript"] for record in records if record.get("transcript", "").strip()]
