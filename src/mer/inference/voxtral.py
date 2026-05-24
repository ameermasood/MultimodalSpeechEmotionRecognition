"""Voxtral zero-shot inference helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm

from mer.data.labels import CANONICAL_EMOTIONS, normalize_prediction_text
from mer.inference.prompts import SYSTEM_PROMPT, build_emotion_instruction


def build_zero_shot_conversation(
    audio_path: str,
    transcript: str = "",
    use_text: bool = False,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
) -> list[dict]:
    """Build the system+user conversation format used for Voxtral zero-shot inference."""
    file_url = "file://" + os.path.abspath(audio_path)
    content = [
        {"type": "audio_url", "audio_url": file_url},
        {"type": "text", "text": build_emotion_instruction(use_text=use_text, labels=labels)},
    ]
    if use_text and transcript and transcript.strip():
        content.append({"type": "text", "text": f"\nTranscript:\n{transcript.strip()}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def predict_emotion(
    model,
    processor,
    audio_path: str,
    transcript: str = "",
    use_text: bool = False,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
    device: str | torch.device = "cuda",
    max_new_tokens: int = 3,
    do_sample: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.95,
    normalizer: Callable[[str], str] | None = None,
) -> str:
    """Run one zero-shot Voxtral prediction and normalize the generated label."""
    conversation = build_zero_shot_conversation(
        audio_path=audio_path,
        transcript=transcript,
        use_text=use_text,
        labels=labels,
    )
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    if normalizer is not None:
        return normalizer(decoded)
    return normalize_prediction_text(decoded, labels=labels)


def predict_emotion_batch(
    model,
    processor,
    rows: pd.DataFrame,
    use_text: bool,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
    audio_col: str = "audio_path",
    transcript_col: str = "transcript",
    device: str | torch.device = "cuda",
    max_new_tokens: int = 3,
    do_sample: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.95,
    normalizer: Callable[[str], str] | None = None,
) -> list[str]:
    """Run batched zero-shot Voxtral predictions for dataframe rows."""
    conversations = [
        build_zero_shot_conversation(
            audio_path=row[audio_col],
            transcript=str(row.get(transcript_col, "")),
            use_text=use_text,
            labels=labels,
        )
        for _, row in rows.iterrows()
    ]
    inputs = processor.apply_chat_template(conversations, tokenize=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
    if normalizer is not None:
        return [normalizer(text) for text in decoded]
    return [normalize_prediction_text(text, labels=labels) for text in decoded]


def add_zero_shot_predictions(
    model,
    processor,
    df: pd.DataFrame,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
    audio_col: str = "audio_path",
    transcript_col: str = "transcript",
    audio_pred_col: str = "pred_audio",
    text_pred_col: str = "pred_both",
    batch_size: int = 8,
    device: str | torch.device = "cuda",
    normalizer: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Add audio-only and audio-plus-text predictions to a dataframe."""
    output = df.reset_index(drop=True).copy()
    audio_preds: list[str] = []
    text_preds: list[str] = []
    n_batches = (len(output) + batch_size - 1) // batch_size

    for start in tqdm(range(0, len(output), batch_size), total=n_batches, desc="Voxtral batches"):
        batch = output.iloc[start : start + batch_size]
        audio_preds.extend(
            predict_emotion_batch(
                model,
                processor,
                batch,
                use_text=False,
                labels=labels,
                audio_col=audio_col,
                transcript_col=transcript_col,
                device=device,
                normalizer=normalizer,
            )
        )
        text_preds.extend(
            predict_emotion_batch(
                model,
                processor,
                batch,
                use_text=True,
                labels=labels,
                audio_col=audio_col,
                transcript_col=transcript_col,
                device=device,
                normalizer=normalizer,
            )
        )

    output[audio_pred_col] = audio_preds
    output[text_pred_col] = text_preds
    return output
