#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: Zero-Shot Evaluation on Emotional Speech Dataset (ESD)
==============================================================

Usage:
    python3 scripts/evaluation/evaluate_zero_shot_esd.py \
        --meta_dir /path/to/EmoBox \
        --audio_root /path/to/esd \
        --base_model mistralai/Voxtral-Mini-3B-2507 \
        --out_dir /path/to/outputs \
        --fold 2
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

# Configure matplotlib to run in headless environment safely
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mer.data import CANONICAL_EMOTIONS, normalize_prediction_text
from mer.evaluation import classification_metrics
from mer.inference.voxtral import add_zero_shot_predictions
from mer.visualization import (
    plot_confusion_matrices,
    plot_correctness_overlap,
    plot_dominant_confusions,
    plot_per_class_metric_comparison,
    plot_transcript_length_analysis,
    set_premium_plot_style,
)

LABELS = list(CANONICAL_EMOTIONS)
LABELS_LOWER = [x.lower() for x in LABELS]

ESD_LABEL_MAP = {
    "angry": "Angry",
    "happy": "Happy",
    "sad": "Sad",
    "neutral": "Neutral",
}

ESD_LABEL_SYNONYMS = {
    "anger": "Angry",
    "angry": "Angry",
    "happiness": "Happy",
    "happy": "Happy",
    "sadness": "Sad",
    "sad": "Sad",
    "neutral": "Neutral",
    "calm": "Neutral",
}


def parse_esd_key(key: str) -> tuple[str, str]:
    """Parse the speaker and utterance number from an EmoBox ESD key."""
    parts = key.split("-")
    spk = parts[1]  # '0001'
    utt_idx = parts[2]  # '000001'
    return spk, utt_idx


def is_english_speaker(spk: str) -> bool:
    """Return whether the speaker number is in the English range (0011 to 0020)."""
    try:
        val = int(spk)
        return 11 <= val <= 20
    except ValueError:
        return False


def normalize_label(raw_text: str) -> str:
    """Map raw Voxtral output to one ESD label or Unknown."""
    return normalize_prediction_text(
        raw_text,
        labels=LABELS,
        default="Unknown",
        synonyms=ESD_LABEL_SYNONYMS,
    )


def load_esd_test_df(meta_dir: str, audio_root: str, fold: int) -> pd.DataFrame:
    """Load the ESD test split and build absolute path records."""
    split_dirs = [
        os.path.join(meta_dir, "data", "esd", f"fold_{fold}"),
        os.path.join(meta_dir, "esd", f"fold_{fold}"),
    ]
    candidates = [
        os.path.join(split_dir, filename)
        for split_dir in split_dirs
        for filename in (f"esd_test_fold_{fold}.jsonl", f"test_fold_{fold}.jsonl")
    ]
    test_jsonl = next((path for path in candidates if os.path.exists(path)), None)
    if test_jsonl is None:
        raise FileNotFoundError(f"Missing ESD split JSONL. Tried: {candidates}")

    # Preload transcripts for quick lookup
    transcripts = {}
    for spk_val in range(11, 21):
        spk_str = f"{spk_val:04d}"
        txt_path = os.path.join(audio_root, spk_str, f"{spk_str}.txt")
        if not os.path.exists(txt_path):
            txt_path = os.path.join(meta_dir, "downloads", "esd", spk_str, f"{spk_str}.txt")
        if not os.path.exists(txt_path):
            continue

        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        utt_id = parts[0]
                        # Join transcript parts, stripping off a trailing emotion label if present
                        last = parts[-1].strip().lower()
                        words = parts[1:-1] if last in LAB_LOWER_SET else parts[1:]
                        transcripts[(spk_str, utt_id)] = (" ".join(words).strip(), None)
        except OSError:
            pass

    rows = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            key = ex["key"]
            raw_label = ex.get("label", ex.get("emo"))
            if raw_label is None:
                continue

            spk, utt_suffix = parse_esd_key(key)
            if not is_english_speaker(spk):
                continue

            emo = ESD_LABEL_MAP.get(raw_label.lower(), raw_label)
            if emo not in LABELS:
                continue

            utt_id = f"{spk}_{utt_suffix}"
            wav_path = os.path.join(audio_root, spk, emo, f"{utt_id}.wav")

            if not os.path.exists(wav_path):
                cands = glob.glob(os.path.join(audio_root, spk, "**", f"{utt_id}.wav"), recursive=True)
                if cands:
                    wav_path = cands[0]
                else:
                    continue

            sent, _ = transcripts.get((spk, utt_id), ("", None))
            rows.append({
                "key": key,
                "speaker": spk,
                "utt_id": utt_id,
                "audio_path": wav_path,
                "transcript": sent,
                "emotion": emo,
            })

    return pd.DataFrame(rows)


LAB_LOWER_SET = frozenset(LABELS_LOWER)


def main():
    ap = argparse.ArgumentParser(description="Zero-Shot Voxtral Evaluation on ESD Dataset")
    ap.add_argument("--meta_dir", required=True, help="Path to EmoBox repository root")
    ap.add_argument("--audio_root", required=True, help="Path to raw ESD audio dataset root")
    ap.add_argument("--base_model", required=True, help="HuggingFace model ID or local directory")
    ap.add_argument("--out_dir", required=True, help="Output directory for predictions and metrics")
    ap.add_argument("--fold", type=int, default=1, help="ESD dataset test fold index")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for Voxtral prediction")
    ap.add_argument("--device", default="cuda", help="Inference device: 'cuda', 'cpu', or 'mps'")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    print(f"Loading ESD fold {args.fold} split...")
    df = load_esd_test_df(args.meta_dir, args.audio_root, args.fold)
    print(f"Kept {len(df)} English utterances.")
    if df.empty:
        raise RuntimeError("DataFrame is empty - please check dataset directories.")

    # Load model and processor
    print(f"Loading base model {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else {"": "cpu"},
        trust_remote_code=True,
    )
    model.eval()

    # Run batched predictions
    print("Running zero-shot Voxtral predictions (audio-only vs audio+transcript)...")
    df_eval = add_zero_shot_predictions(
        model=model,
        processor=processor,
        df=df,
        labels=LABELS,
        audio_col="audio_path",
        transcript_col="transcript",
        audio_pred_col="pred_audio",
        text_pred_col="pred_both",
        batch_size=args.batch_size,
        device=device,
        normalizer=normalize_label,
    )

    # Save predictions
    pred_path = os.path.join(args.out_dir, "predictions.csv")
    df_eval.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    # Compute metrics
    y_true = df_eval["emotion"].values
    y_pa = df_eval["pred_audio"].values
    y_pb = df_eval["pred_both"].values

    metrics_audio = classification_metrics(y_true, y_pa)
    metrics_both = classification_metrics(y_true, y_pb)

    metrics_out = {
        "dataset": f"ESD_fold_{args.fold}",
        "audio_only": metrics_audio,
        "audio_plus_text": metrics_both,
    }

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    if plt is not None:
        # Plot confusion matrices
        print("Plotting confusion matrices...")
        plot_confusion_matrices(y_true, y_pa, y_pb, LABELS)
        plt.savefig(os.path.join(args.out_dir, "confusion_matrices.png"), dpi=200)
        plt.close()

        # Plot F1 scores
        print("Plotting per-class F1 comparisons...")
        plot_per_class_metric_comparison(
            y_true,
            y_pa,
            y_pb,
            LABELS,
            metric="f1-score",
            title="F1 per emotion: audio vs audio+text (ESD)",
        )
        plt.savefig(os.path.join(args.out_dir, "f1_perclass.png"), dpi=200)
        plt.close()

        # Plot correctness overlap
        print("Plotting correctness overlap...")
        plot_correctness_overlap(y_true, y_pa, y_pb)
        plt.savefig(os.path.join(args.out_dir, "correctness_overlap.png"), dpi=200)
        plt.close()

        # Plot dominant confusions
        print("Plotting dominant confusions...")
        plot_dominant_confusions(y_true, y_pa, y_pb, LABELS)
        plt.savefig(os.path.join(args.out_dir, "dominant_confusions.png"), dpi=200)
        plt.close()

        # Plot transcript length analysis
        print("Plotting transcript length analysis...")
        plot_transcript_length_analysis(df_eval, "transcript", "emotion", "pred_audio", "pred_both", LABELS)
        plt.savefig(os.path.join(args.out_dir, "transcript_length_analysis.png"), dpi=200)
        plt.close()
    else:
        print("[WARN] matplotlib is not installed. Skipping plot generation.")

    print("Zero-shot evaluation successfully completed!")


if __name__ == "__main__":
    main()
