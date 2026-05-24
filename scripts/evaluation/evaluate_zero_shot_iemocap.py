#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: Zero-Shot Evaluation on IEMOCAP Dataset
===============================================

Usage:
    python3 scripts/evaluation/evaluate_zero_shot_iemocap.py \
        --meta_path /path/to/EmoBox/data/iemocap/iemocap.json \
        --audio_root /path/to/iemocap \
        --base_model mistralai/Voxtral-Mini-3B-2507 \
        --out_dir /path/to/outputs
"""

import argparse
import json
import os
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

from mer.data import (
    normalize_prediction_text,
)
from mer.data.iemocap import infer_gender_from_utt
from mer.evaluation import classification_metrics
from mer.inference.voxtral import add_zero_shot_predictions
from mer.visualization import (
    plot_confusion_matrices,
    plot_correctness_overlap,
    plot_gender_comparison,
    plot_global_metric_comparison,
    plot_per_class_metric_comparison,
)

LABEL_LIST = ["neutral", "anger", "happiness", "sadness"]

IEMOCAP_LABEL_SYNONYMS = {
    "neutral": "neutral",
    "calm": "neutral",
    "anger": "anger",
    "angry": "anger",
    "happiness": "happiness",
    "happy": "happiness",
    "sadness": "sadness",
    "sad": "sadness",
}


def normalize_label_from_text(txt: str) -> str:
    """Map Voxtral free text to the IEMOCAP label set."""
    return normalize_prediction_text(
        txt,
        labels=LABEL_LIST,
        default="neutral",
        synonyms=IEMOCAP_LABEL_SYNONYMS,
    )


def load_iemocap_test_df(meta_path: str, audio_root: str) -> pd.DataFrame:
    """Load the IEMOCAP EmoBox split and build absolute path records."""
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing EmoBox IEMOCAP JSON: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        test_dict = json.load(f)

    df_raw = pd.DataFrame.from_dict(test_dict, orient="index").reset_index().rename(columns={"index": "key"})

    def abs_audio(rel_path: str) -> str:
        rel = str(rel_path)
        prefix = "downloads/iemocap/"
        if rel.startswith(prefix):
            rel = rel[len(prefix) :]
        return os.path.join(audio_root, rel)

    df_raw["audio_path"] = df_raw["wav"].astype(str).apply(abs_audio)
    df_kept = df_raw[df_raw["audio_path"].apply(os.path.exists)].reset_index(drop=True)

    df_kept["utt_id"] = df_kept["wav"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    df_kept["gender"] = df_kept["utt_id"].apply(infer_gender_from_utt)

    # Filter and map labels
    df_kept["raw_label"] = df_kept["emo"].astype(str).str.lower()
    raw2canon = {
        "ang": "anger",
        "hap": "happiness",
        "exc": "happiness",  # excite -> happiness mapping
        "sad": "sadness",
        "neu": "neutral",
    }
    df_mapped = df_kept[df_kept["raw_label"].isin(raw2canon.keys())].reset_index(drop=True)
    df_mapped["label"] = df_mapped["raw_label"].map(raw2canon)

    return df_mapped


def main():
    ap = argparse.ArgumentParser(description="Zero-Shot Voxtral Evaluation on IEMOCAP Dataset")
    ap.add_argument("--meta_path", required=True, help="Path to EmoBox iemocap.json file")
    ap.add_argument("--audio_root", required=True, help="Path to raw IEMOCAP audio dataset root")
    ap.add_argument("--base_model", required=True, help="HuggingFace model ID or local directory")
    ap.add_argument("--out_dir", required=True, help="Output directory for predictions and metrics")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size for Voxtral prediction")
    ap.add_argument("--device", default="cuda", help="Inference device: 'cuda', 'cpu', or 'mps'")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    print(f"Loading IEMOCAP EmoBox split from {args.meta_path}...")
    df = load_iemocap_test_df(args.meta_path, args.audio_root)
    print(f"Kept {len(df)} test utterances.")
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
        labels=LABEL_LIST,
        audio_col="audio_path",
        transcript_col="transcription",  # transcription is transcript column name in EmoBox IEMOCAP
        audio_pred_col="pred_audio",
        text_pred_col="pred_bimodal",
        batch_size=args.batch_size,
        device=device,
        normalizer=normalize_label_from_text,
    )

    # Save predictions
    pred_path = os.path.join(args.out_dir, "predictions.csv")
    df_eval.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    # Compute metrics
    y_true = df_eval["label"].tolist()
    y_a = df_eval["pred_audio"].tolist()
    y_b = df_eval["pred_bimodal"].tolist()

    metrics_audio = classification_metrics(y_true, y_a)
    metrics_both = classification_metrics(y_true, y_b)

    metrics_out = {
        "dataset": "IEMOCAP",
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
        plot_confusion_matrices(y_true, y_a, y_b, LABEL_LIST)
        plt.savefig(os.path.join(args.out_dir, "confusion_matrices.png"), dpi=200)
        plt.close()

        # Plot global score comparisons
        print("Plotting global metrics comparison...")
        plot_global_metric_comparison(y_true, y_a, y_b, LABEL_LIST)
        plt.savefig(os.path.join(args.out_dir, "global_compare.png"), dpi=200)
        plt.close()

        # Plot per-class accuracy (recall)
        print("Plotting per-class accuracy comparisons...")
        plot_per_class_metric_comparison(
            y_true,
            y_a,
            y_b,
            LABEL_LIST,
            metric="recall",
            title="Per-Class Accuracy (Recall) by Modality (IEMOCAP)",
            ylabel="Accuracy (Recall)",
        )
        plt.savefig(os.path.join(args.out_dir, "recall_perclass.png"), dpi=200)
        plt.close()

        # Plot per-class F1
        print("Plotting per-class F1 comparisons...")
        plot_per_class_metric_comparison(
            y_true,
            y_a,
            y_b,
            LABEL_LIST,
            metric="f1-score",
            title="Per-Class F1: Audio vs Audio+Text (IEMOCAP)",
        )
        plt.savefig(os.path.join(args.out_dir, "f1_perclass.png"), dpi=200)
        plt.close()

        # Plot correctness overlap
        print("Plotting correctness overlap...")
        plot_correctness_overlap(y_true, y_a, y_b)
        plt.savefig(os.path.join(args.out_dir, "correctness_overlap.png"), dpi=200)
        plt.close()

        # Plot gender metrics
        print("Plotting gender comparisons...")
        plot_gender_comparison(df_eval, "gender", "label", "label", "pred_audio", "pred_bimodal", LABEL_LIST)
        plt.savefig(os.path.join(args.out_dir, "gender_compare.png"), dpi=200)
        plt.close()
    else:
        print("[WARN] matplotlib is not installed. Skipping plot generation.")

    print("Zero-shot IEMOCAP evaluation successfully completed!")


if __name__ == "__main__":
    main()
