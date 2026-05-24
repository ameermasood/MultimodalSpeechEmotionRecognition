#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: IEMOCAP Multi-Adapter Evaluation Framework (Fold 1 Test)
================================================================

Abstract:
    This script implements a comprehensive evaluation pipeline for Speech Emotion 
    Recognition (SER) models on the IEMOCAP dataset (Fold 1). It is designed to 
    benchmark Voxtral-mini-3b PEFT adapters across multiple dimensions of performance.

    Methodology:
    1.  **Multimodal Ablation**: Each adapter is evaluated in two regimes:
        (A) Audio-Only: Pure acoustic modeling.
        (B) Audio + Text: Multimodal modeling using ground-truth transcripts.
        This quantifies the "Linguistic Gain" contributed by textual context.
    
    2.  **Robustness Analysis**:
        - **Duration Binning**: Disaggregates performance by utterance length (<2s, 2-4s, etc.) 
          to identify length bias.
        - **Demographic Analysis**: Breaks down accuracy by Gender and Session to detect 
          speaker-dependent overfitting.

    3.  **Statistical Significance**:
        - **McNemar's Test**: Applied pairwise between modalities and between different adapters 
          to determine if performance differences are statistically significant ($p < 0.05$).
        - **Confidence Calibration**: Computes Expected Calibration Error (ECE) and Selective 
          Risk (Area Under Risk-Coverage Curve).

    4.  **Operational Metrics**: Tracks inference latency (p50/p90/p99) essential for 
        real-time deployment assessments.


"""

import os, json, time, argparse, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoProcessor, VoxtralForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mer.data import (
    CANONICAL_EMOTION_SET,
    CANONICAL_EMOTIONS,
    infer_gender_from_utt,
    infer_session_from_utt,
    normalize_prediction_text,
    resolve_iemocap_audio_path,
)
from mer.evaluation import classification_metrics, mcnemar_from_two_preds, selective_accuracy_curve
from mer.inference import build_user_only_conversation
from mer.modeling import adapter_tag_from_path, find_adapter_candidates

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Emotion Taxonomy (Canonical) ---
EMOS = list(CANONICAL_EMOTIONS)
EMOSET = set(CANONICAL_EMOTION_SET)

# IEMOCAP raw label mapping (Excited -> Happy is standard practice)
RAW2CANON = {
    "ang": "Angry",
    "hap": "Happy",
    "exc": "Happy",   # common IEMOCAP practice: excited -> Happy
    "sad": "Sad",
    "neu": "Neutral",
}

# -------------------------
# Utility Functions
# -------------------------

def to_abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# -------------------------
# Audio Duration (Robustness Check)
# -------------------------
def get_duration_sec(wav_path: str) -> float:
    """Computes audio duration to analyze performance vs. length."""
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        if info and info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import torchaudio
        x, sr = torchaudio.load(wav_path)
        if sr and x is not None:
            return float(x.shape[-1]) / float(sr)
    except Exception:
        pass
    return float("nan")

# -------------------------
# Data Loading (EmoBox Schema)
# -------------------------

def load_iemocap_fold_json(meta_dir: str, fold: int) -> Tuple[pd.DataFrame, str]:
    p = os.path.join(meta_dir, "iemocap", f"fold_{fold}", f"iemocap_test_fold_{fold}.json")
    with open(p, "r") as f:
        d = json.load(f)
    # EmoBox format is typically dict-of-dicts; convert to DataFrame
    df = pd.DataFrame.from_dict(d, orient="index").reset_index().rename(columns={"index": "key"})
    return df, p

def load_transcripts(transcript_csv: str) -> pd.DataFrame:
    """Flexible CSV loader that handles various column naming conventions."""
    df = pd.read_csv(transcript_csv)
    # Check for standard naming variations
    if {"file_name", "transcription"}.issubset(df.columns):
        df["utt_id"] = df["file_name"].astype(str)
        return df[["utt_id", "transcription"]].copy()
    if {"utt_id", "transcription"}.issubset(df.columns):
        return df[["utt_id", "transcription"]].copy()
    if {"utt_id", "text"}.issubset(df.columns):
        out = df[["utt_id", "text"]].copy()
        out = out.rename(columns={"text": "transcription"})
        return out
    raise ValueError(f"Transcript CSV missing expected columns. Found: {df.columns.tolist()}")

def build_dataset_df(
    meta_dir: str,
    audio_root: str,
    fold: int,
    transcript_csv: Optional[str],
    add_duration: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Constructs the master DataFrame for evaluation.
    Merges metadata, audio paths, ground-truth labels, and optional transcripts.
    """
    df_raw, json_path = load_iemocap_fold_json(meta_dir, fold)

    df_raw["audio_path"] = df_raw["wav"].astype(str).apply(lambda x: resolve_iemocap_audio_path(audio_root, x))
    df = df_raw[df_raw["audio_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))].copy()

    df["utt_id"] = df["wav"].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    df["gender"] = df["utt_id"].apply(infer_gender_from_utt)
    df["session"] = df["utt_id"].apply(infer_session_from_utt)

    # Standardize labels (handle 'exc' -> 'hap' merge)
    df["raw_label"] = df["emo"].astype(str).str.lower()
    df = df[df["raw_label"].isin(RAW2CANON.keys())].reset_index(drop=True)
    df["label"] = df["raw_label"].map(RAW2CANON)
    df = df[df["label"].isin(EMOSET)].reset_index(drop=True)

    if transcript_csv:
        df_txt = load_transcripts(transcript_csv)
        df_txt["utt_id_norm"] = df_txt["utt_id"].astype(str).apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        df["utt_id_norm"] = df["utt_id"].astype(str)
        df = df.merge(df_txt[["utt_id_norm", "transcription"]], on="utt_id_norm", how="left")
        df = df.drop(columns=["utt_id_norm"])
    else:
        df["transcription"] = ""
    df["transcription"] = df["transcription"].fillna("")

    if add_duration:
        df["duration_sec"] = df["audio_path"].apply(get_duration_sec)

    return df, json_path

# -------------------------
# Model Loading
# -------------------------

def load_base_model(base_model: str, load_in_4bit: bool):
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = VoxtralForConditionalGeneration.from_pretrained(
            base_model,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="sdpa",
        )
    else:
        try:
            base = VoxtralForConditionalGeneration.from_pretrained(
                base_model, trust_remote_code=True, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
            )
        except TypeError:
            base = VoxtralForConditionalGeneration.from_pretrained(
                base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
            )
    try:
        base.config.use_cache = True
    except Exception:
        pass
    device = next(base.parameters()).device
    return processor, base, device

def load_adapter_on_base(base, adapter_dir: str):
    """Dynamically loads adapter weights onto the base model."""
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    return model

# -------------------------
# Inference Logic (Voxtral-Safe)
# -------------------------

@torch.no_grad()
def predict_one(
    processor, model, device,
    audio_path: str,
    transcript: str,
    use_text: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float
):
    if not os.path.isfile(os.path.abspath(audio_path)):
        raise FileNotFoundError(f"Missing audio file: {os.path.abspath(audio_path)}")
    conversation = build_user_only_conversation(
        audio_path=audio_path,
        transcript=transcript,
        use_text=use_text,
        labels=EMOS,
    )

    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    t0 = time.time()
    out = model.generate(**inputs, **gen_kwargs)
    latency_ms = (time.time() - t0) * 1000.0

    in_len = int(inputs["input_ids"].shape[1])
    seq = out.sequences
    decoded = processor.batch_decode(seq[:, in_len:], skip_special_tokens=True)[0]
    pred = normalize_prediction_text(decoded)

    # Confidence: Softmax probability of the FIRST generated token
    # (Proxy for model certainty)
    conf = float("nan")
    try:
        if out.scores and len(out.scores) > 0:
            logits0 = out.scores[0][0].float()
            p0 = torch.softmax(logits0, dim=-1)
            chosen_id = int(seq[0, in_len].item())
            conf = float(p0[chosen_id].item())
    except Exception:
        pass

    return pred, decoded, conf, latency_ms

# -------------------------
# Plotting Helpers
# -------------------------

def plot_confusion(cm: np.ndarray, labels: List[str], out_png: str, title: str, normalize: bool):
    plt.figure(figsize=(7, 6))
    mat = cm.astype(np.float64)
    if normalize:
        denom = np.maximum(mat.sum(axis=1, keepdims=True), 1.0)
        mat = mat / denom

    plt.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = mat.max() / 2.0 if mat.size else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = f"{mat[i, j]:.2f}" if normalize else f"{int(mat[i, j])}"
            col = "white" if mat[i, j] > thresh else "black"
            plt.text(j, i, val, ha="center", va="center", color=col)

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_bar_grouped(df: pd.DataFrame, xcol: str, series: List[str], out_png: str, title: str, ylim=(0,1.0)):
    xvals = df[xcol].tolist()
    x = np.arange(len(xvals))
    w = 0.8 / max(1, len(series))
    plt.figure(figsize=(max(10, len(xvals)*0.65), 5))
    for i, s in enumerate(series):
        plt.bar(x - 0.4 + w/2 + i*w, df[s].values, width=w, label=s)
    plt.xticks(x, xvals, rotation=45, ha="right")
    plt.ylim(*ylim)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_latency_hist(lat_ms: np.ndarray, out_png: str, title: str):
    lat_ms = np.asarray(lat_ms, dtype=np.float64)
    lat_ms = lat_ms[np.isfinite(lat_ms)]
    if len(lat_ms) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(lat_ms, bins=40, density=False, alpha=0.8)
    plt.title(title)
    plt.xlabel("Latency (ms) per sample")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_selective_curve(cover, acc, out_png: str, title: str):
    if len(cover) == 0:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(cover, acc, "o-")
    plt.title(title)
    plt.xlabel("Coverage (fraction kept)")
    plt.ylabel("Accuracy on kept")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# -------------------------
# Metrics & Reporting
# -------------------------

def compute_metrics(df: pd.DataFrame) -> Dict:
    return classification_metrics(
        df["true"].tolist(),
        df["pred"].tolist(),
        confidence=df["confidence_first_token"].to_numpy(dtype=float) if "confidence_first_token" in df.columns else None,
        latencies_ms=df["latency_ms"].to_numpy(dtype=float) if "latency_ms" in df.columns else None,
    )

def save_classification_report(df: pd.DataFrame, out_csv: str):
    rep = classification_report(df["true"], df["pred"], labels=EMOS, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out_csv, index=True)

def per_group_accuracy(df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    out = {}
    if group_col not in df.columns:
        return out
    for g, sub in df.groupby(group_col):
        if len(sub) == 0: continue
        out[str(g)] = float(accuracy_score(sub["true"], sub["pred"]))
    return out

def duration_bins_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Groups accuracy by audio duration bins to test robustness vs length."""
    if "duration_sec" not in df.columns: return pd.DataFrame()
    d = df.copy()
    d = d[np.isfinite(d["duration_sec"].values)]
    if len(d) < 10: return pd.DataFrame()
    bins = [0, 2, 4, 6, 10, 20, 60, 9999]
    labels = ["<2s","2-4","4-6","6-10","10-20","20-60",">60"]
    d["dur_bin"] = pd.cut(d["duration_sec"], bins=bins, labels=labels, include_lowest=True)
    d["correct"] = (d["true"] == d["pred"]).astype(int)
    g = d.groupby("dur_bin", observed=False)["correct"].agg(["mean","count"]).reset_index()
    g = g.rename(columns={"mean":"accuracy"})
    return g

def plot_duration_accuracy(g: pd.DataFrame, out_png: str, title: str):
    if g is None or len(g) == 0: return
    x = np.arange(len(g))
    plt.figure(figsize=(9,5))
    plt.bar(x, g["accuracy"].values, edgecolor="black")
    plt.xticks(x, g["dur_bin"].astype(str).values, rotation=0)
    plt.ylim(0,1.05)
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.3)
    for i, (a, c) in enumerate(zip(g["accuracy"].values, g["count"].values)):
        plt.text(i, a+0.02, f"n={int(c)}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# -------------------------
# Execution Logic
# -------------------------

def run_one_mode(df: pd.DataFrame, processor, model, device, use_text: bool, args, out_csv: str) -> pd.DataFrame:
    y_true, y_pred, raw_txt, conf1, lat_ms = [], [], [], [], []

    for i, row in df.iterrows():
        pred, decoded, conf, lat = predict_one(
            processor, model, device,
            audio_path=row["audio_path"],
            transcript=row["transcription"],
            use_text=use_text,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        y_true.append(row["label"])
        y_pred.append(pred)
        raw_txt.append(decoded)
        conf1.append(conf)
        lat_ms.append(lat)

        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            tag = "audio_plus_text" if use_text else "audio_only"
            print(f"[{args.adapter_name}|{tag}] {i+1}/{len(df)}", flush=True)

    out = df.copy()
    out["true"] = y_true
    out["pred"] = y_pred
    out["raw_text"] = raw_txt
    out["confidence_first_token"] = conf1
    out["latency_ms"] = lat_ms
    out.to_csv(out_csv, index=False)
    return out

def save_mode_reports(df_mode: pd.DataFrame, out_dir: str, tag: str):
    ensure_dir(out_dir)
    # Metrics JSON
    met = compute_metrics(df_mode)
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(met, f, indent=2)

    # Full Reports
    save_classification_report(df_mode, os.path.join(out_dir, f"classification_report_{tag}.csv"))

    # Visualizations
    cm = confusion_matrix(df_mode["true"], df_mode["pred"], labels=EMOS)
    plot_confusion(cm, EMOS, os.path.join(out_dir, f"confusion_counts_{tag}.png"), f"Confusion Counts - {tag}", normalize=False)
    plot_confusion(cm, EMOS, os.path.join(out_dir, f"confusion_norm_{tag}.png"), f"Confusion Row-Norm - {tag}", normalize=True)

    plot_latency_hist(df_mode["latency_ms"].to_numpy(dtype=float),
                      os.path.join(out_dir, f"latency_hist_{tag}.png"),
                      f"Latency Histogram - {tag}")

    # Selective Curve
    conf = df_mode["confidence_first_token"].to_numpy(dtype=float)
    correct = (df_mode["true"] == df_mode["pred"]).to_numpy(dtype=float)
    cover, acc, risk = selective_accuracy_curve(conf, correct)
    if len(cover) > 0:
        plot_selective_curve(cover, acc, os.path.join(out_dir, f"selective_curve_{tag}.png"),
                             f"Selective Accuracy vs Coverage (risk={risk:.3f}) - {tag}")

    # Group Analysis
    gacc = per_group_accuracy(df_mode, "gender")
    sacc = per_group_accuracy(df_mode, "session")
    if gacc: pd.DataFrame([{"group":k, "accuracy":v} for k,v in gacc.items()]).to_csv(os.path.join(out_dir, f"gender_acc_{tag}.csv"), index=False)
    if sacc: pd.DataFrame([{"group":k, "accuracy":v} for k,v in sacc.items()]).to_csv(os.path.join(out_dir, f"session_acc_{tag}.csv"), index=False)

    # Duration Analysis
    g = duration_bins_accuracy(df_mode)
    if len(g) > 0:
        g.to_csv(os.path.join(out_dir, f"duration_bins_{tag}.csv"), index=False)
        plot_duration_accuracy(g, os.path.join(out_dir, f"duration_accuracy_{tag}.png"),
                               f"Accuracy vs Duration Bin - {tag}")

    return met

def helped_hurt_audio_vs_text(df_audio: pd.DataFrame, df_text: pd.DataFrame, out_dir: str, adapter_tag: str):
    ensure_dir(out_dir)
    m = df_audio[["utt_id","true","pred","audio_path","transcription"]].merge(
        df_text[["utt_id","pred"]].rename(columns={"pred":"pred_text"}),
        on="utt_id",
        how="inner"
    )
    m["corr_audio"] = (m["pred"] == m["true"])
    m["corr_text"]  = (m["pred_text"] == m["true"])

    helped = m[(~m["corr_audio"]) & (m["corr_text"])].copy()
    hurt   = m[(m["corr_audio"]) & (~m["corr_text"])].copy()

    helped.to_csv(os.path.join(out_dir, f"helped_by_text_{adapter_tag}.csv"), index=False)
    hurt.to_csv(os.path.join(out_dir, f"hurt_by_text_{adapter_tag}.csv"), index=False)

    cm_shift = confusion_matrix(m["pred"], m["pred_text"], labels=EMOS)
    plot_confusion(cm_shift, EMOS, os.path.join(out_dir, f"modality_shift_{adapter_tag}.png"),
                   f"Pred Shift: Audio -> Audio+Text ({adapter_tag})", normalize=False)

    st = mcnemar_from_two_preds(m["true"].values, m["pred"].values, m["pred_text"].values)
    with open(os.path.join(out_dir, f"mcnemar_audio_vs_text_{adapter_tag}.json"), "w") as f:
        json.dump(st, f, indent=2)
    return st

def pairwise_mcnemar_same_modality(dfA: pd.DataFrame, dfB: pd.DataFrame) -> Dict:
    m = dfA[["utt_id","true","pred"]].merge(dfB[["utt_id","pred"]], on="utt_id", suffixes=("_a","_b"), how="inner")
    return mcnemar_from_two_preds(m["true"].values, m["pred_a"].values, m["pred_b"].values)

def best_vs_rest_helped(df_best: pd.DataFrame, df_other: pd.DataFrame, out_csv: str):
    m = df_best[["utt_id","true","pred","audio_path","transcription"]].merge(
        df_other[["utt_id","pred"]].rename(columns={"pred":"pred_other"}),
        on="utt_id",
        how="inner"
    )
    m["best_correct"] = (m["pred"] == m["true"])
    m["other_correct"] = (m["pred_other"] == m["true"])
    helped = m[(m["best_correct"]) & (~m["other_correct"])].copy()
    helped.to_csv(out_csv, index=False)

# -------------------------
# Main Execution
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="IEMOCAP Multi-Adapter Evaluation Framework")

    ap.add_argument("--meta_dir", required=True, help="Path to IEMOCAP metadata")
    ap.add_argument("--audio_root", required=True, help="Path to audio root")
    ap.add_argument("--base_model", required=True, help="Voxtral base model ID")

    ap.add_argument("--adapters_root", default="", help="Directory with multiple adapters")
    ap.add_argument("--adapters", nargs="*", default=[], help="Specific adapter paths")

    ap.add_argument("--transcript_csv", default="", help="Optional CSV for transcripts")
    ap.add_argument("--out_root", required=True, help="Output directory")

    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--load_in_4bit", action="store_true", help="Use QLoRA 4-bit loading")
    ap.add_argument("--max_new_tokens", type=int, default=3)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--progress_every", type=int, default=100)

    ap.add_argument("--include_base_zero_shot", action="store_true", help="Eval base model without adapters")

    args = ap.parse_args()

    # Path Normalization
    args.meta_dir = to_abs(args.meta_dir)
    args.audio_root = to_abs(args.audio_root)
    args.out_root = to_abs(args.out_root)
    ensure_dir(args.out_root)
    if args.transcript_csv:
        args.transcript_csv = to_abs(args.transcript_csv)

    # 1. Load Data
    df, json_path = build_dataset_df(
        args.meta_dir, args.audio_root, fold=args.fold,
        transcript_csv=(args.transcript_csv if args.transcript_csv else None),
        add_duration=True,
    )
    print(f"Fold json: {json_path}", flush=True)
    print(f"Usable samples: {len(df)}", flush=True)
    print("Label counts:", dict(Counter(df["label"].tolist())), flush=True)

    # 2. Load Base Model
    print(f"\nLoading BASE model: {args.base_model}", flush=True)
    processor, base, device = load_base_model(args.base_model, args.load_in_4bit)
    print(f"DEVICE = {device}", flush=True)

    # 3. Discover Adapters
    adapter_dirs = []
    if args.adapters_root:
        adapter_dirs.extend(find_adapter_candidates(args.adapters_root))
    if args.adapters:
        adapter_dirs.extend([to_abs(a) for a in args.adapters])
    
    seen = set()
    dedup = []
    for a in adapter_dirs:
        a = os.path.abspath(a)
        if a not in seen:
            seen.add(a)
            dedup.append(a)
    adapter_dirs = dedup

    entries = []
    if args.include_base_zero_shot:
        entries.append(("ZERO_SHOT_BASE", None))
    for ad in adapter_dirs:
        entries.append((adapter_tag_from_path(ad), ad))

    if len(entries) == 0:
        raise RuntimeError("No adapters found.")

    # 4. Evaluation Loop
    agg_rows = []
    per_adapter_predictions = {}

    for adapter_name, adapter_dir in entries:
        args.adapter_name = adapter_name
        print(f"\n=== EVALUATING: {adapter_name} ===", flush=True)
        out_adapter = os.path.join(args.out_root, f"fold_{args.fold}", adapter_name)
        ensure_dir(out_adapter)

        if adapter_dir is None:
            model = base 
        else:
            model = load_adapter_on_base(base, adapter_dir)

        # Mode A: Audio Only
        df_audio = run_one_mode(df, processor, model, device, use_text=False, args=args,
                                out_csv=os.path.join(out_adapter, "predictions_audio_only.csv"))
        met_audio = save_mode_reports(df_audio, out_adapter, "audio_only")

        # Mode B: Audio + Text
        df_text = run_one_mode(df, processor, model, device, use_text=True, args=args,
                               out_csv=os.path.join(out_adapter, "predictions_audio_plus_text.csv"))
        met_text = save_mode_reports(df_text, out_adapter, "audio_plus_text")

        # Compare Modalities
        cmp_dir = os.path.join(out_adapter, "compare")
        ensure_dir(cmp_dir)
        mcn = helped_hurt_audio_vs_text(df_audio, df_text, cmp_dir, adapter_name)

        agg_rows.append({
            "adapter": adapter_name, "mode": "audio_only",
            **{k: met_audio.get(k, np.nan) for k in ["accuracy","balanced_accuracy","f1_macro","f1_weighted","mcc","kappa","ece_10bins","latency_ms_p50","latency_ms_p90","latency_ms_p99","selective_risk_area"]},
        })
        agg_rows.append({
            "adapter": adapter_name, "mode": "audio_plus_text",
            **{k: met_text.get(k, np.nan) for k in ["accuracy","balanced_accuracy","f1_macro","f1_weighted","mcc","kappa","ece_10bins","latency_ms_p50","latency_ms_p90","latency_ms_p99","selective_risk_area"]},
            "mcnemar_chi2_audio_vs_text": mcn.get("chi_sq_stat", np.nan),
            "mcnemar_sig_p05_audio_vs_text": bool(mcn.get("significant_p05", False)),
        })

        per_adapter_predictions[adapter_name] = {"audio_only": df_audio, "audio_plus_text": df_text}

        if adapter_dir is not None:
            del model
            torch.cuda.empty_cache()

    # 5. Global Aggregation
    agg_dir = os.path.join(args.out_root, f"fold_{args.fold}", "AGGREGATE")
    ensure_dir(agg_dir)

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(agg_dir, "leaderboard_all_adapters_all_modes.csv"), index=False)

    # Pivot Plots
    piv = df_agg.pivot_table(index="adapter", columns="mode", values=["accuracy","f1_macro"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a,b in piv.columns]
    piv = piv.reset_index()

    if len(piv) > 0:
        cols_acc = [c for c in piv.columns if c.startswith("accuracy_")]
        cols_f1  = [c for c in piv.columns if c.startswith("f1_macro_")]
        plot_bar_grouped(piv, "adapter", cols_acc, os.path.join(agg_dir, "compare_accuracy_by_adapter.png"), "Accuracy per Adapter")
        plot_bar_grouped(piv, "adapter", cols_f1, os.path.join(agg_dir, "compare_macro_f1_by_adapter.png"), "Macro-F1 per Adapter")

    # Modality Gain
    gain_rows = []
    for adapter, dd in per_adapter_predictions.items():
        a, t = dd["audio_only"], dd["audio_plus_text"]
        gain_rows.append({
            "adapter": adapter,
            "acc_gain": float(accuracy_score(t["true"], t["pred"]) - accuracy_score(a["true"], a["pred"])),
            "f1m_gain": float(f1_score(t["true"], t["pred"], average="macro") - f1_score(a["true"], a["pred"], average="macro")),
        })
    pd.DataFrame(gain_rows).sort_values("f1m_gain", ascending=False).to_csv(os.path.join(agg_dir, "modality_gain_audio_to_text.csv"), index=False)

    # Pairwise Adapter Comparison (McNemar)
    adapters = list(per_adapter_predictions.keys())
    for mode in ["audio_only", "audio_plus_text"]:
        pair_rows = []
        for i in range(len(adapters)):
            for j in range(i+1, len(adapters)):
                A, B = adapters[i], adapters[j]
                st = pairwise_mcnemar_same_modality(per_adapter_predictions[A][mode], per_adapter_predictions[B][mode])
                pair_rows.append({"mode": mode, "A": A, "B": B, "chi2": st["chi_sq_stat"], "sig_p05": st["significant_p05"]})
        pd.DataFrame(pair_rows).to_csv(os.path.join(agg_dir, f"pairwise_mcnemar_{mode}.csv"), index=False)

    print(f"\nDONE. Results saved to: {os.path.join(args.out_root, f'fold_{args.fold}')}", flush=True)

if __name__ == "__main__":
    main()
