#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: IEMOCAP DoRA-Only Evaluation Framework (Fold 1)
=======================================================

Abstract:
    This is a specialized version of the IEMOCAP evaluation pipeline.
    It performs the same rigorous multimodal ablation (Audio vs Audio+Text)
    and robustness checks, but it enforces a strict filter to ONLY evaluate 
    adapters trained with **Weight-Decomposed LoRA (DoRA)**.

    Key Features:
    1.  **DoRA Auto-Detection**: Inspects `adapter_config.json` to verify `use_dora=True`.
    2.  **Multimodal Metrics**: Accuracy, Balanced Accuracy, F1, and ECE.
    3.  **Voxtral-Safe Inference**: Uses User-Only prompting strategies.


"""

import os, re, json, time, argparse, glob
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

# 1. FIXED IMPORT: Added balanced_accuracy_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    f1_score,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMOS = ["Angry", "Happy", "Sad", "Neutral"]
EMOSET = set(EMOS)

RAW2CANON = {
    "ang": "Angry",
    "hap": "Happy",
    "exc": "Happy",
    "sad": "Sad",
    "neu": "Neutral",
}

# -------------------------
# Basic Utils
# -------------------------

def to_abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s.strip("_")[:120] if s else "adapter"

def infer_gender_from_utt(utt_id: str) -> str:
    if not isinstance(utt_id, str) or not utt_id:
        return "unknown"
    parts = utt_id.split("_")
    if parts:
        sess_code = parts[0]
        if sess_code.endswith("F"): return "female"
        if sess_code.endswith("M"): return "male"
    return "unknown"

def infer_session_from_utt(utt_id: str) -> str:
    m = re.match(r"(Ses\d\d)", utt_id or "")
    return m.group(1) if m else "Unknown"

def normalize_label_from_text(txt: str) -> str:
    if txt is None:
        return "Neutral"
    t = txt.strip().lower()
    if not t:
        return "Neutral"
    first = re.split(r"[\s\.\,\!\?\:\;\-\(\)\[\]\{\}]+", t)[0]
    m = {
        "angry": "Angry", "anger": "Angry",
        "happy": "Happy", "happiness": "Happy",
        "sad": "Sad", "sadness": "Sad",
        "neutral": "Neutral",
    }
    if first in m:
        return m[first]
    for k, v in m.items():
        if re.search(rf"\b{k}\b", t):
            return v
    return "Neutral"

def get_duration_sec(wav_path: str) -> float:
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
# Data Loading
# -------------------------

def load_iemocap_fold_json(meta_dir: str, fold: int) -> Tuple[pd.DataFrame, str]:
    p = os.path.join(meta_dir, "iemocap", f"fold_{fold}", f"iemocap_test_fold_{fold}.json")
    with open(p, "r") as f:
        d = json.load(f)
    df = pd.DataFrame.from_dict(d, orient="index").reset_index().rename(columns={"index": "key"})
    return df, p

def resolve_audio_path(audio_root: str, wav_field: str) -> Optional[str]:
    if not wav_field:
        return None
    w = str(wav_field).replace("\\", "/")
    if os.path.isabs(w) and os.path.isfile(w):
        return w
    c1 = os.path.join(audio_root, w)
    if os.path.isfile(c1):
        return c1
    w2 = w.lstrip("/")
    c2 = os.path.join(audio_root, w2)
    if os.path.isfile(c2):
        return c2
    return None

def load_transcripts(transcript_csv: str) -> pd.DataFrame:
    df = pd.read_csv(transcript_csv)
    if {"file_name", "transcription"}.issubset(df.columns):
        df["utt_id"] = df["file_name"].astype(str)
        return df[["utt_id", "transcription"]].copy()
    if {"utt_id", "transcription"}.issubset(df.columns):
        return df[["utt_id", "transcription"]].copy()
    if {"utt_id", "text"}.issubset(df.columns):
        out = df[["utt_id", "text"]].copy()
        return out.rename(columns={"text": "transcription"})
    raise ValueError(f"Transcript CSV missing expected columns. Found: {df.columns.tolist()}")

def build_dataset_df(meta_dir: str, audio_root: str, fold: int, transcript_csv: Optional[str], add_duration: bool = True):
    df_raw, json_path = load_iemocap_fold_json(meta_dir, fold)
    df_raw["audio_path"] = df_raw["wav"].astype(str).apply(lambda x: resolve_audio_path(audio_root, x))
    df = df_raw[df_raw["audio_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))].copy()

    df["utt_id"] = df["wav"].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    df["gender"] = df["utt_id"].apply(infer_gender_from_utt)
    df["session"] = df["utt_id"].apply(infer_session_from_utt)

    df["raw_label"] = df["emo"].astype(str).str.lower()
    df = df[df["raw_label"].isin(RAW2CANON.keys())].reset_index(drop=True)
    df["label"] = df["raw_label"].map(RAW2CANON)
    df = df[df["label"].isin(EMOSET)].reset_index(drop=True)

    if transcript_csv:
        df_txt = load_transcripts(transcript_csv)
        df_txt["utt_id_norm"] = df_txt["utt_id"].astype(str).apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        df["utt_id_norm"] = df["utt_id"].astype(str)
        df = df.merge(df_txt[["utt_id_norm", "transcription"]], on="utt_id_norm", how="left").drop(columns=["utt_id_norm"])
    else:
        df["transcription"] = ""

    df["transcription"] = df["transcription"].fillna("")

    if add_duration:
        df["duration_sec"] = df["audio_path"].apply(get_duration_sec)

    return df, json_path

# -------------------------
# Model & Adapter
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
            base_model, trust_remote_code=True, quantization_config=bnb, device_map="auto", attn_implementation="sdpa"
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
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    return model

# 2. FIXED: DoRA Adapter Filter Logic
def is_dora_adapter(adapter_dir: str) -> bool:
    """Checks adapter_config.json for use_dora=True."""
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg, "r") as f:
            d = json.load(f)
        return bool(d.get("use_dora", False))
    except Exception:
        return False

def find_adapters(adapters_root: str) -> List[str]:
    adapters_root = to_abs(adapters_root)
    if not os.path.isdir(adapters_root):
        return []
    candidates = []
    for sub in sorted(glob.glob(os.path.join(adapters_root, "*"))):
        if not os.path.isdir(sub):
            continue
        fa = os.path.join(sub, "final_adapter")
        if os.path.isdir(fa):
            candidates.append(fa)
        else:
            candidates.append(sub)
    good = []
    for d in candidates:
        if not os.path.isdir(d):
            continue
        has_cfg = os.path.isfile(os.path.join(d, "adapter_config.json"))
        has_w = any(os.path.isfile(os.path.join(d, x)) for x in ["adapter_model.safetensors", "adapter_model.bin"])
        if has_cfg or has_w:
            good.append(d)
    seen, out = set(), []
    for g in good:
        g = os.path.abspath(g)
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out

# -------------------------
# Inference
# -------------------------

def build_instruction(use_text: bool) -> str:
    label_str = ", ".join(EMOS)
    s = (
        "You are an emotion classifier for speech.\n"
        f"Possible emotions: {label_str}.\n"
        "From the given audio"
    )
    if use_text:
        s += " and its transcript"
    s += (
        ", classify the SPEAKER's emotion.\n"
        f"Answer with EXACTLY one word from this set: {label_str}.\n"
        "Do not add extra words."
    )
    return s

def build_conversation_voxtral(audio_path: str, instruction: str, transcript: str, use_text: bool):
    wav_path = os.path.abspath(audio_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Missing audio file: {wav_path}")

    audio_part = {
        "type": "audio",
        "audio_url": {"url": wav_path},
        "content": wav_path,
        "path": wav_path,
        "url": wav_path,
    }

    content = [audio_part, {"type": "text", "text": instruction}]
    if use_text and transcript and transcript.strip():
        content.append({"type": "text", "text": f"Transcript:\n{transcript.strip()}"})

    return [{"role": "user", "content": content}]

@torch.no_grad()
def predict_one(processor, model, device, audio_path, transcript, use_text, max_new_tokens, do_sample, temperature, top_p):
    instruction = build_instruction(use_text)
    conversation = build_conversation_voxtral(audio_path, instruction, transcript, use_text)

    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, return_dict_in_generate=True, output_scores=True)
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    t0 = time.time()
    out = model.generate(**inputs, **gen_kwargs)
    latency_ms = (time.time() - t0) * 1000.0

    in_len = int(inputs["input_ids"].shape[1])
    seq = out.sequences
    decoded = processor.batch_decode(seq[:, in_len:], skip_special_tokens=True)[0]
    pred = normalize_label_from_text(decoded)

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
# Metrics & Plots
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
            val = f"{mat[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            col = "white" if mat[i, j] > thresh else "black"
            plt.text(j, i, val, ha="center", va="center", color=col)
    plt.ylabel("True")
    plt.xlabel("Pred")
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

def reliability_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    m = np.isfinite(conf)
    conf = conf[m]
    correct = correct[m]
    if len(conf) == 0:
        return float("nan"), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(conf, bins) - 1
    ids = np.clip(ids, 0, n_bins - 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_cnt = np.zeros(n_bins)
    for b in range(n_bins):
        mb = (ids == b)
        if mb.sum() > 0:
            bin_acc[b] = correct[mb].mean()
            bin_conf[b] = conf[mb].mean()
            bin_cnt[b] = mb.sum()
    N = max(1, len(conf))
    ece = float(np.sum((bin_cnt / N) * np.abs(bin_acc - bin_conf)))
    return ece, bin_acc, bin_conf, bin_cnt

def selective_accuracy_curve(conf: np.ndarray, correct: np.ndarray, n_points: int = 20):
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    m = np.isfinite(conf)
    conf = conf[m]
    correct = correct[m]
    if len(conf) == 0:
        return [], [], float("nan")
    order = np.argsort(-conf)
    correct_sorted = correct[order]
    coverages = np.linspace(0.1, 1.0, n_points)
    cov_list, acc_list = [], []
    for c in coverages:
        k = max(1, int(round(c * len(correct_sorted))))
        acc = float(correct_sorted[:k].mean())
        cov_list.append(float(k / len(correct_sorted)))
        acc_list.append(acc)
    risk = np.trapz([1.0 - a for a in acc_list], x=cov_list)
    return cov_list, acc_list, float(risk)

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

def compute_metrics(df: pd.DataFrame) -> Dict:
    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()
    out = {
        "num_samples": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "true_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),
    }
    lat = df["latency_ms"].to_numpy(dtype=float)
    lat = lat[np.isfinite(lat)]
    if len(lat) > 0:
        out["latency_ms_p50"] = float(np.percentile(lat, 50))
        out["latency_ms_p90"] = float(np.percentile(lat, 90))
        out["latency_ms_p99"] = float(np.percentile(lat, 99))
        out["latency_ms_mean"] = float(np.mean(lat))
    if "confidence_first_token" in df.columns:
        conf = df["confidence_first_token"].to_numpy(dtype=float)
        correct = (df["true"] == df["pred"]).to_numpy(dtype=float)
        ece, _, _, _ = reliability_ece(conf, correct)
        out["ece_10bins"] = float(ece) if np.isfinite(ece) else float("nan")
        cover, acc, risk = selective_accuracy_curve(conf, correct)
        out["selective_risk_area"] = float(risk) if np.isfinite(risk) else float("nan")
    return out

def save_classification_report(df: pd.DataFrame, out_csv: str):
    rep = classification_report(df["true"], df["pred"], labels=EMOS, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out_csv, index=True)

def per_group_accuracy(df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    out = {}
    if group_col not in df.columns:
        return out
    for g, sub in df.groupby(group_col):
        if len(sub) == 0:
            continue
        out[str(g)] = float(accuracy_score(sub["true"], sub["pred"]))
    return out

def duration_bins_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    if "duration_sec" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d = d[np.isfinite(d["duration_sec"].values)]
    if len(d) < 10:
        return pd.DataFrame()
    bins = [0, 2, 4, 6, 10, 20, 60, 9999]
    labels = ["<2s","2-4","4-6","6-10","10-20","20-60",">60"]
    d["dur_bin"] = pd.cut(d["duration_sec"], bins=bins, labels=labels, include_lowest=True)
    d["correct"] = (d["true"] == d["pred"]).astype(int)
    g = d.groupby("dur_bin", observed=False)["correct"].agg(["mean","count"]).reset_index()
    return g.rename(columns={"mean":"accuracy"})

def plot_duration_accuracy(g: pd.DataFrame, out_png: str, title: str):
    if g is None or len(g) == 0:
        return
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
    met = compute_metrics(df_mode)
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(met, f, indent=2)
    save_classification_report(df_mode, os.path.join(out_dir, f"classification_report_{tag}.csv"))
    cm = confusion_matrix(df_mode["true"], df_mode["pred"], labels=EMOS)
    plot_confusion(cm, EMOS, os.path.join(out_dir, f"confusion_counts_{tag}.png"), f"Confusion Counts - {tag}", normalize=False)
    plot_confusion(cm, EMOS, os.path.join(out_dir, f"confusion_norm_{tag}.png"), f"Confusion Row-Norm - {tag}", normalize=True)
    plot_latency_hist(df_mode["latency_ms"].to_numpy(dtype=float),
                      os.path.join(out_dir, f"latency_hist_{tag}.png"),
                      f"Latency Histogram - {tag}")
    conf = df_mode["confidence_first_token"].to_numpy(dtype=float)
    correct = (df_mode["true"] == df_mode["pred"]).to_numpy(dtype=float)
    cover, acc, risk = selective_accuracy_curve(conf, correct)
    plot_selective_curve(cover, acc, os.path.join(out_dir, f"selective_curve_{tag}.png"),
                         f"Selective Accuracy vs Coverage (risk={risk:.3f}) - {tag}")
    g = duration_bins_accuracy(df_mode)
    if len(g) > 0:
        g.to_csv(os.path.join(out_dir, f"duration_bins_{tag}.csv"), index=False)
        plot_duration_accuracy(g, os.path.join(out_dir, f"duration_accuracy_{tag}.png"),
                               f"Accuracy vs Duration Bin - {tag}")
    return met

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapters_root", default="")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--transcript_csv", default="")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=3)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--progress_every", type=int, default=100)
    ap.add_argument("--include_base_zero_shot", action="store_true")
    args = ap.parse_args()

    args.meta_dir = to_abs(args.meta_dir)
    args.audio_root = to_abs(args.audio_root)
    args.out_root = to_abs(args.out_root)
    ensure_dir(args.out_root)
    if args.transcript_csv:
        args.transcript_csv = to_abs(args.transcript_csv)

    df, json_path = build_dataset_df(
        args.meta_dir, args.audio_root, fold=args.fold,
        transcript_csv=(args.transcript_csv if args.transcript_csv else None),
        add_duration=True,
    )
    print(f"Fold json: {json_path}", flush=True)
    print(f"Usable samples: {len(df)}", flush=True)
    print("Label counts:", dict(Counter(df["label"].tolist())), flush=True)

    print(f"\nLoading BASE model: {args.base_model}", flush=True)
    processor, base, device = load_base_model(args.base_model, args.load_in_4bit)
    print(f"DEVICE = {device}", flush=True)

    adapter_dirs = []
    if args.adapters_root:
        adapter_dirs.extend(find_adapters(args.adapters_root))
    if args.adapters:
        adapter_dirs.extend([to_abs(a) for a in args.adapters])

    # de-dup
    seen, dedup = set(), []
    for a in adapter_dirs:
        a = os.path.abspath(a)
        if a not in seen:
            seen.add(a)
            dedup.append(a)
    adapter_dirs = dedup

    # ---- DoRA ONLY filter ----
    before = len(adapter_dirs)
    adapter_dirs = [a for a in adapter_dirs if is_dora_adapter(a)]
    print(f"DoRA-only filter: {len(adapter_dirs)}/{before} adapters kept", flush=True)

    entries = []
    if args.include_base_zero_shot:
        entries.append(("ZERO_SHOT_BASE", None))
    for ad in adapter_dirs:
        entries.append((safe_name(os.path.basename(os.path.dirname(ad)) if os.path.basename(ad)=="final_adapter" else os.path.basename(ad)), ad))

    if len(entries) == 0:
        raise RuntimeError("No DoRA adapters found. Use --adapters_root or --adapters (or --include_base_zero_shot).")

    for adapter_name, adapter_dir in entries:
        args.adapter_name = adapter_name
        print("\n==============================", flush=True)
        print(f"ADAPTER: {adapter_name}", flush=True)
        print(f"PATH: {adapter_dir if adapter_dir else '[BASE ZERO-SHOT]'}", flush=True)
        print("==============================", flush=True)

        out_adapter = os.path.join(args.out_root, f"fold_{args.fold}", adapter_name)
        ensure_dir(out_adapter)

        if adapter_dir is None:
            model = base
        else:
            model = load_adapter_on_base(base, adapter_dir)

        df_audio = run_one_mode(
            df, processor, model, device,
            use_text=False, args=args,
            out_csv=os.path.join(out_adapter, "predictions_audio_only.csv")
        )
        save_mode_reports(df_audio, out_adapter, "audio_only")

        df_text = run_one_mode(
            df, processor, model, device,
            use_text=True, args=args,
            out_csv=os.path.join(out_adapter, "predictions_audio_plus_text.csv")
        )
        save_mode_reports(df_text, out_adapter, "audio_plus_text")

        if adapter_dir is not None:
            del model
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"\nDONE. Results saved to: {os.path.join(args.out_root, f'fold_{args.fold}')}", flush=True)

if __name__ == "__main__":
    main()
