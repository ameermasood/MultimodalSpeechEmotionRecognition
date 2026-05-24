#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: Voxtral-mini-3b Multi-Adapter Evaluation Framework (ESD Test Fold)
==========================================================================

Abstract:
    This script performs a rigorous evaluation of fine-tuned PEFT adapters on the 
    Emotional Speech Dataset (ESD) test splits. It goes beyond standard accuracy metrics 
    by incorporating uncertainty quantification and statistical significance testing.

    Key Methodologies:
    1.  **Multimodal Ablation**: Evaluates models in two regimes: 'Audio-Only' vs. 
        'Audio + Text Transcript' to quantify the marginal gain of linguistic context.
    2.  **Uncertainty Quantification**: Computes Expected Calibration Error (ECE) and 
        Selective Accuracy (Risk-Coverage curves) to measure model reliability.
    3.  **Statistical Significance**: Applies McNemar's Test (paired non-parametric) 
        to determine if performance differences between adapters (or modalities) are 
        statistically significant ($p < 0.05$).
    4.  **Operational Metrics**: Tracks inference latency percentiles (p50, p90, p99) 
        crucial for real-time deployment.

Usage:
    Run via command line pointing to the test set metadata and the directory containing 
    fine-tuned adapters.
"""

import os, json, time, argparse, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoProcessor, VoxtralForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mer.data import (
    CANONICAL_EMOTION_SET,
    CANONICAL_EMOTIONS,
    normalize_emotion_name,
    normalize_prediction_text,
    read_esd_transcript,
    resolve_esd_wav_path,
    speaker_id_from_esd_path,
    utterance_id_from_esd_path,
)
from mer.evaluation import classification_metrics, mcnemar_from_two_preds, selective_accuracy_curve
from mer.modeling import adapter_tag_from_path, discover_adapters, resolve_adapter_dir

# Disable parallelism to prevent deadlocks in high-throughput inference environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMOS = list(CANONICAL_EMOTIONS)
KEEP = set(CANONICAL_EMOTION_SET)

# -------------------------
# Utility Functions
# -------------------------

def to_abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# -------------------------
# Data Loading
# -------------------------
def load_esd_test_rows(meta_dir: str, audio_root: str, fold: int) -> Tuple[str, List[dict], int]:
    """Loads the specific JSONL fold file designed for testing."""
    jsonl = os.path.join(meta_dir, "esd", f"fold_{fold}", f"esd_test_fold_{fold}.jsonl")
    rows, missing = [], 0

    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)

            emo = normalize_emotion_name(ex.get("emo", ex.get("label", "")))
            if emo not in KEEP: continue

            wav_rel = (ex.get("wav", "") or "").replace("\\", "/")
            wav_abs = resolve_esd_wav_path(audio_root, wav_rel)
            if not wav_abs:
                missing += 1
                continue

            length = ex.get("length", None)
            spk = speaker_id_from_esd_path(wav_rel) or "Unknown"

            rows.append({
                "utt_id": utterance_id_from_esd_path(wav_abs),
                "wav": wav_abs,
                "label": emo,
                "length_sec": float(length) if length is not None else float("nan"),
                "speaker": str(spk),
            })

    return jsonl, rows, missing
def load_base_and_processor(base_model: str, load_in_4bit: bool):
    """Loads the base Voxtral model (Mistral-Audio) and Processor."""
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
    return processor, base

def wrap_with_adapter(base, adapter_dir: str):
    """Injects LoRA/DoRA weights into the base model."""
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    device = next(model.parameters()).device
    return model, device

# -------------------------
# Inference Logic
# -------------------------
def build_user_only_conversation(wav_path: str, prompt_text: str):
    """
    Constructs a 'User-Only' single-turn chat payload.
    This avoids role validation errors in mistral_common by not forcing an 'assistant' role.
    """
    wav_path = os.path.abspath(wav_path)
    if not os.path.isfile(wav_path):
        raise RuntimeError(f"Missing wav: {wav_path}")
    audio_part = {
        "type": "audio", "audio_url": {"url": wav_path},
        "content": wav_path, "path": wav_path, "url": wav_path,
    }
    return [{"role": "user", "content": [audio_part, {"type": "text", "text": prompt_text}]}]

def infer_one(processor, model, device, wav_path, prompt_text, max_new_tokens=8, do_sample=False, temperature=0.2, top_p=0.95):
    conv = build_user_only_conversation(wav_path, prompt_text)
    inputs = processor.apply_chat_template(conv, tokenize=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, return_dict_in_generate=True, output_scores=True)
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(**inputs, **gen_kwargs)
    latency_ms = (time.time() - t0) * 1000.0

    in_len = int(inputs["input_ids"].shape[1])
    seq = gen_out.sequences
    new_tokens = seq[:, in_len:]
    txt = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    pred = normalize_prediction_text(txt)
    
    # Extract Confidence (Softmax probability of the first generated token)
    conf = float("nan")
    try:
        if gen_out.scores and len(gen_out.scores) > 0:
            s0 = gen_out.scores[0][0] # Score of first token for first batch item
            p0 = torch.softmax(s0.float(), dim=-1)
            chosen_id = int(seq[0, in_len].item())
            conf = float(p0[chosen_id].item())
    except Exception:
        pass

    return pred, txt, conf, latency_ms

def compute_metrics_extended(y_true, y_pred, conf, latencies):
    return classification_metrics(y_true, y_pred, confidence=conf, latencies_ms=latencies)

# -------------------------
# Plotting
# -------------------------
def plot_confusion(mat, labels, out_png, title, normalize=False):
    plt.figure(figsize=(7, 6))
    m = mat.astype(np.float64)
    if normalize:
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1.0)
    plt.imshow(m, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    thresh = float(np.max(m)) / 2.0 if m.size else 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = f"{m[i, j]:.2f}" if normalize else f"{int(mat[i, j])}"
            col = "white" if m[i, j] > thresh else "black"
            plt.text(j, i, val, ha="center", va="center", color=col)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_bar_grouped(df, xcol, series, out_png, title, ylim=(0,1)):
    xvals = df[xcol].tolist()
    x = np.arange(len(xvals))
    w = 0.8 / max(1, len(series))
    plt.figure(figsize=(max(10, len(xvals)*0.7), 5))
    for i, s in enumerate(series):
        plt.bar(x - 0.4 + w/2 + i*w, df[s].values, width=w, label=s)
    plt.xticks(x, xvals, rotation=45, ha="right")
    plt.ylim(*ylim)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# -------------------------
# Execution Logic
# -------------------------
def run_mode(args, mode_name, processor, model, device, rows, out_dir):
    ensure_dir(out_dir)

    base_prompt = "You are an expert at recognizing emotions from speech.\nListen to the audio and output only ONE label from:\nAngry, Happy, Sad, Neutral."
    text_prompt = "You are an expert at recognizing emotions from speech.\nListen to the audio.\nYou also have the transcript (may help disambiguate emotion):\nTRANSCRIPT: {transcript}\n\nOutput only ONE label from:\nAngry, Happy, Sad, Neutral."

    y_true, y_pred, raw_txt, conf1, lat_ms = [], [], [], [], []
    lengths, wavs, transcripts, speakers, utts = [], [], [], [], []
    found_t = 0

    for i, r in enumerate(rows, 1):
        transcript_val = None
        if mode_name == "audio_text":
            t = read_esd_transcript(args.audio_root, r["wav"])
            if t and t.strip():
                transcript_val = t.strip()
                found_t += 1

        prompt = text_prompt.format(transcript=transcript_val) if (mode_name == "audio_text" and transcript_val) else base_prompt

        pred, txt, conf, lat = infer_one(
            processor, model, device, r["wav"], prompt,
            max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
            temperature=args.temperature, top_p=args.top_p
        )

        y_true.append(r["label"])
        y_pred.append(pred)
        raw_txt.append(txt)
        conf1.append(conf)
        lat_ms.append(lat)
        lengths.append(r.get("length_sec", float("nan")))
        wavs.append(r["wav"])
        transcripts.append(transcript_val or "")
        speakers.append(r["speaker"])
        utts.append(r["utt_id"])

        if args.progress_every > 0 and (i % args.progress_every == 0):
            print(f"{mode_name}: {i}/{len(rows)} done", flush=True)

    df = pd.DataFrame({
        "utt_id": utts, "wav": wavs, "speaker": speakers,
        "true": y_true, "pred": y_pred, "raw_text": raw_txt,
        "confidence_first_token": conf1, "latency_ms": lat_ms,
        "length_sec": lengths, "transcript": transcripts,
    })
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    metrics = compute_metrics_extended(y_true, y_pred, conf1, lat_ms)
    if mode_name == "audio_text":
        metrics["transcripts_found_rate"] = float(found_t / max(1, len(rows)))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_helped_plots(df, out_dir, mode_name, metrics["ece_10bins"])
    return df, metrics

def save_helped_plots(df, out_dir, tag, ece):
    cm = confusion_matrix(df["true"], df["pred"], labels=EMOS)
    plot_confusion(cm, EMOS, os.path.join(out_dir, "confusion_counts.png"), f"{tag} Confusion (Counts)", normalize=False)
    plot_confusion(cm, EMOS, os.path.join(out_dir, "confusion_norm.png"), f"{tag} Confusion (Norm)", normalize=True)
    
    # Selective Curve
    correct = (df["true"] == df["pred"]).astype(int)
    cover, acc, risk = selective_accuracy_curve(df["confidence_first_token"].values, correct.values)
    if len(cover) > 0:
        plt.figure(figsize=(7, 5))
        plt.plot(cover, acc, "o-")
        plt.title(f"Selective Accuracy (Risk={risk:.3f}) - {tag}")
        plt.xlabel("Coverage"); plt.ylabel("Accuracy")
        plt.grid(alpha=0.3); plt.savefig(os.path.join(out_dir, "selective_curve.png"), dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Voxtral Multi-Adapter Evaluation Framework")
    ap.add_argument("--meta_dir", required=True, help="Path to ESD metadata root")
    ap.add_argument("--audio_root", required=True, help="Path to audio files root")
    ap.add_argument("--base_model", required=True, help="Base model ID (HuggingFace)")
    ap.add_argument("--out_root", required=True, help="Output directory for evaluations")
    ap.add_argument("--fold", type=int, default=2, help="Test fold index")
    ap.add_argument("--load_in_4bit", action="store_true", help="Use QLoRA 4-bit loading")
    ap.add_argument("--adapters_root", default="", help="Directory containing multiple adapter folders")
    ap.add_argument("--adapter_dirs", nargs="*", default=[], help="Specific adapter paths to evaluate")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--progress_every", type=int, default=200)
    args = ap.parse_args()

    # Normalize filesystem paths. Keep base_model unchanged so it can be a
    # Hugging Face model ID or a local path.
    for k in ["meta_dir", "audio_root", "out_root"]:
        setattr(args, k, to_abs(getattr(args, k)))
    
    fold_root = os.path.join(args.out_root, f"fold_{args.fold}")
    adapters_out_root = os.path.join(fold_root, "adapters")
    global_cmp_root = os.path.join(fold_root, "global_compare")
    ensure_dir(global_cmp_root)

    # Load Test Data
    jsonl_path, rows, missing = load_esd_test_rows(args.meta_dir, args.audio_root, args.fold)
    print(f"=== ESD Fold{args.fold} | Rows: {len(rows)} | Missing: {missing} ===", flush=True)

    # Discover Adapters
    adapter_list = []
    if args.adapters_root:
        adapter_list.extend(discover_adapters(args.adapters_root))
    for p in args.adapter_dirs:
        rd = resolve_adapter_dir(p)
        if rd: adapter_list.append(rd)
    adapter_list = list(set(adapter_list))
    if not adapter_list:
        raise SystemExit("No adapters found.")

    agg_rows = []
    per_adapter_dfs = {}

    for idx, adapter_dir in enumerate(adapter_list, 1):
        tag = adapter_tag_from_path(adapter_dir)
        print(f"\n[{idx}/{len(adapter_list)}] Adapter: {tag}", flush=True)

        processor, base = load_base_and_processor(args.base_model, args.load_in_4bit)
        model, device = wrap_with_adapter(base, adapter_dir)
        
        adapter_root = os.path.join(adapters_out_root, tag)
        df_a, met_a = run_mode(args, "audio_only", processor, model, device, rows, os.path.join(adapter_root, "audio_only"))
        df_b, met_b = run_mode(args, "audio_text", processor, model, device, rows, os.path.join(adapter_root, "audio_text"))

        # Compare Modes (Internal McNemar)
        cmp_dir = os.path.join(adapter_root, "compare"); ensure_dir(cmp_dir)
        mcn = mcnemar_from_two_preds(df_a["true"], df_a["pred"], df_b["pred"])
        with open(os.path.join(cmp_dir, "mcnemar_audio_vs_text.json"), "w") as f: json.dump(mcn, f, indent=2)

        # Helped/Hurt Analysis
        m = pd.merge(df_a, df_b, on="utt_id", suffixes=("_audio", "_text"))
        m[(m["true_audio"]!=m["pred_audio"]) & (m["true_text"]==m["pred_text"])].to_csv(os.path.join(cmp_dir, "helped_by_text.csv"))
        m[(m["true_audio"]==m["pred_audio"]) & (m["true_text"]!=m["pred_text"])].to_csv(os.path.join(cmp_dir, "hurt_by_text.csv"))

        # Aggregate for Leaderboard
        for mode, met in [("audio_only", met_a), ("audio_plus_text", met_b)]:
            row = {"adapter": tag, "mode": mode}
            row.update(met) # metrics
            if mode == "audio_plus_text":
                row.update({k: v for k,v in mcn.items() if k in ["chi_sq_stat", "significant_p05"]})
            agg_rows.append(row)
        
        per_adapter_dfs[tag] = {"audio_only": df_a, "audio_plus_text": df_b}

        del model, base; torch.cuda.empty_cache()

    # --- Global Comparisons ---
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(global_cmp_root, "leaderboard_full.csv"), index=False)

    # Comparison Plots
    piv = df_agg.pivot_table(index="adapter", columns="mode", values=["accuracy", "f1_macro"])
    piv.columns = [f"{a}_{b}" for a,b in piv.columns]
    piv = piv.reset_index()
    plot_bar_grouped(piv, "adapter", [c for c in piv.columns if "accuracy" in c], 
                     os.path.join(global_cmp_root, "compare_accuracy.png"), "Accuracy by Adapter")
    
    # Pairwise McNemar (Audio Only comparison between different adapters)
    adapters = list(per_adapter_dfs.keys())
    pair_rows = []
    for i in range(len(adapters)):
        for j in range(i+1, len(adapters)):
            A, B = adapters[i], adapters[j]
            st = mcnemar_from_two_preds(per_adapter_dfs[A]["audio_only"]["true"], 
                                        per_adapter_dfs[A]["audio_only"]["pred"], 
                                        per_adapter_dfs[B]["audio_only"]["pred"])
            pair_rows.append({"A": A, "B": B, **st})
    pd.DataFrame(pair_rows).to_csv(os.path.join(global_cmp_root, "pairwise_mcnemar_audio_only.csv"), index=False)

    print("\n=== DONE ===", flush=True)

if __name__ == "__main__":
    main()
