#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: Voxtral-mini-3b DoRA Fine-Tuning with Stochastic Transcript Gating
==========================================================================

Abstract:
    This script implements a robust fine-tuning pipeline for the Voxtral-mini-3b 
    multimodal model using **Weight-Decomposed Low-Rank Adaptation (DoRA)**. 
    
    Novel Contribution - "Implicit Gating":
    Instead of adding an explicit architectural gate to weigh audio vs. text, this 
    script induces gating behavior via **Stochastic Data Augmentation**:
    1.  **Modality Dropout ($P_{drop}$):** Transcripts are randomly omitted during 
        training, forcing the model to rely solely on audio features.
    2.  **Modality Corruption ($P_{corrupt}$):** Transcripts are randomly swapped 
        with mismatched text, forcing the model to detect and ignore unreliable 
        linguistic cues in favor of acoustic prosody.

    Methodology:
    - **Optimization:** DoRA (Rank $r=16$, Alpha $\alpha=32$) decomposes updates 
      into magnitude and direction components for stable learning.
    - **Data Processing:** Dynamically loads ESD fold splits and parses speaker-specific 
      transcript files.
    - **Prompt Engineering:** Uses a "User-Only" prompt structure to bypass role 
      validation constraints.


"""

import os
# Disable tokenizer parallelism to prevent deadlocks in multi-process DataLoaders.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import random
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from transformers import (
    AutoProcessor,
    VoxtralForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

EMOS = ["Angry", "Happy", "Sad", "Neutral"]
KEEP = set(EMOS)

# -------------------------
# Utils
# -------------------------

def to_abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def norm_emo(e: str) -> str:
    return (e or "").strip().capitalize()

def speaker_is_english(wav_rel: str) -> bool:
    """Filters for English speakers (IDs 11-20) in ESD."""
    m = re.search(r"downloads/esd/(\d{4})/", (wav_rel or "").replace("\\", "/"))
    if not m:
        return False
    spk = int(m.group(1))
    return 11 <= spk <= 20

def speaker_id_from_esd_path(wav_rel_or_abs: str) -> Optional[str]:
    s = (wav_rel_or_abs or "").replace("\\", "/")
    m = re.search(r"downloads/esd/(\d{4})/", s)
    return m.group(1) if m else None

def utt_id_from_wav(wav_abs: str) -> Optional[str]:
    base = os.path.basename(wav_abs)
    m = re.match(r"(\d{4}_\d{6})\.wav$", base)
    return m.group(1) if m else None

def read_esd_transcript(audio_root: str, wav_abs: str) -> Optional[str]:
    """
    Parses ESD transcript files.
    Structure: UTT_ID [Transcript] [EmotionLabel]
    We strip the ID and the optional trailing emotion label.
    """
    spk = speaker_id_from_esd_path(wav_abs)
    utt = utt_id_from_wav(wav_abs)
    if not spk or not utt:
        return None

    paths_to_try = [
        os.path.join(audio_root, "downloads", "esd", spk, f"{spk}.txt"),
        os.path.join(os.path.dirname(audio_root.rstrip("/")), "downloads", "esd", spk, f"{spk}.txt"),
        os.path.join(audio_root, spk, f"{spk}.txt"),
    ]

    txt_path = None
    for p in paths_to_try:
        if os.path.isfile(p):
            txt_path = p
            break
    if not txt_path:
        return None

    utt_re = re.compile(rf"^{re.escape(utt)}\b")
    emo_lower = set([e.lower() for e in EMOS])

    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if not utt_re.match(line): continue

                parts = line.split()
                if len(parts) < 2: return None

                last = parts[-1].strip().lower()
                if last in emo_lower and len(parts) >= 3:
                    content_parts = parts[1:-1]
                else:
                    content_parts = parts[1:]

                txt = " ".join(content_parts).strip()
                return txt if txt else None
    except Exception:
        return None
    return None

class SimpleLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        if "loss" in logs:
            print(f"step={state.global_step} | loss={logs['loss']:.4f}", flush=True)
        if "eval_loss" in logs:
            print(f"epoch={state.epoch:.2f} | step={state.global_step} | eval_loss={logs['eval_loss']:.4f}", flush=True)

# -------------------------
# Transform: The "Gating" Logic
# -------------------------

class VoxtralChatAudioTextGateTransform:
    """
    Data Transform with Stochastic Gating.
    
    Implements the training tricks to force robustness:
    1. text_drop_prob: Simulates "Audio-Only" conditions.
    2. text_corrupt_prob: Simulates "Bad Transcript" conditions (Negative Sampling).
    """
    def __init__(
        self,
        processor,
        prompt_text: str,
        max_new_tokens: int = 8,
        text_drop_prob: float = 0.5,
        text_corrupt_prob: float = 0.15,
        transcript_pool: Optional[List[str]] = None,
        seed: int = 42,
        debug_once: bool = True,
    ):
        self.proc = processor
        self.tok = processor.tokenizer
        self.prompt_text = prompt_text
        self.max_new_tokens = int(max_new_tokens)

        self.text_drop_prob = float(text_drop_prob)
        self.text_corrupt_prob = float(text_corrupt_prob)
        self.pool = [t for t in (transcript_pool or []) if t and t.strip()]
        self.rng = random.Random(seed + 12345)

        self.debug_once = bool(debug_once)
        self._printed = False

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _maybe_use_transcript(self, transcript: str) -> Tuple[bool, str, str]:
        """
        Decision Logic for Transcript Injection.
        """
        t = (transcript or "").strip()
        if not t:
            return False, "", "no_transcript"

        # 1. Modality Dropout: Randomly omit valid transcript
        if self.rng.random() < self.text_drop_prob:
            return False, "", "dropped"

        # 2. Modality Corruption: Randomly swap with a distractor
        if self.pool and (self.rng.random() < self.text_corrupt_prob):
            t2 = self.pool[self.rng.randrange(0, len(self.pool))]
            if t2 and t2.strip():
                return True, t2.strip(), "corrupted"

        return True, t, "kept"

    def _user_content(self, wav_path: str, transcript: str):
        used, t_use, reason = self._maybe_use_transcript(transcript)

        content = [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": self.prompt_text},
        ]

        if used and t_use:
            # Explicitly warn the model about potential unreliability
            content.append({"type": "text", "text": f"Transcript (may be missing or incorrect):\n{t_use}"})

        return content, used, reason

    def _encode_one(self, wav_path: str, transcript: str, label_text: str) -> Dict[str, torch.Tensor]:
        user_content, used_t, why = self._user_content(wav_path, transcript)
        # Single-turn User message to avoid 'assistant' role constraints
        msgs = [{"role": "user", "content": user_content}]

        enc = self.proc.apply_chat_template(
            msgs,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        prefix_ids = enc["input_ids"][0]
        prefix_attn = enc["attention_mask"][0]
        prefix_len = int(prefix_ids.numel())

        # Append Label
        lab_ids = self.tok.encode(" " + label_text, add_special_tokens=False)
        if self.tok.eos_token_id is not None:
            lab_ids = lab_ids + [self.tok.eos_token_id]
        lab_ids = lab_ids[: max(1, self.max_new_tokens)]
        lab = torch.tensor(lab_ids, dtype=torch.long)

        input_ids = torch.cat([prefix_ids, lab], dim=0)
        attention_mask = torch.cat([prefix_attn, torch.ones_like(lab)], dim=0)

        # Mask Loss on Prefix
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # Copy audio features
        for k, v in enc.items():
            if k in out: continue
            if torch.is_tensor(v): out[k] = v[0]

        if self.debug_once and (not self._printed):
            self._printed = True
            print("DEBUG (first sample) transcript_gate:", {"used": used_t, "reason": why}, flush=True)
            print("DEBUG out keys:", sorted(list(out.keys())), flush=True)
            print("DEBUG prefix_len:", prefix_len, "| total_len:", int(input_ids.numel()), flush=True)

        return out

    def __call__(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        is_batched = isinstance(ex["audio_path"], list)
        def as_list(x): return x if isinstance(x, list) else [x]

        audio_paths = as_list(ex["audio_path"])
        transcripts = as_list(ex.get("transcript", [""] * len(audio_paths)))
        labels_txt  = as_list(ex["label"])

        outs = [self._encode_one(wp, tr, lb) for wp, tr, lb in zip(audio_paths, transcripts, labels_txt)]
        keys = outs[0].keys()
        packed = {k: [o[k] for o in outs] for k in keys}
        return packed if is_batched else {k: v[0] for k, v in packed.items()}

# -------------------------
# Collator
# -------------------------

def _pad_right_to_shape(t: torch.Tensor, target_shape: List[int], pad_value: float = 0.0) -> torch.Tensor:
    if list(t.shape) == target_shape: return t
    pad = []
    for cur, tgt in zip(reversed(t.shape), reversed(target_shape)):
        pad.extend([0, int(tgt) - int(cur)])
    return F.pad(t, pad, mode="constant", value=pad_value)

@dataclass
class VoxtralPaddingCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            tensors = [f[key] for f in features]
            max_len = max(int(t.size(0)) for t in tensors)
            if key == "labels": pv = -100
            elif key == "attention_mask": pv = 0
            else: pv = self.pad_token_id
            padded = []
            for t in tensors:
                diff = max_len - int(t.size(0))
                if diff > 0:
                    pad = torch.full((diff,), pv, dtype=t.dtype)
                    padded.append(torch.cat([t, pad], dim=0))
                else:
                    padded.append(t)
            batch[key] = torch.stack(padded)

        extra_keys = set(features[0].keys()) - {"input_ids", "attention_mask", "labels"}
        for key in sorted(extra_keys):
            v0 = features[0][key]
            if not torch.is_tensor(v0): continue
            tensors = [f[key] for f in features]
            nd = tensors[0].dim()
            if any(t.dim() != nd for t in tensors): continue
            max_shape = [max(int(t.shape[d]) for t in tensors) for d in range(nd)]
            pv = 0.0 if tensors[0].dtype.is_floating_point else 0
            batch[key] = torch.stack([_pad_right_to_shape(t, max_shape, pv) for t in tensors])
        return batch

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_bs", type=int, default=1)
    ap.add_argument("--eval_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_per_class", type=int, default=100)

    # Transcript Gate Knobs
    ap.add_argument("--text_drop_prob", type=float, default=0.5, help="Prob to drop transcript entirely")
    ap.add_argument("--text_corrupt_prob", type=float, default=0.15, help="Prob to swap transcript with random sample")

    ap.add_argument("--load_in_4bit", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)
    safe_makedirs(args.output_dir)

    prompt_text = (
        "You are an expert at recognizing emotions from speech.\n"
        "Listen to the audio carefully. The transcript may help, but it may also be missing or incorrect.\n"
        "Output only ONE label from:\n"
        "Angry, Happy, Sad, Neutral."
    )

    # -------- Load Data --------
    jsonl_path = os.path.join(args.meta_dir, "esd", f"fold_{args.fold}", f"esd_train_fold_{args.fold}.jsonl")
    with open(jsonl_path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    recs = []
    missing_audio = 0
    for r in rows:
        emo = norm_emo(r.get("emo", r.get("label", "")))
        wav_rel = (r.get("wav") or "").replace("\\", "/")

        if emo not in KEEP: continue
        if not speaker_is_english(wav_rel): continue

        wav_abs = to_abs(os.path.join(args.audio_root, wav_rel))
        if not os.path.isfile(wav_abs):
            missing_audio += 1
            continue

        tr = read_esd_transcript(args.audio_root, wav_abs) or ""
        recs.append({"audio_path": wav_abs, "transcript": tr, "label": emo})

    if not recs:
        raise RuntimeError("No training samples found.")

    has_t = sum(1 for r in recs if r["transcript"].strip())
    print(f"Loaded train-json samples: {len(recs)} | missing_audio={missing_audio}", flush=True)
    print(f"Transcript availability: {has_t}/{len(recs)} = {has_t/max(1,len(recs)):.3f}", flush=True)

    transcript_pool = [r["transcript"] for r in recs if r["transcript"].strip()]

    # -------- Split Train/Val --------
    rng = random.Random(args.seed)
    by = {e: [] for e in EMOS}
    for r in recs: by[r["label"]].append(r)

    train_l, val_l = [], []
    for e in EMOS:
        lst = by[e]
        rng.shuffle(lst)
        v = min(int(args.val_per_class), len(lst))
        val_l.extend(lst[:v])
        train_l.extend(lst[v:])

    # Balance Train
    tcnt = Counter([r["label"] for r in train_l])
    min_c = min(tcnt.values())
    t_by = {e: [] for e in EMOS}
    for r in train_l: t_by[r["label"]].append(r)
    final_train = []
    for e in EMOS: final_train.extend(t_by[e][:min_c])
    rng.shuffle(final_train)

    print("Train size:", len(final_train), " | Val size:", len(val_l), flush=True)
    
    train_ds = Dataset.from_list(final_train)
    val_ds = Dataset.from_list(val_l)

    # -------- Model & Transform --------
    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    transform = VoxtralChatAudioTextGateTransform(
        proc,
        prompt_text=prompt_text,
        max_new_tokens=8,
        text_drop_prob=args.text_drop_prob,
        text_corrupt_prob=args.text_corrupt_prob,
        transcript_pool=transcript_pool,
        seed=args.seed,
        debug_once=True,
    )
    train_ds.set_transform(transform)
    val_ds.set_transform(transform)

    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = VoxtralForConditionalGeneration.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = VoxtralForConditionalGeneration.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable()
        except: pass

    # -------- PEFT: DoRA Configuration --------
    # DoRA decomposes weights W = m * (V / ||V||)
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_dora=True, # Enables Weight-Decomposed LoRA
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    pad_id = proc.tokenizer.pad_token_id
    if pad_id is None: pad_id = proc.tokenizer.eos_token_id

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=VoxtralPaddingCollator(pad_token_id=pad_id),
        callbacks=[SimpleLogCallback(), EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("\n=== START TRAIN (DoRA + Gating) ===", flush=True)
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_adapter")
    safe_makedirs(final_dir)
    trainer.model.save_pretrained(final_dir)
    proc.save_pretrained(final_dir)
    print("Saved adapter:", final_dir, flush=True)

if __name__ == "__main__":
    main()


# Voxtral-Mini-3B DoRA Fine-Tuning with Stochastic Transcript Gating

## Abstract
This repository implements a specialized fine-tuning pipeline for the **Voxtral-mini-3b** multimodal model on the **ESD (Emotional Speech Dataset)**. It leverages **Weight-Decomposed Low-Rank Adaptation (DoRA)** for parameter efficiency and introduces a novel **Stochastic Transcript Gating** mechanism to train the model to robustly handle missing or unreliable textual context.

## Key Innovations

### 1. DoRA (Weight-Decomposed LoRA)

Unlike standard LoRA, which adapts weights solely through low-rank matrices ($W' = W + \Delta W$), DoRA decomposes the pre-trained weight matrix into two components:
* **Magnitude ($m$):** A scaling vector.
* **Direction ($V$):** A normalized directional matrix.

This decomposition ($W = m \frac{V}{||V||}$) allows DoRA to learn magnitude and direction updates separately, often resulting in higher learning capacity and stability closer to full fine-tuning.

### 2. Stochastic Transcript Gating (Implicit MoE)

We aim to train a model that acts as a "soft gate": relying on text when accurate, but falling back to audio prosody when text is missing or wrong. Instead of building an explicit Mixture-of-Experts (MoE) architecture, we induce this behavior via **Data Augmentation** during training:

* **Modality Dropout ($P_{drop}=0.5$):** * *Mechanism:* Randomly strips the transcript from the input.
    * *Effect:* Forces the model to maintain high performance using **Audio-Only** features.
* **Modality Corruption ($P_{corrupt}=0.15$):** * *Mechanism:* Replaces the correct transcript with a random, mismatched transcript from the batch.
    * *Effect:* Forces the model to detect dissonance between audio and text, learning to **ignore** the text modality when it contradicts the acoustic signal.

## Methodology

### Data Handling
* **Source:** ESD Train Folds (JSONL).
* **Transcript Parsing:** Automatically extracts transcripts from ESD speaker text files (`downloads/esd/<spk>/<spk>.txt`).
* **Splitting:** Dynamically carves a stratified Validation set (e.g., 100 samples/class) from the training fold, ensuring no leakage from the Test fold.

### Training Configuration
| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Adapter Type** | DoRA | `use_dora=True` |
| **Rank ($r$)** | 16 | Low-rank dimension |
| **Alpha ($\alpha$)** | 32 | Scaling factor |
| **Targets** | q, k, v, o | All linear attention projections |
| **Text Drop** | 0.5 | 50% chance to see Audio-Only |
| **Text Corrupt** | 0.15 | 15% chance to see Misleading Text |
