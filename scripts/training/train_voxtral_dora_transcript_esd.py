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

import argparse
from collections import Counter

from datasets import Dataset

from transformers import (
    AutoProcessor,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)

from peft import (
    get_peft_model,
)

from mer.data.labels import CANONICAL_EMOTIONS
from mer.modeling.voxtral import load_voxtral_for_training, tokenizer_pad_id
from mer.training.arguments import create_training_arguments
from mer.training.collators import VoxtralPaddingCollator
from mer.training.esd import load_esd_training_records, split_balanced_train_val, transcript_pool_from_records
from mer.training.peft import create_lora_config
from mer.training.transforms import VoxtralChatAudioTextGateTransform
from mer.training.utils import safe_makedirs, set_seed

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

    recs, load_stats = load_esd_training_records(
        args.meta_dir,
        args.audio_root,
        args.fold,
        labels=CANONICAL_EMOTIONS,
        include_transcripts=True,
    )

    if not recs:
        raise RuntimeError("No training samples found.")

    has_t = sum(1 for r in recs if r["transcript"].strip())
    print(f"Loaded train-json samples: {len(recs)} | missing_audio={load_stats['missing_audio']}", flush=True)
    print(f"Transcript availability: {has_t}/{len(recs)} = {has_t/max(1,len(recs)):.3f}", flush=True)

    transcript_pool = transcript_pool_from_records(recs)

    final_train, val_l = split_balanced_train_val(
        recs,
        val_per_class=args.val_per_class,
        seed=args.seed,
        labels=CANONICAL_EMOTIONS,
    )

    print("Train size:", len(final_train), " | Val size:", len(val_l), flush=True)
    
    train_ds = Dataset.from_list(final_train)
    val_ds = Dataset.from_list(val_l)

    # -------- Model & Transform --------
    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    train_transform = VoxtralChatAudioTextGateTransform(
        proc,
        prompt_text=prompt_text,
        max_new_tokens=8,
        text_drop_prob=args.text_drop_prob,
        text_corrupt_prob=args.text_corrupt_prob,
        transcript_pool=transcript_pool,
        seed=args.seed,
        debug_once=True,
    )
    val_transform = VoxtralChatAudioTextGateTransform(
        proc,
        prompt_text=prompt_text,
        max_new_tokens=8,
        text_drop_prob=0.0,
        text_corrupt_prob=0.0,
        transcript_pool=transcript_pool,
        seed=args.seed,
        debug_once=False,
    )
    train_ds.set_transform(train_transform)
    val_ds.set_transform(val_transform)

    model = load_voxtral_for_training(args.model_id, load_in_4bit=args.load_in_4bit)

    # -------- PEFT: DoRA Configuration --------
    # DoRA decomposes weights W = m * (V / ||V||)
    peft_cfg = create_lora_config(r=16, alpha=32, dropout=0.1, use_dora=True)
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    pad_id = tokenizer_pad_id(proc.tokenizer)

    train_args = create_training_arguments(
        output_dir=args.output_dir,
        train_batch_size=args.train_bs,
        eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        dataloader_num_workers=args.num_workers,
        load_in_4bit=args.load_in_4bit,
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
