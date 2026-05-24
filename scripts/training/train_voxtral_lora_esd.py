#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: Voxtral-mini-3b PEFT Fine-Tuning (LoRA/DoRA) on ESD Fold Splits
=======================================================================

Abstract:
    This script implements a Parameter-Efficient Fine-Tuning (PEFT) pipeline for the
    Voxtral-mini-3b multimodal model. It is designed for Speech Emotion Recognition (SER)
    tasks using the Emotional Speech Dataset (ESD).

    Methodology:
    1.  **Data Ingestion**: Loads a specific fold from the ESD dataset metadata.
    2.  **Stratified Partitioning**: Splits the fold into training and validation sets,
        ensuring a fixed number of samples per class (Stratified Sampling) to rigorously
        evaluate generalization.
    3.  **Prompt Engineering**: Implements a novel "User-Only" chat template strategy
        to bypass role-validation constraints in the base model, enabling direct supervised
        learning on the generated tokens.
    4.  **Optimization**: Utilizes QLoRA (4-bit quantization) or standard LoRA for
        memory-efficient adaptation of linear projection layers (q, k, v, o).


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
from mer.training.esd import load_esd_training_records, split_balanced_train_val
from mer.training.peft import create_lora_config
from mer.training.transforms import VoxtralChatAudioTransform
from mer.training.utils import safe_makedirs, set_seed

class SimpleLogCallback(TrainerCallback):
    """
    Custom callback for logging training metrics (Loss, Eval Loss) to stdout
    in a concise format for real-time monitoring.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            print(f"step={state.global_step} | loss={logs['loss']:.4f}", flush=True)
        if "eval_loss" in logs:
            print(f"epoch={state.epoch:.2f} | step={state.global_step} | eval_loss={logs['eval_loss']:.4f}", flush=True)

# -------------------------
# Main Execution Block
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Voxtral-mini-3b Fine-Tuning Pipeline")
    # Path Arguments
    ap.add_argument("--meta_dir", required=True, help="Directory containing ESD metadata JSONLs")
    ap.add_argument("--audio_root", required=True, help="Root directory for audio files")
    ap.add_argument("--model_id", required=True, help="HuggingFace model ID or path")
    ap.add_argument("--output_dir", required=True, help="Directory to save checkpoints")
    
    # Experiment Arguments
    ap.add_argument("--fold", type=int, default=2, help="ESD Fold index to use for training")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--val_per_class", type=int, default=100, help="Samples per class reserved for validation")

    # Training Hyperparameters
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=1, help="Early stopping patience")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    ap.add_argument("--train_bs", type=int, default=1, help="Batch size per device")
    ap.add_argument("--eval_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)

    # PEFT / LoRA Arguments
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA Rank")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA Scaling Factor")
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--use_dora", action="store_true", help="Enable DoRA (Weight-Decomposed LoRA)")

    # Quantization Arguments
    ap.add_argument("--load_in_4bit", action="store_true", help="Enable QLoRA (4-bit quantization)")

    args = ap.parse_args()
    set_seed(args.seed)
    safe_makedirs(args.output_dir)

    prompt_text = (
        "You are an expert at recognizing emotions from speech.\n"
        "Listen to the audio and output only ONE label from:\n"
        "Angry, Happy, Sad, Neutral."
    )

    recs, load_stats = load_esd_training_records(
        args.meta_dir,
        args.audio_root,
        args.fold,
        labels=CANONICAL_EMOTIONS,
        include_transcripts=False,
    )

    if not recs:
        raise RuntimeError("No training samples found. Check meta_dir/audio_root and speaker filter.")
    print(
        f"Loaded train-json samples: {len(recs)} | missing_audio={load_stats['missing_audio']}",
        flush=True,
    )

    final_train, val_l = split_balanced_train_val(
        recs,
        val_per_class=args.val_per_class,
        seed=args.seed,
        labels=CANONICAL_EMOTIONS,
    )

    print("Train size:", len(final_train), " | Val size:", len(val_l), flush=True)
    print("Train class counts:", Counter([r["label"] for r in final_train]), flush=True)
    print("Val class counts:", Counter([r["label"] for r in val_l]), flush=True)

    train_ds = Dataset.from_list(final_train)
    val_ds = Dataset.from_list(val_l)

    # ---------------------------------------------------------
    # Model Initialization & Configuration
    # ---------------------------------------------------------
    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    transform = VoxtralChatAudioTransform(
        proc,
        prompt_text=prompt_text,
        max_new_tokens=8,
        debug_once=True,
    )
    train_ds.set_transform(transform)
    val_ds.set_transform(transform)

    model = load_voxtral_for_training(args.model_id, load_in_4bit=args.load_in_4bit)

    # ---------------------------------------------------------
    # PEFT Adapter Injection (LoRA/DoRA)
    # ---------------------------------------------------------
    peft_cfg = create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        use_dora=args.use_dora,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    pad_id = tokenizer_pad_id(proc.tokenizer)

    # ---------------------------------------------------------
    # Training Loop Definition
    # ---------------------------------------------------------
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

    callbacks = [SimpleLogCallback(), EarlyStoppingCallback(early_stopping_patience=args.patience)]

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=VoxtralPaddingCollator(pad_token_id=pad_id),
        callbacks=callbacks,
    )

    print("\n=== START TRAIN ===", flush=True)
    trainer.train()

    # Save final adapter weights
    final_dir = os.path.join(args.output_dir, "final_adapter")
    safe_makedirs(final_dir)
    trainer.model.save_pretrained(final_dir)
    proc.save_pretrained(final_dir)
    print("Saved adapter:", final_dir, flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
