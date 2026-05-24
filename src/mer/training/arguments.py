"""Hugging Face training argument builders."""

from __future__ import annotations


def create_training_arguments(
    output_dir: str,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: int,
    weight_decay: float,
    dataloader_num_workers: int,
    load_in_4bit: bool = False,
):
    """Create the shared TrainingArguments used by Voxtral training scripts."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_num_workers,
        report_to="none",
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
    )
