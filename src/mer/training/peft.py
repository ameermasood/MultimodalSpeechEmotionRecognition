"""PEFT configuration helpers for Voxtral training."""

from __future__ import annotations

VOXTRAL_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def create_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    use_dora: bool = False,
    target_modules: tuple[str, ...] = VOXTRAL_LORA_TARGET_MODULES,
):
    """Create the LoRA/DoRA config used by the Voxtral training scripts."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        use_dora=use_dora,
    )
