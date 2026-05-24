"""Voxtral model loading helpers."""

from __future__ import annotations


def load_voxtral_for_training(model_id: str, load_in_4bit: bool = False):
    """Load Voxtral for PEFT training, optionally with 4-bit quantization."""
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, VoxtralForConditionalGeneration

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    return model


def tokenizer_pad_id(tokenizer) -> int:
    """Return a usable tokenizer pad ID or raise a clear error."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")
    return pad_id
