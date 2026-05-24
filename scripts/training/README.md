# Training Scripts

This folder contains the public fine-tuning entry point.

## Contents

- **`train_voxtral_lora_esd.py`**  
  Fine-tunes Voxtral-Mini-3B on ESD using PEFT adapters.

## Main Characteristics

- ESD training data
- audio-only input
- frozen base model
- trainable low-rank adapter weights
- LoRA by default
- DoRA available through script arguments

## Usage

Preferred CLI command:

```bash
mer train-lora-esd --help
```

Direct script command:

```bash
python scripts/training/train_voxtral_lora_esd.py --help
```