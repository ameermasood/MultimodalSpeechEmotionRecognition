# Training Scripts

This folder contains the public training entry point for ESD fine-tuning.

## `train_voxtral_lora_esd.py`

Fine-tunes Voxtral-Mini-3B on ESD using PEFT adapters.

Main characteristics:

- Uses ESD as the training dataset
- Uses audio-only input
- Freezes the base model
- Trains low-rank adapter weights
- Supports LoRA by default
- Can enable DoRA through the script arguments

Preferred CLI usage:

```bash
mer train-lora-esd --help
```

Direct script usage:

```bash
python scripts/training/train_voxtral_lora_esd.py --help
```

## Outputs

Training outputs should be written to an ignored local checkpoint folder, for
example:

```text
checkpoints/<run_name>/
```

Do not commit model weights, optimizer states, logs, or generated checkpoints.

## Local Experiments

Additional local training scripts may exist in a working copy. They are not part
of the public paper-aligned workflow unless they are explicitly restored and
documented.
