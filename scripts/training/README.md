# Training Scripts

This folder contains scripts used to fine-tune the Voxtral-Mini-3B model on
emotion recognition tasks using Parameter-Efficient Fine-Tuning (PEFT).

All training is performed on the **ESD dataset**, which provides a balanced
distribution of emotion classes.

---

## Scripts

### `train_voxtral_lora_esd.py`
Fine-tunes Voxtral-Mini-3B using **LoRA** adapters.

Main characteristics:
- Uses audio-only input
- Freezes the base model
- Trains low-rank adapters on selected layers

---

Additional local experimental training scripts may exist in a working copy, but
the public training entry point is the LoRA/DoRA-capable ESD script above.
