# Training Scripts

This folder contains scripts used to fine-tune the Voxtral-Mini-3B model on
emotion recognition tasks using Parameter-Efficient Fine-Tuning (PEFT).

All training is performed on the **ESD dataset**, which provides a balanced
distribution of emotion classes.

---

## Scripts

### `03_train_lora_voxtral_esd.py`
Fine-tunes Voxtral-Mini-3B using **LoRA** adapters.

Main characteristics:
- Uses audio-only input
- Freezes the base model
- Trains low-rank adapters on selected layers

---

### `03_train_dora_voxtral_with_transcript.py`
Fine-tunes Voxtral-Mini-3B using **DoRA** adapters with optional transcript input.

Main characteristics:
- Uses audio + text when transcripts are available
- Applies transcript dropout to improve robustness
- Trains only DoRA adapters while freezing the base model
