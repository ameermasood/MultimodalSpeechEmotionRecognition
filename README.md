# Multimodal Speech Emotion Recognition with Voxtral

This project explores speech emotion recognition with Voxtral-Mini-3B, a large
audio-language model. It compares zero-shot inference with parameter-efficient
fine-tuning, studies audio-only versus audio-plus-transcript inputs, and tests
whether models trained on acted speech generalize to conversational speech.

## What It Does

Given a speech recording, the system predicts one of four emotion labels:

```text
Angry, Happy, Sad, Neutral
```

The model can use:

- Audio only
- Audio plus transcript text, when available

Speech emotion recognition is treated as constrained text generation: Voxtral is
prompted to listen to the utterance and output exactly one emotion label.

## Why This Project Matters

Modern audio-language models can understand speech, but they are not necessarily
strong emotion recognizers out of the box. This project investigates:

- How well Voxtral performs without task-specific training
- How much LoRA and DoRA adapters improve performance
- Whether transcripts help or interfere with emotion recognition
- How well an ESD-trained model transfers to IEMOCAP

## Datasets

The experiments use two speech emotion datasets:

- **ESD**, the Emotional Speech Dataset, used for fine-tuning and in-domain
  evaluation.
- **IEMOCAP**, used for cross-domain evaluation on conversational speech.

## Method Overview

```text
Audio file
   |
   +-- optional transcript
   |
   v
Voxtral-Mini-3B
   |
   +-- zero-shot prompting
   |
   +-- LoRA / DoRA PEFT adapters
   |
   v
Emotion label: Angry, Happy, Sad, or Neutral
```

The main experiment flow is:

```text
Zero-shot Voxtral
        |
        v
Fine-tune PEFT adapters on ESD
        |
        v
Evaluate on ESD
        |
        v
Evaluate transfer to IEMOCAP
        |
        v
Analyze audio-only vs audio-plus-transcript behavior
```

## Key Experiments

| Experiment | Purpose |
| --- | --- |
| Zero-shot ESD | Measure Voxtral without fine-tuning on acted speech |
| Zero-shot IEMOCAP | Measure Voxtral without fine-tuning on conversational speech |
| LoRA fine-tuning on ESD | Adapt Voxtral efficiently with low-rank adapters |
| DoRA fine-tuning on ESD | Compare weight-decomposed adapters against LoRA |
| ESD adapter evaluation | Test in-domain fine-tuned performance |
| IEMOCAP adapter evaluation | Test cross-domain generalization |
| Audio vs audio + transcript | Measure whether text helps or hurts predictions |

## Results Summary

The paper reports that zero-shot Voxtral has limited balanced performance for
this task, while PEFT adaptation improves results substantially.

| Setting | Reported macro-F1 |
| --- | ---: |
| Zero-shot ESD | 0.13 |
| Zero-shot IEMOCAP | 0.37 |
| Best ESD fine-tuned result | 0.84 |
| Best IEMOCAP cross-domain result | 0.63 |

Macro-F1 is emphasized because it treats each emotion class equally, which is
important when class difficulty and label distributions differ.

## Repository Layout

```text
.
+-- src/mer/              # Reusable project package
+-- scripts/training/     # Fine-tuning entry points
+-- scripts/evaluation/   # Zero-shot and adapter evaluation scripts
+-- notebooks/            # Dataset exploration and analysis
+-- tests/                # Lightweight tests for reusable helpers
+-- docs/                 # Artifact notes and project documentation
+-- results/              # Generated metrics, plots, and reports
+-- pyproject.toml
+-- README.md
```

## Main Scripts

After installing the project, common workflows are available through the `mer`
command:

```bash
mer list
mer zero-shot-esd --help
mer train-lora-esd --help
mer evaluate-iemocap --help
```

Zero-shot evaluation:

```text
scripts/evaluation/evaluate_zero_shot_esd.py
scripts/evaluation/evaluate_zero_shot_iemocap.py
```

Training:

```text
scripts/training/train_voxtral_lora_esd.py
```

Adapter evaluation:

```text
scripts/evaluation/evaluate_esd_adapters.py
scripts/evaluation/evaluate_esd_dora_transcript.py
scripts/evaluation/evaluate_iemocap_adapters.py
scripts/evaluation/evaluate_iemocap_dora_transcript.py
```

## Setup

Install the project in editable mode:

```bash
pip install -e .
```

Then provide local paths to datasets, metadata, model artifacts, and output
directories when running scripts.


## Authors

This project was developed by:

- Amir Masoud Almasi
- Ashkan Shafiei
- Parastoo Alavi



## Notes

This repository is for academic and portfolio purposes. Dataset licenses,
external frameworks, and pretrained model licenses remain with their original owners.