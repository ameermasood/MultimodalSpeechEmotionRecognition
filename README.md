# Multimodal Speech Emotion Recognition via Parameter-Efficient Fine-Tuning of Audio-Language Models

This repository contains our multimodal speech emotion recognition project using
Voxtral-Mini-3B. The system treats emotion recognition as constrained text
generation, compares zero-shot and fine-tuned settings, and evaluates whether
audio-only or audio-plus-transcript prompting is more reliable across datasets.

## Method Overview

The project follows a staged speech emotion recognition pipeline:

1. Prepare local ESD and IEMOCAP paths and metadata.
2. Run zero-shot Voxtral evaluation on both datasets.
3. Fine-tune PEFT adapters on ESD.
4. Evaluate adapters on the ESD test split.
5. Evaluate ESD-trained adapters on IEMOCAP for cross-domain transfer.
6. Compare audio-only and audio-plus-transcript inference.
7. Report metrics, confusion patterns, and results.

The model outputs one label from the four-class emotion set:

```text
Angry, Happy, Sad, Neutral
```

## Repository Structure

```text
.
+-- data/            # dataset placeholders and local layout notes
+-- notebooks/       # exploration notebook
+-- results/         # result summaries and figures
+-- scripts/         # task-based training and evaluation entry points
+-- src/mer/         # reusable Python package
+-- tests/           # lightweight tests for package helpers
+-- adsp_mer_t11p7_paper.pdf
+-- pyproject.toml   # package metadata and mer CLI entry point
+-- requirements.txt # Python dependencies
+-- README.md
```

Large local artifacts such as raw audio, checkpoints, logs, generated outputs,
and vendor code are intentionally kept out of git.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install the Local Package Entry Point

```bash
python3 -m pip install -e .
```

This enables:

```bash
mer --help
mer list
```

If editable install is blocked by your Python setup, use the package entry point
directly from the repository root:

```bash
PYTHONPATH=src python3 -m mer.cli --help
PYTHONPATH=src python3 -m mer.cli list
```

### 3. Prepare Local Artifacts

Download or place the required artifacts locally:

- ESD audio
- IEMOCAP audio
- metadata JSONL files
- Voxtral base model or Hugging Face model ID
- PEFT adapter folders for adapter evaluation

## Quick Start

List available workflows:

```bash
mer list
```

Inspect the main commands:

```bash
mer zero-shot-esd --help
mer zero-shot-iemocap --help
mer train-lora-esd --help
mer evaluate-esd --help
mer evaluate-iemocap --help
```

## Script-Based Workflow

### Run Zero-Shot Baselines

```bash
mer zero-shot-esd --help
mer zero-shot-iemocap --help
```

### Fine-Tune on ESD

```bash
mer train-lora-esd --help
```

### Evaluate Fine-Tuned Adapters

```bash
mer evaluate-esd --help
mer evaluate-iemocap --help
```

### Evaluate DoRA Adapters

```bash
mer evaluate-esd-dora --help
mer evaluate-iemocap-dora --help
```

The intended chained flow is:

1. Run zero-shot baselines.
2. Train or provide PEFT adapters.
3. Evaluate adapters on ESD.
4. Evaluate transfer to IEMOCAP.

## Outputs

Generated artifacts are organized under:

- `checkpoints/`
- `logs/`
- `results/`

Only curated summaries and figures should be committed under `results/`; raw
logs, checkpoints, large prediction files, and model weights should stay local.

## Notebook Workflow

The public notebook is:

```text
notebooks/dataset_exploration.ipynb
```

It is used for dataset exploration, label checks, duration analysis, and visual
inspection. Training and evaluation are handled by scripts and the `mer` CLI.

## Results

The paper reports that zero-shot Voxtral has limited balanced performance on
speech emotion recognition, while PEFT adaptation improves results substantially.

### ESD Dataset - In-domain


| Setting | Reported macro-F1 | Delta |
|---|---:|---:|
| Zero-shot | 0.13 | n/a |
| Fine-tuned | 0.84 | **+0.71** |

### IEMOCAP Dataset - Cross-domain

| Setting | Reported macro-F1 | Delta |
|---|---:|---:|
| Zero-shot | 0.37 | n/a |
| Fine-tuned | 0.63 | **+0.26** |

For the full written analysis, see:

```text
adsp_mer_t11p7_paper.pdf
```

## Authors

This project was developed as part of the course Applied Data Science Project (ADSP) at Politecnico di Torino by:

| Author | Contact |
|---|---|
| Amir Masoud Almasi | amirmasoud.almasi@studenti.polito.it |
| Ashkan Shafiei | ashkan.shafiei@studenti.polito.it |
| Parastoo Alavi | parastoo.alavi@studenti.polito.it |

## Notes

This repository is for academic purposes. Dataset licenses,
external frameworks, and pretrained model licenses remain with their original
owners.

## Acknowledgments

This project was carried out in collaboration with LINKS Foundation and Politecnico di Torino. We sincerely thank
Prof. Giuseppe Rizzo, Federico D'Asaro, and Juan José Márquez Villacís for their
supervision throughout the project.
