# Multimodal Speech Emotion Recognition with Voxtral

This project explores speech emotion recognition with Voxtral-Mini-3B, a large
audio-language model. It compares zero-shot inference with parameter-efficient
fine-tuning, studies audio-only versus audio-plus-transcript inputs, and tests
whether models trained on acted speech generalize to conversational speech.

The project was originally developed for the Applied Data Science Project course
at Politecnico di Torino and has been organized into a reproducible Python
package with runnable training and evaluation scripts.

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

This makes the project useful as both a machine learning experiment and a
portfolio project: it includes model adaptation, multimodal inference,
cross-dataset evaluation, and a path toward deployment.

## Datasets

The experiments use two speech emotion datasets:

- **ESD**, the Emotional Speech Dataset, used for fine-tuning and in-domain
  evaluation.
- **IEMOCAP**, used for cross-domain evaluation on conversational speech.

The main label setup follows the paper-aligned four-class task:

- ESD `Surprise` is excluded.
- IEMOCAP `excited` is mapped to `Happy`.

Raw audio and large dataset artifacts are not included in this repository.
They must be downloaded separately according to each dataset's license.

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

`src/mer` contains the reusable code for data handling, prompts, inference,
model loading, training utilities, metrics, and visualizations. Scripts are kept
as command-line entry points around that package.

`src/EmoBox` is treated as external/vendor code and is not tracked as part of
the cleaned project source.

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

See the README files under `scripts/` for script-specific usage.

## Setup

Install the project in editable mode:

```bash
pip install -e .
```

Then provide local paths to datasets, metadata, model artifacts, and output
directories when running scripts.

Large model and audio dependencies depend on your machine and GPU environment.
Training and full evaluation are expected to require a suitable CUDA setup.

## Required External Artifacts

This repository does not include large files such as datasets, model weights,
or generated checkpoints. A full run expects:

- Raw ESD audio
- Raw IEMOCAP audio
- Metadata JSONL files for dataset splits
- Voxtral base model, usually from Hugging Face
- Fine-tuned PEFT adapter folders for adapter evaluation
- Output folders for metrics, predictions, plots, logs, and checkpoints

See `docs/artifacts.md` for a more detailed artifact checklist.

## Future Demo

After the training and evaluation workflow is stable, the project can be turned
into a small inference demo. The demo should accept an audio file, optionally
accept a transcript, load the selected Voxtral adapter, and return the predicted
emotion with basic metadata.

This demo is intentionally planned as a later deployment step, after the core
research pipeline is documented and reproducible.

## Limitations

- Emotion labels are subjective and sometimes ambiguous.
- ESD is acted speech, while IEMOCAP is conversational speech.
- Transcripts may help in some cases but conflict with vocal tone in others.
- Cross-domain generalization remains difficult.
- Large audio-language models require significant compute.

## Authors

This project was developed by:

- Amir Masoud Almasi
- Parastoo Alavi
- Ashkan Shafiei

Course context: Applied Data Science Project, Politecnico di Torino.

## Notes

This repository is for academic and portfolio purposes. Dataset licenses,
external frameworks, and pretrained model licenses remain with their original
owners. Raw datasets, generated checkpoints, and large model artifacts should
not be committed to git.
