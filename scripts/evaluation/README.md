# Evaluation Scripts

This folder contains scripts for zero-shot and adapter-based evaluation on ESD
and IEMOCAP.

Evaluations compare:

- Audio-only inference
- Audio-plus-transcript inference, when transcripts are available

## Label Policy

The public evaluation setup uses four labels:

```text
Angry, Happy, Sad, Neutral
```

Dataset-specific rules:

- ESD `Surprise` is excluded.
- IEMOCAP `excited` is mapped to `Happy`.

## Zero-Shot Evaluation

```bash
mer zero-shot-esd --help
mer zero-shot-iemocap --help
```

Direct scripts:

```text
scripts/evaluation/evaluate_zero_shot_esd.py
scripts/evaluation/evaluate_zero_shot_iemocap.py
```

These scripts evaluate the base Voxtral model without task-specific adapter
weights.

## Adapter Evaluation

```bash
mer evaluate-esd --help
mer evaluate-iemocap --help
```

Direct scripts:

```text
scripts/evaluation/evaluate_esd_adapters.py
scripts/evaluation/evaluate_iemocap_adapters.py
```

These scripts evaluate fine-tuned PEFT adapters and compare audio-only versus
audio-plus-transcript inference modes.

## DoRA Adapter Evaluation

```bash
mer evaluate-esd-dora --help
mer evaluate-iemocap-dora --help
```

Direct scripts:

```text
scripts/evaluation/evaluate_esd_dora_transcript.py
scripts/evaluation/evaluate_iemocap_dora_transcript.py
```

These scripts filter for DoRA adapters and run transcript-aware evaluation. They
do not imply that stochastic transcript gating is part of the public training
workflow.

## Outputs

Evaluation outputs should be written to ignored local folders such as:

```text
results/zero_shot_esd/
results/zero_shot_iemocap/
results/esd_adapters/
results/iemocap_adapters/
```

Curated summary figures or tables can be added to `results/` later if they are
useful for the portfolio.
