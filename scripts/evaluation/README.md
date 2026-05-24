# Evaluation Scripts

This folder contains zero-shot and adapter evaluation scripts for ESD and
IEMOCAP.

Evaluations compare:

- audio-only inference
- audio-plus-transcript inference

## Zero-Shot Evaluation

CLI commands:

```bash
mer zero-shot-esd --help
mer zero-shot-iemocap --help
```

Direct scripts:

- `evaluate_zero_shot_esd.py`
- `evaluate_zero_shot_iemocap.py`

These scripts evaluate the base Voxtral model without task-specific adapter
weights.

## Adapter Evaluation

CLI commands:

```bash
mer evaluate-esd --help
mer evaluate-iemocap --help
```

Direct scripts:

- `evaluate_esd_adapters.py`
- `evaluate_iemocap_adapters.py`

These scripts evaluate PEFT adapters and compare inference modes.

## DoRA Adapter Evaluation

CLI commands:

```bash
mer evaluate-esd-dora --help
mer evaluate-iemocap-dora --help
```

Direct scripts:

- `evaluate_esd_dora_transcript.py`
- `evaluate_iemocap_dora_transcript.py`

These scripts filter for DoRA adapters and run transcript-aware evaluation.

## Outputs

Suggested local output folders:

```text
results/zero_shot_esd/
results/zero_shot_iemocap/
results/esd_adapters/
results/iemocap_adapters/
```