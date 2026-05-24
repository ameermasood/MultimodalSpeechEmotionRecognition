# Scripts

This folder contains command-line workflow scripts for training and evaluation.

Most users should prefer the package CLI:

```bash
mer list
```

The scripts remain available as direct Python entry points for debugging,
experimentation, and transparent reproducibility.

## Layout

```text
scripts/
+-- training/
+-- evaluation/
```

## Public Workflows

Training:

```bash
mer train-lora-esd --help
```

Evaluation:

```bash
mer zero-shot-esd --help
mer zero-shot-iemocap --help
mer evaluate-esd --help
mer evaluate-iemocap --help
mer evaluate-esd-dora --help
mer evaluate-iemocap-dora --help
```

All generated predictions, figures, metrics, logs, and checkpoints should be
written to ignored local artifact folders such as `results/`, `logs/`, or
`checkpoints/`.
