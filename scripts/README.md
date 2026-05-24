# Scripts

This folder contains task-based entry points for the modular workflow.

Most users should prefer the package CLI:

```bash
mer list
```

The direct scripts remain useful for debugging, transparency, and reproducing
individual experiment stages.

## Folder Structure

- **`training/`**  
  Fine-tuning entry points.

- **`evaluation/`**  
  Zero-shot and adapter evaluation entry points.

## Main Workflows

```bash
mer zero-shot-esd --help
mer zero-shot-iemocap --help
mer train-lora-esd --help
mer evaluate-esd --help
mer evaluate-iemocap --help
```

Generated predictions, figures, metrics, logs, and checkpoints should be written
to ignored local folders such as `results/`, `logs/`, and `checkpoints/`.
