# Notebooks

This folder contains lightweight notebooks for exploration and interpretation.

The only public notebook is:

```text
dataset_exploration.ipynb
```

It is intended for:

- Dataset exploration
- Label distribution checks
- Speaker/session summaries
- Audio duration analysis
- Visual analysis that supports the project narrative

Training, zero-shot evaluation, and adapter evaluation should stay in scripts or
the `mer` CLI, not in notebooks.

Useful commands:

```bash
mer list
mer zero-shot-esd --help
mer evaluate-esd --help
```
