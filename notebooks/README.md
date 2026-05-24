# Notebooks

This folder contains Jupyter notebooks used for interactive exploration and portfolio narrative.

Notebooks should stay lightweight. Training, zero-shot evaluation, and adapter evaluation are implemented as command-line scripts in `scripts/`.

---

## Available Notebook

### `dataset_exploration.ipynb`
This notebook performs an initial exploration of the ESD and IEMOCAP datasets.
It includes:

- Emotion label distribution analysis
- Speaker and session statistics
- Audio duration analysis
- Basic waveform visualizations

The goal of this notebook is to understand dataset characteristics, identify
potential biases, and guide the design of the experiments.

---

## Script-Based Workflows

Zero-shot experiments now live in:

- `scripts/evaluation/evaluate_zero_shot_esd.py`
- `scripts/evaluation/evaluate_zero_shot_iemocap.py`

This keeps notebooks focused on exploration while making expensive evaluations reproducible from the command line.
