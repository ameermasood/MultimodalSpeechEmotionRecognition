# Notebooks Folder

This folder contains Jupyter notebooks used for data exploration, baseline experiments, and result analysis in the Applied Data Science Project.

The notebooks are mainly used for **interactive analysis and visualization**. All heavy training and large-scale evaluations are implemented in standalone
Python scripts located in 'scripts' folder of the repository.

---

## Notebook Descriptions

### `01_data_exploration.ipynb`
This notebook performs an initial exploration of the ESD and IEMOCAP datasets.
It includes:
- Emotion label distribution analysis
- Speaker and session statistics
- Audio duration analysis
- Basic waveform visualizations

The goal of this notebook is to understand dataset characteristics, identify
potential biases, and guide the design of the experiments.

---

### `02_zero_shot_iemocap.ipynb`
This notebook evaluates the pre-trained Voxtral-Mini-3B model in a **zero-shot**
setting on the IEMOCAP dataset.

Experiments include:
- Audio-only inference
- Audio + transcript inference
- Evaluation using Accuracy and Macro-F1

This notebook establishes a baseline performance before any fine-tuning.

---

### `03_zero_shot_esd.ipynb`
This notebook applies the same zero-shot evaluation strategy to the ESD dataset.

It highlights how general audio-language models perform on emotion recognition
without task-specific training and allows comparison with IEMOCAP results.
