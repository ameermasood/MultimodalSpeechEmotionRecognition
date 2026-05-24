# Evaluation Scripts

This folder contains scripts used to evaluate both zero-shot and fine-tuned
models on the ESD and IEMOCAP datasets.

Evaluations are performed in two modes:
- Audio-only
- Audio + transcript

---

## Scripts

### `evaluate_esd_adapters.py`
Evaluates all available fine-tuned adapters on the **ESD test set**.

The script:
- Compares multiple adapters
- Reports Accuracy, Macro-F1, and Balanced Accuracy
- Includes confidence calibration and latency statistics

---

### `evaluate_iemocap_adapters.py`
Evaluates all available adapters on the **IEMOCAP dataset**.

This script is mainly used to test **cross-dataset generalization**, since models
are trained on ESD but evaluated on IEMOCAP.

---

### `evaluate_esd_dora_transcript.py`
Runs **zero-shot evaluation** on ESD using Voxtral-Mini-3B with DoRA-style prompting.

This script serves as a baseline for comparison with fine-tuned results.

---

### `evaluate_iemocap_dora_transcript.py`
Runs zero-shot evaluation on IEMOCAP using the same setup as the ESD zero-shot
script, allowing direct comparison between datasets.
