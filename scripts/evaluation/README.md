# Evaluation Scripts

This folder contains scripts used to evaluate both zero-shot and fine-tuned
models on the ESD and IEMOCAP datasets.

Evaluations are performed in two modes:
- Audio-only
- Audio + transcript

---

## Scripts

### `04_eval_all_adapters_esd.py`
Evaluates all available fine-tuned adapters on the **ESD test set**.

The script:
- Compares multiple adapters
- Reports Accuracy, Macro-F1, and Balanced Accuracy
- Includes confidence calibration and latency statistics

---

### `04_eval_all_adapters_iemocap.py`
Evaluates all available adapters on the **IEMOCAP dataset**.

This script is mainly used to test **cross-dataset generalization**, since models
are trained on ESD but evaluated on IEMOCAP.

---

### `04_zero_shot_dora_with_transcript_esd.py`
Runs **zero-shot evaluation** on ESD using Voxtral-Mini-3B with DoRA-style prompting.

This script serves as a baseline for comparison with fine-tuned results.

---

### `04_zero_shot_dora_with_transcript_iemocap.py`
Runs zero-shot evaluation on IEMOCAP using the same setup as the ESD zero-shot
script, allowing direct comparison between datasets.
