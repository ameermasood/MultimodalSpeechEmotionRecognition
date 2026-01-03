# 2025-P7-Multimodal-Emotion-Recognition

---

# 🎧 Multimodal Emotion Recognition with Large Audio-Language Models

This repository contains the work developed for the **Applied Data Science Project**
course. The project focuses on **multimodal emotion recognition from speech** using
large audio-language models and parameter-efficient fine-tuning techniques.

The goal is to study how well general audio-language models recognize emotions
in zero-shot settings and how their performance improves after task-specific
adaptation.

---

## 👥 Team

* **Amir Masoud Almasi - s337006 - amirmasoud.almasi@studenti.polito.it**
* **Parastoo Alavi - s340942 - parastoo.alavi@studenti.polito.it**
* **Ashkan Shafiei - s342583 - ashkan.shafiei@studenti.polito.it**

---

## 📌 Project Overview

Emotion recognition aims to classify emotional states (e.g., Angry, Happy, Sad,
Neutral) from speech signals. While recent large audio-language models can process
audio and text jointly, they are not explicitly trained for emotion recognition.

In this project, we:

* Evaluate a pre-trained audio-language model in **zero-shot** settings
* Apply **Parameter-Efficient Fine-Tuning (PEFT)** using LoRA and DoRA
* Compare **audio-only** vs **audio + transcript** inputs
* Test generalization across datasets

---

## 📂 Repository Structure

```text
.
├── data/           # Dataset structure and metadata (no raw audio files)
├── notebooks/      # Data exploration and analysis notebooks
├── scripts/        # Training and evaluation scripts
│   ├── training/
│   └── evaluation/
├── src/            # External source code (EmoBox framework)
├── results/        # Figures, metrics, and outputs
├── slides/         # Checkpoint and final presentations
└── README.md
```

Each folder contains its own `README.md` explaining its contents.

---

## 📊 Datasets

The project uses two emotional speech datasets:

* **ESD (Emotional Speech Dataset)** — used mainly for training
* **IEMOCAP** — used mainly for evaluation and generalization testing

⚠️ **IMPORTANT**
Raw audio files are **not included** in this repository due to dataset licensing
restrictions and large file sizes. Users must download the datasets separately
and place them in the appropriate folders.

---

## 🧠 Methodology

1. **Data Exploration**
   Analyze label distributions, speaker statistics, and audio durations.

2. **Zero-Shot Evaluation**
   Evaluate the pre-trained model without fine-tuning on both datasets.

3. **PEFT Fine-Tuning**
   Adapt the model using LoRA and DoRA while freezing most parameters.

4. **Evaluation & Analysis**
   Compare zero-shot and fine-tuned performance using multiple metrics.

---

## 📈 Evaluation Metrics

We report multiple metrics to ensure fair evaluation:

* Accuracy
* Balanced Accuracy
* Macro-F1
* Cohen’s Kappa
* Matthews Correlation Coefficient (MCC)

---

## ⚠️ Risks & Limitations

* Emotion labels are subjective and noisy
* Training data may contain speaker or recording bias
* Text transcripts may conflict with vocal tone (e.g., sarcasm)
* Experiments are limited by available computational resources

These limitations are explicitly discussed in the presentation and analysis.

---

## 📄 Course Information

This project was developed as part of the **Applied Data Science Project** course in **Politecnico di Torino** and follows the structure required for:

* Checkpoint presentations
* Final report
* Public GitHub repository submission

---

## 📝 License & Notes

* This repository is for **academic purposes only**
* External libraries (e.g., EmoBox) are used under their original licenses
* The authors do not claim ownership of the datasets or other external frameworks
