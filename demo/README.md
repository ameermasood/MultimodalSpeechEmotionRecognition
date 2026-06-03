# Gradio Demo

This folder contains the local Gradio demo for speech emotion recognition.

The demo is intentionally thin: UI code lives here, while model loading and
prediction logic stays in the reusable `mer` package.

## What It Shows

The app is titled:

```text
Multimodal Speech Emotion Recognition
```

It supports:

- Uploading or recording a speech sample
- Optional transcript input for audio-plus-text inference
- Prediction over the four labels: `Angry`, `Happy`, `Sad`, `Neutral`
- A dynamic prediction panel with label confidence
- A label distribution section inside the prediction panel

In this project, multimodal means multimodal input: speech audio can be used
alone, or combined with optional transcript text. The output is a classification
result, not a multimodal generated output.

## Required Artifacts

The local demo expects:

```text
checkpoints/final_adapter_dora/
```

with PEFT adapter files such as:

```text
adapter_config.json
adapter_model.safetensors
```

The base model is loaded from Hugging Face by default:

```text
mistralai/Voxtral-Mini-3B-2507
```

You can override runtime paths with environment variables:

```bash
BASE_MODEL_ID=mistralai/Voxtral-Mini-3B-2507
ADAPTER_PATH=checkpoints/final_adapter_dora
DEVICE=auto
MAX_NEW_TOKENS=8
LOAD_IN_4BIT=false
```

## Run Locally

Install the project in editable mode:

```bash
python3 -m pip install -e .
```

Voxtral's processor also requires `mistral-common`, which is included in the
project requirements.

Start the app:

```bash
python3 demo/gradio_app.py
```

For UI work, Gradio's CLI is more convenient because it can reload after code
changes:

```bash
PYTHONPATH=src gradio demo/gradio_app.py
```

Then upload or record an audio file and optionally paste a transcript.

The displayed label confidence is computed by scoring each allowed emotion label
(`Angry`, `Happy`, `Sad`, `Neutral`) as a candidate continuation and normalizing
the scores across those four labels. It is more useful than raw generation
probability, but it is still not a calibrated clinical or scientific confidence
estimate.

## Notes

Voxtral-Mini-3B is large. A practical local run usually needs a compatible GPU,
enough memory, and the correct PyTorch/Transformers/PEFT stack. The first run may
download the base model into the Hugging Face cache.

Runtime settings are read from environment variables when needed, including
`BASE_MODEL_ID`, `ADAPTER_PATH`, `DEVICE`, `MAX_NEW_TOKENS`, and
`LOAD_IN_4BIT`.
