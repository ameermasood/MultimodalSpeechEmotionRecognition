# Demo

This folder contains the local Gradio demo for speech emotion recognition.

The demo is intentionally thin: UI code lives here, while model loading and
prediction logic stays in the reusable `mer` package.

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

Then upload an audio file and optionally paste a transcript.

The displayed label confidence is computed by scoring each allowed emotion label
(`Angry`, `Happy`, `Sad`, `Neutral`) as a candidate continuation and normalizing
the scores across those four labels. It is more useful than raw generation
probability, but it is still not a calibrated clinical or scientific confidence
estimate.

## Notes

Voxtral-Mini-3B is large. A practical local run usually needs a compatible GPU,
enough memory, and the correct PyTorch/Transformers/PEFT stack. The first run may
download the base model into the Hugging Face cache.

4-bit loading requires `bitsandbytes` and compatible GPU support. If
`bitsandbytes` is not installed, keep **Load in 4-bit** disabled in the demo
advanced settings.
