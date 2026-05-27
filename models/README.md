# Local Base Models

This folder is a local placeholder for downloaded base models.

The Voxtral base model is large, so the actual model files are not uploaded to
this repository. Keep downloaded weights, tokenizer files, processor files, and
other model artifacts local.

## Recommended Usage

For most runs, use the Hugging Face model ID directly:

```bash
BASE_MODEL_ID=mistralai/Voxtral-Mini-3B-2507
```

The code can load the model from Hugging Face and cache it in your local Hugging
Face cache directory. This avoids storing large model files inside the project.

## Optional Local Download

If Hugging Face loading is unavailable or you want a fixed local copy, download
the base model into this folder, for example:

```text
models/
  voxtral-mini-3b-2507/
    config.json
    generation_config.json
    model-*.safetensors
    processor_config.json
    tokenizer.json
    tokenizer_config.json
    ...
```

Then point the project to the local path:

```bash
BASE_MODEL_ID=models/voxtral-mini-3b-2507
```

Do not commit the downloaded model files. This folder is configured so only this
README is tracked by git.

## Demo Notes

The Streamlit demo expects:

- A Voxtral base model, either from Hugging Face or a local path here
- A fine-tuned PEFT adapter under `checkpoints/`, for example
  `checkpoints/final_adapter_dora`

The base model and adapter are separate: the base model contains the original
Voxtral weights, while the adapter contains the smaller fine-tuned PEFT weights
from this project.
