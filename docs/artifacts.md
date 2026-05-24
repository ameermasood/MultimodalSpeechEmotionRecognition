# External Artifacts

Large runtime artifacts should stay outside git. This includes base models,
fine-tuned adapters, checkpoints, logs, datasets, generated predictions, and
figures.

Use this document to reconstruct the expected layout locally or in Google Drive.

## Recommended Layout

```text
artifacts/
├── models/
│   ├── base/
│   │   └── voxtral-mini-3b/
│   └── adapters/
│       ├── lora_esd/
│       │   └── final_adapter/
│       └── dora_esd/
│           └── final_adapter/
├── checkpoints/
│   ├── lora_esd/
│   └── dora_esd/
├── logs/
│   ├── training/
│   └── evaluation/
└── results/
    ├── predictions/
    ├── metrics/
    └── figures/
```

The exact root can be local or remote, for example:

```text
/path/to/artifacts
/content/drive/MyDrive/adsp
```

## Required Model Files

For inference and the future demo, the most important artifact is a final PEFT
adapter directory.

A usable adapter directory should contain files such as:

```text
final_adapter/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer_config.json
├── tokenizer.json
├── preprocessor_config.json
└── special_tokens_map.json
```

Some tokenizer or processor files may be loaded from the base model instead,
depending on how the adapter was saved.

## Demo Requirements

The demo should not train models. It should load:

- A base Voxtral model from a Hugging Face model ID or local path
- One selected fine-tuned adapter, preferably the best validated DoRA or LoRA run
- One user-provided audio file
- An optional transcript

Suggested environment variables for the demo:

```text
BASE_MODEL_ID=mistralai/...
ADAPTER_PATH=/path/to/artifacts/models/adapters/dora_esd/final_adapter
LOAD_IN_4BIT=true
DEVICE=auto
```

Do not commit `.env` files with local paths.

## Git Policy

The following artifact folders should remain ignored:

- `artifacts/`
- `models/`
- `checkpoints/`
- `logs/`
- `results/`
- raw audio and dataset folders

Only lightweight documentation, configs, tests, and source code should be
tracked.

## Selection Notes

Before building the demo, identify the adapter to use and record:

- Dataset and fold
- Training method: LoRA or DoRA
- Base model ID/path
- Validation/test metrics
- Whether transcript input was used during training
- Known limitations
