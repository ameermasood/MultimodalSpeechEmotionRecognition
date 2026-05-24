# Source Code

This folder contains the reusable Python source code.

## Folder Structure

- **`mer/`**  
  Main project package used by scripts, tests, notebooks, and future demo work.

## `src/mer`

The package is organized by responsibility:

- `mer.data`: dataset paths, metadata, transcripts, and labels
- `mer.inference`: prompt construction and Voxtral prediction helpers
- `mer.modeling`: Voxtral loading, quantization, and PEFT adapter helpers
- `mer.training`: collators, transforms, splits, and PEFT training utilities
- `mer.evaluation`: metrics and statistical evaluation helpers
- `mer.visualization`: reusable plotting utilities
- `mer.config`: runtime configuration helpers

Project-specific code should live in `src/mer`.

## CLI

The package exposes the project command:

```bash
mer list
```
