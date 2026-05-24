# Source Code

This folder contains the Python source code for the project.

```text
src/
+-- mer/      # Project package
+-- EmoBox/   # Optional local vendor/external code, ignored by git
```

## `src/mer`

`src/mer` is the reusable package for this project. It contains the code used by
the command-line scripts and future demo work:

- `mer.data`: dataset paths, metadata, transcripts, and labels
- `mer.inference`: prompt construction and Voxtral prediction helpers
- `mer.modeling`: Voxtral loading, quantization, and PEFT adapter helpers
- `mer.training`: collators, transforms, splits, and PEFT training utilities
- `mer.evaluation`: metrics and statistical evaluation helpers
- `mer.visualization`: reusable plotting utilities
- `mer.config`: runtime configuration helpers

## `src/EmoBox`

`src/EmoBox` is treated as external/vendor code and is ignored by git in the
cleaned portfolio version of the repository.

If you need EmoBox metadata or preprocessing tools, keep a local copy of EmoBox
outside version control and point scripts to the generated metadata files.

Project-specific logic should live in `src/mer`, not in `src/EmoBox`.
