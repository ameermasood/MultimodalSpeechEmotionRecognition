# ESD

This folder is reserved for a local copy of the Emotional Speech Dataset.

ESD is used for:

- Zero-shot evaluation
- PEFT fine-tuning on acted emotional speech
- In-domain adapter evaluation

The public project uses the four-class paper-aligned setup:

```text
Angry, Happy, Sad, Neutral
```

ESD `Surprise` is excluded from the main experiments.

Raw audio is not included in git. Download ESD from its official source and keep
the original directory structure locally.
