# Data

This folder documents the expected local dataset layout.

Raw datasets are not committed to git. They must be downloaded separately and
placed locally according to each dataset license.

Tracked files in this folder are placeholders only.

## Expected Layout

```text
data/
+-- raw/
|   +-- ESD/
|   +-- IEMOCAP/
```

## Project Label Policy

The public experiment setup uses four labels:

```text
Angry, Happy, Sad, Neutral
```

Dataset-specific rules:

- ESD `Surprise` is excluded from the main paper-aligned setup.
- IEMOCAP `excited` is mapped to `Happy`.

## Notes

- Do not commit raw audio files.
- Do not commit generated dataset exports unless they are intentionally curated.
- Scripts receive dataset and metadata paths through command-line arguments.
