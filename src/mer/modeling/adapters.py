"""PEFT adapter discovery helpers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHT_FILES: tuple[str, ...] = ("adapter_model.safetensors", "adapter_model.bin")


def safe_adapter_name(value: str | os.PathLike[str]) -> str:
    """Create a filesystem-safe adapter name."""
    text = str(value)
    text = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", text)
    return text.strip("_")[:120] if text.strip("_") else "adapter"


def adapter_tag_from_path(path: str | os.PathLike[str]) -> str:
    """Return a stable display tag for an adapter path."""
    adapter_path = Path(path)
    if adapter_path.name == "final_adapter":
        return safe_adapter_name(adapter_path.parent.name)
    return safe_adapter_name(adapter_path.name)


def has_adapter_config(path: str | os.PathLike[str]) -> bool:
    """Return whether a directory contains an adapter config."""
    return (Path(path) / ADAPTER_CONFIG).is_file()


def has_adapter_weights(path: str | os.PathLike[str]) -> bool:
    """Return whether a directory contains common PEFT adapter weight files."""
    adapter_path = Path(path)
    return any((adapter_path / filename).is_file() for filename in ADAPTER_WEIGHT_FILES)


def resolve_adapter_dir(path: str | os.PathLike[str]) -> str | None:
    """Resolve a path to a PEFT adapter directory.

    Accepts either the adapter directory itself or a parent containing
    ``final_adapter``.
    """
    adapter_path = Path(path).expanduser().resolve()
    if has_adapter_config(adapter_path):
        return str(adapter_path)

    final_adapter = adapter_path / "final_adapter"
    if has_adapter_config(final_adapter):
        return str(final_adapter)

    return None


def discover_adapters(adapters_root: str | os.PathLike[str]) -> list[str]:
    """Discover adapter directories under one root directory."""
    root = Path(adapters_root).expanduser().resolve()
    if not root.is_dir():
        return []

    adapters: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        resolved = resolve_adapter_dir(child)
        if resolved is not None:
            adapters.append(resolved)

    return _dedupe(adapters)


def find_adapter_candidates(adapters_root: str | os.PathLike[str]) -> list[str]:
    """Find adapter-like directories that have config or weights.

    This mirrors the broader IEMOCAP script behavior, where a directory with
    adapter weights but no config was still considered adapter-like.
    """
    root = Path(adapters_root).expanduser().resolve()
    if not root.is_dir():
        return []

    candidates: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / "final_adapter" if (child / "final_adapter").is_dir() else child
        if has_adapter_config(candidate) or has_adapter_weights(candidate):
            candidates.append(str(candidate.resolve()))

    return _dedupe(candidates)


def load_adapter_config(adapter_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load an adapter config JSON file."""
    config_path = Path(adapter_dir) / ADAPTER_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_dora_adapter(adapter_dir: str | os.PathLike[str]) -> bool:
    """Return whether an adapter config has ``use_dora=true``."""
    try:
        config = load_adapter_config(adapter_dir)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(config.get("use_dora", False))


def _dedupe(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        output.append(path)
    return output
