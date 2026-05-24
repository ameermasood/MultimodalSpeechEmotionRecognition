"""Runtime configuration helpers for inference and demos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options shared by local inference and future demos."""

    base_model_id: str
    adapter_path: str
    load_in_4bit: bool = True
    device: str = "auto"
    max_new_tokens: int = 8
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.95

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeConfig":
        """Create config from environment variables."""
        values = env if env is not None else os.environ
        return cls(
            base_model_id=_get_required(values, "BASE_MODEL_ID"),
            adapter_path=_get_required(values, "ADAPTER_PATH"),
            load_in_4bit=_parse_bool(values.get("LOAD_IN_4BIT", "true"), "LOAD_IN_4BIT"),
            device=values.get("DEVICE", "auto"),
            max_new_tokens=_parse_int(values.get("MAX_NEW_TOKENS", "8"), "MAX_NEW_TOKENS"),
            do_sample=_parse_bool(values.get("DO_SAMPLE", "false"), "DO_SAMPLE"),
            temperature=_parse_float(values.get("TEMPERATURE", "0.2"), "TEMPERATURE"),
            top_p=_parse_float(values.get("TOP_P", "0.95"), "TOP_P"),
        )


def _get_required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _parse_bool(value: str, key: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {key}: {value!r}")


def _parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {key}: {value!r}") from exc


def _parse_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {key}: {value!r}") from exc
