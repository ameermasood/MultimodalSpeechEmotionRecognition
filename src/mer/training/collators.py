"""Batch collation helpers for Voxtral fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def _pad_right_to_shape(
    tensor: torch.Tensor,
    target_shape: list[int],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Right-pad a tensor to the requested shape."""
    if list(tensor.shape) == target_shape:
        return tensor

    pad = []
    for current, target in zip(reversed(tensor.shape), reversed(target_shape)):
        pad.extend([0, int(target) - int(current)])
    return F.pad(tensor, pad, mode="constant", value=pad_value)


@dataclass
class VoxtralPaddingCollator:
    """Dynamically pad text tokens, labels, and multimodal tensors for Voxtral batches."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {}

        for key in ["input_ids", "attention_mask", "labels"]:
            tensors = [feature[key] for feature in features]
            max_len = max(int(tensor.size(0)) for tensor in tensors)
            if key == "labels":
                pad_value = -100
            elif key == "attention_mask":
                pad_value = 0
            else:
                pad_value = self.pad_token_id

            padded = []
            for tensor in tensors:
                diff = max_len - int(tensor.size(0))
                if diff > 0:
                    pad = torch.full((diff,), pad_value, dtype=tensor.dtype)
                    padded.append(torch.cat([tensor, pad], dim=0))
                else:
                    padded.append(tensor)
            batch[key] = torch.stack(padded)

        extra_keys = set(features[0].keys()) - {"input_ids", "attention_mask", "labels"}
        for key in sorted(extra_keys):
            first_value = features[0][key]
            if not torch.is_tensor(first_value):
                continue

            tensors = [feature[key] for feature in features]
            ndim = tensors[0].dim()
            if any(tensor.dim() != ndim for tensor in tensors):
                continue

            max_shape = [max(int(tensor.shape[dim]) for tensor in tensors) for dim in range(ndim)]
            pad_value = 0.0 if tensors[0].dtype.is_floating_point else 0
            batch[key] = torch.stack(
                [_pad_right_to_shape(tensor, max_shape, pad_value) for tensor in tensors]
            )

        return batch
