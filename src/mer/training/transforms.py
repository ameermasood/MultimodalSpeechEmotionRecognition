"""Dataset transforms for supervised Voxtral emotion fine-tuning."""

from __future__ import annotations

import random
from typing import Any

import torch


class VoxtralChatAudioTransform:
    """Encode audio-only training examples and mask loss to target label tokens."""

    def __init__(self, processor, prompt_text: str, max_new_tokens: int = 8, debug_once: bool = True):
        self.proc = processor
        self.tok = processor.tokenizer
        self.prompt_text = prompt_text
        self.max_new_tokens = int(max_new_tokens)
        self.debug_once = bool(debug_once)
        self._printed = False

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _user_content(self, wav_path: str) -> list[dict[str, str]]:
        return [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": self.prompt_text},
        ]

    def _encode_one(self, wav_path: str, label_text: str) -> dict[str, torch.Tensor]:
        messages = [{"role": "user", "content": self._user_content(wav_path)}]
        encoded = self.proc.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        prefix_ids = encoded["input_ids"][0]
        prefix_attn = encoded["attention_mask"][0]
        prefix_len = int(prefix_ids.numel())

        label_ids = self.tok.encode(" " + label_text, add_special_tokens=False)
        if self.tok.eos_token_id is not None:
            label_ids = label_ids + [self.tok.eos_token_id]
        label_ids = label_ids[: max(1, self.max_new_tokens)]
        label_tensor = torch.tensor(label_ids, dtype=torch.long)

        input_ids = torch.cat([prefix_ids, label_tensor], dim=0)
        attention_mask = torch.cat([prefix_attn, torch.ones_like(label_tensor)], dim=0)
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        output = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        for key, value in encoded.items():
            if key not in output and torch.is_tensor(value):
                output[key] = value[0]

        if self.debug_once and not self._printed:
            self._printed = True
            print("DEBUG out keys:", sorted(output.keys()), flush=True)
            if "input_features" in output:
                print("DEBUG input_features:", tuple(output["input_features"].shape), flush=True)
            print("DEBUG prefix_len:", prefix_len, "| total_len:", int(input_ids.numel()), flush=True)
            print("DEBUG label_text:", label_text, flush=True)

        return output

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        is_batched = isinstance(example["audio_path"], list)

        def as_list(value):
            return value if isinstance(value, list) else [value]

        outputs = [
            self._encode_one(wav_path, label)
            for wav_path, label in zip(as_list(example["audio_path"]), as_list(example["label"]))
        ]
        if not outputs:
            return {}

        packed = {key: [output[key] for output in outputs] for key in outputs[0]}
        return packed if is_batched else {key: value[0] for key, value in packed.items()}


class VoxtralChatAudioTextGateTransform(VoxtralChatAudioTransform):
    """Encode audio-plus-text examples with stochastic transcript dropout/corruption."""

    def __init__(
        self,
        processor,
        prompt_text: str,
        max_new_tokens: int = 8,
        text_drop_prob: float = 0.5,
        text_corrupt_prob: float = 0.15,
        transcript_pool: list[str] | None = None,
        seed: int = 42,
        debug_once: bool = True,
    ):
        super().__init__(processor, prompt_text, max_new_tokens=max_new_tokens, debug_once=debug_once)
        self.text_drop_prob = float(text_drop_prob)
        self.text_corrupt_prob = float(text_corrupt_prob)
        self.pool = [text for text in (transcript_pool or []) if text and text.strip()]
        self.rng = random.Random(seed + 12345)

    def _maybe_use_transcript(self, transcript: str) -> tuple[bool, str, str]:
        text = (transcript or "").strip()
        if not text:
            return False, "", "no_transcript"
        if self.rng.random() < self.text_drop_prob:
            return False, "", "dropped"
        if self.pool and self.rng.random() < self.text_corrupt_prob:
            corrupt_text = self.pool[self.rng.randrange(0, len(self.pool))]
            if corrupt_text and corrupt_text.strip():
                return True, corrupt_text.strip(), "corrupted"
        return True, text, "kept"

    def _user_content(self, wav_path: str, transcript: str = "") -> tuple[list[dict[str, str]], bool, str]:
        used, text, reason = self._maybe_use_transcript(transcript)
        content = [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": self.prompt_text},
        ]
        if used and text:
            content.append({"type": "text", "text": f"Transcript (may be missing or incorrect):\n{text}"})
        return content, used, reason

    def _encode_one(self, wav_path: str, transcript: str, label_text: str) -> dict[str, torch.Tensor]:
        user_content, used_transcript, reason = self._user_content(wav_path, transcript)
        messages = [{"role": "user", "content": user_content}]
        encoded = self.proc.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        prefix_ids = encoded["input_ids"][0]
        prefix_attn = encoded["attention_mask"][0]
        prefix_len = int(prefix_ids.numel())

        label_ids = self.tok.encode(" " + label_text, add_special_tokens=False)
        if self.tok.eos_token_id is not None:
            label_ids = label_ids + [self.tok.eos_token_id]
        label_ids = label_ids[: max(1, self.max_new_tokens)]
        label_tensor = torch.tensor(label_ids, dtype=torch.long)

        input_ids = torch.cat([prefix_ids, label_tensor], dim=0)
        attention_mask = torch.cat([prefix_attn, torch.ones_like(label_tensor)], dim=0)
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        output = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        for key, value in encoded.items():
            if key not in output and torch.is_tensor(value):
                output[key] = value[0]

        if self.debug_once and not self._printed:
            self._printed = True
            print("DEBUG (first sample) transcript_gate:", {"used": used_transcript, "reason": reason}, flush=True)
            print("DEBUG out keys:", sorted(output.keys()), flush=True)
            print("DEBUG prefix_len:", prefix_len, "| total_len:", int(input_ids.numel()), flush=True)

        return output

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        is_batched = isinstance(example["audio_path"], list)

        def as_list(value):
            return value if isinstance(value, list) else [value]

        audio_paths = as_list(example["audio_path"])
        transcripts = as_list(example.get("transcript", [""] * len(audio_paths)))
        labels = as_list(example["label"])

        outputs = [
            self._encode_one(wav_path, transcript, label)
            for wav_path, transcript, label in zip(audio_paths, transcripts, labels)
        ]
        if not outputs:
            return {}

        packed = {key: [output[key] for output in outputs] for key in outputs[0]}
        return packed if is_batched else {key: value[0] for key, value in packed.items()}
