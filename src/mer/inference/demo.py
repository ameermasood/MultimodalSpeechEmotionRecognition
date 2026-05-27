"""Demo-oriented single-file inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from mer.config import RuntimeConfig
from mer.data.labels import CANONICAL_EMOTIONS
from mer.inference.voxtral import build_zero_shot_conversation
from mer.modeling import adapter_tag_from_path, resolve_adapter_dir


@dataclass(frozen=True)
class DemoPrediction:
    """Serializable prediction result returned by demo frontends."""

    label: str
    confidence: float | None
    audio_path: str
    transcript_used: bool
    adapter: str
    adapter_path: str
    base_model: str
    raw_text: str | None = None
    label_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "audio_path": self.audio_path,
            "transcript_used": self.transcript_used,
            "adapter": self.adapter,
            "adapter_path": self.adapter_path,
            "base_model": self.base_model,
            "raw_text": self.raw_text,
            "label_scores": self.label_scores,
        }


class DemoEmotionPredictor:
    """Lazy-loading prediction service for local demos and APIs."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        adapter_dir = resolve_adapter_dir(config.adapter_path)
        if adapter_dir is None:
            raise ValueError(
                "Adapter path must contain adapter_config.json or a final_adapter/ directory: "
                f"{config.adapter_path}"
            )
        self.adapter_path = adapter_dir
        self.adapter_name = adapter_tag_from_path(adapter_dir)
        self._processor = None
        self._model = None
        self._device = None

    @classmethod
    def from_env(cls) -> "DemoEmotionPredictor":
        """Create a predictor from environment variables."""
        return cls(RuntimeConfig.from_env())

    def predict(self, audio_path: str | Path, transcript: str = "") -> DemoPrediction:
        """Predict one emotion label for an uploaded/provided audio file."""
        audio = Path(audio_path).expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio}")

        processor, model, device = self._load()
        transcript_text = transcript.strip()
        label, raw_text, confidence, label_scores = _predict_label_with_confidence(
            model=model,
            processor=processor,
            audio_path=str(audio),
            transcript=transcript_text,
            use_text=bool(transcript_text),
            labels=CANONICAL_EMOTIONS,
            device=device,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        return DemoPrediction(
            label=label,
            confidence=confidence,
            audio_path=str(audio),
            transcript_used=bool(transcript_text),
            adapter=self.adapter_name,
            adapter_path=self.adapter_path,
            base_model=self.config.base_model_id,
            raw_text=raw_text,
            label_scores=label_scores,
        )

    def _load(self):
        """Load model, processor, and adapter once per process."""
        if self._processor is not None and self._model is not None and self._device is not None:
            return self._processor, self._model, self._device

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, VoxtralForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.config.base_model_id, trust_remote_code=True)

        device = _select_runtime_device(self.config.device)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        if self.config.load_in_4bit:
            _require_bitsandbytes()
            load_kwargs["device_map"] = "auto" if self.config.device == "auto" else {"": device}
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16 if device != "cpu" else torch.float32

        base = VoxtralForConditionalGeneration.from_pretrained(self.config.base_model_id, **load_kwargs)
        model = PeftModel.from_pretrained(base, self.adapter_path).eval()
        if not self.config.load_in_4bit:
            model = model.to(device)

        self._model = model
        self._device = _infer_device(model, requested_device=device)
        return self._processor, self._model, self._device


def _select_runtime_device(requested_device: str) -> str:
    """Resolve ``auto`` into a concrete local device."""
    if requested_device != "auto":
        return requested_device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _infer_device(model, requested_device: str):
    """Infer the tensor device for processor outputs."""
    if requested_device != "auto":
        return requested_device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _predict_label_with_confidence(
    model,
    processor,
    audio_path: str,
    transcript: str,
    use_text: bool,
    labels=CANONICAL_EMOTIONS,
    device: str = "cpu",
    max_new_tokens: int = 8,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> tuple[str, str, float | None, dict[str, float] | None]:
    """Predict by scoring each valid label as a candidate continuation."""
    label_scores = _score_candidate_labels(
        model=model,
        processor=processor,
        audio_path=audio_path,
        transcript=transcript,
        use_text=use_text,
        labels=labels,
        device=device,
    )
    if not label_scores:
        return "Unknown", "Unknown", None, None

    label = max(label_scores, key=label_scores.get)
    return label, label, label_scores[label], label_scores


def _score_candidate_labels(
    model,
    processor,
    audio_path: str,
    transcript: str,
    use_text: bool,
    labels=CANONICAL_EMOTIONS,
    device: str = "cpu",
) -> dict[str, float]:
    """Return softmax-normalized scores over the allowed emotion labels."""
    import torch

    conversation = build_zero_shot_conversation(
        audio_path=audio_path,
        transcript=transcript,
        use_text=use_text,
        labels=labels,
    )
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    candidate_scores: dict[str, float] = {}
    with torch.no_grad():
        for label in labels:
            token_ids = _candidate_token_ids(processor, label, inputs["input_ids"].device)
            if token_ids.numel() == 0:
                continue
            full_inputs = _append_candidate_tokens(inputs, token_ids)
            outputs = model(**full_inputs)
            candidate_scores[label] = _candidate_mean_logprob(
                outputs.logits,
                prompt_length=inputs["input_ids"].shape[1],
                token_ids=token_ids,
            )

    return _softmax_scores(candidate_scores)


def _candidate_token_ids(processor, label: str, device):
    """Tokenize one candidate label without adding chat/template special tokens."""
    tokenizer = processor.tokenizer
    encoded = tokenizer(label, add_special_tokens=False, return_tensors="pt")
    return encoded["input_ids"].to(device)


def _append_candidate_tokens(inputs: dict[str, Any], token_ids) -> dict[str, Any]:
    """Append candidate text tokens to the prompt input IDs and attention mask."""
    import torch

    full_inputs = dict(inputs)
    full_inputs["input_ids"] = torch.cat([inputs["input_ids"], token_ids], dim=1)
    if "attention_mask" in inputs:
        extra_mask = torch.ones_like(token_ids)
        full_inputs["attention_mask"] = torch.cat([inputs["attention_mask"], extra_mask], dim=1)
    return full_inputs


def _candidate_mean_logprob(logits, prompt_length: int, token_ids) -> float:
    """Mean log probability of candidate tokens conditioned on the prompt."""
    import torch

    log_probs: list[float] = []
    flat_tokens = token_ids[0]
    for index, token_id in enumerate(flat_tokens):
        prediction_position = prompt_length + index - 1
        token_log_probs = torch.log_softmax(logits[0, prediction_position].float(), dim=-1)
        log_probs.append(float(token_log_probs[int(token_id)]))
    if not log_probs:
        return float("-inf")
    return float(sum(log_probs) / len(log_probs))


def _softmax_scores(scores: dict[str, float]) -> dict[str, float]:
    """Softmax-normalize candidate log scores."""
    import math

    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exp_scores.values())
    if total == 0:
        return {label: 0.0 for label in scores}
    return {label: value / total for label, value in exp_scores.items()}


def _require_bitsandbytes() -> None:
    """Raise a clear error when 4-bit loading is requested without bitsandbytes."""
    try:
        metadata.version("bitsandbytes")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "4-bit loading requires bitsandbytes, but it is not installed. "
            "Disable 'Load in 4-bit' in the demo sidebar, or install bitsandbytes "
            "on a compatible GPU environment."
        ) from exc
