import unittest

import torch

from mer.training.transforms import VoxtralChatAudioTextGateTransform


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = None

    def encode(self, text, add_special_tokens=False):
        return [10 + len(text.strip())]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.messages = []

    def apply_chat_template(self, messages, tokenize=True, return_dict=True, return_tensors="pt"):
        self.messages.append(messages)
        return {
            "input_ids": torch.tensor([[1, 3]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "input_features": torch.ones((1, 2, 2)),
        }


class VoxtralChatAudioTextGateTransformTests(unittest.TestCase):
    def test_drops_transcript_when_drop_probability_is_one(self):
        processor = FakeProcessor()
        transform = VoxtralChatAudioTextGateTransform(
            processor,
            prompt_text="classify",
            text_drop_prob=1.0,
            transcript_pool=["wrong transcript"],
            debug_once=False,
        )

        output = transform({"audio_path": "a.wav", "transcript": "right transcript", "label": "Happy"})

        content = processor.messages[0][0]["content"]
        self.assertEqual([item["type"] for item in content], ["audio", "text"])
        self.assertEqual(output["labels"].tolist()[:2], [-100, -100])

    def test_keeps_transcript_when_drop_and_corrupt_probabilities_are_zero(self):
        processor = FakeProcessor()
        transform = VoxtralChatAudioTextGateTransform(
            processor,
            prompt_text="classify",
            text_drop_prob=0.0,
            text_corrupt_prob=0.0,
            debug_once=False,
        )

        transform({"audio_path": "a.wav", "transcript": "right transcript", "label": "Happy"})

        content = processor.messages[0][0]["content"]
        self.assertIn("right transcript", content[-1]["text"])


if __name__ == "__main__":
    unittest.main()
