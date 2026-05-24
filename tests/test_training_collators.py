import unittest

import torch

from mer.training.collators import VoxtralPaddingCollator


class VoxtralPaddingCollatorTests(unittest.TestCase):
    def test_pads_token_fields_with_expected_values(self):
        collator = VoxtralPaddingCollator(pad_token_id=99)

        batch = collator(
            [
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                    "labels": torch.tensor([-100, 4, 5]),
                },
                {
                    "input_ids": torch.tensor([6]),
                    "attention_mask": torch.tensor([1]),
                    "labels": torch.tensor([7]),
                },
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3], [6, 99, 99]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 0, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 4, 5], [7, -100, -100]])

    def test_pads_extra_tensor_fields_to_max_shape(self):
        collator = VoxtralPaddingCollator(pad_token_id=0)

        batch = collator(
            [
                {
                    "input_ids": torch.tensor([1]),
                    "attention_mask": torch.tensor([1]),
                    "labels": torch.tensor([2]),
                    "input_features": torch.ones((2, 3)),
                },
                {
                    "input_ids": torch.tensor([3]),
                    "attention_mask": torch.tensor([1]),
                    "labels": torch.tensor([4]),
                    "input_features": torch.ones((1, 2)) * 5,
                },
            ]
        )

        self.assertEqual(tuple(batch["input_features"].shape), (2, 2, 3))
        self.assertTrue(torch.equal(batch["input_features"][1], torch.tensor([[5.0, 5.0, 0.0], [0.0, 0.0, 0.0]])))


if __name__ == "__main__":
    unittest.main()
