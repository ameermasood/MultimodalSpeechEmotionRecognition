import unittest

from mer.config import RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_from_env_reads_required_values_and_defaults(self):
        config = RuntimeConfig.from_env(
            {
                "BASE_MODEL_ID": "mistralai/voxtral-mini-3b",
                "ADAPTER_PATH": "checkpoints/final_adapter_dora",
            }
        )

        self.assertEqual(config.base_model_id, "mistralai/voxtral-mini-3b")
        self.assertEqual(config.adapter_path, "checkpoints/final_adapter_dora")
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.max_new_tokens, 8)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.top_p, 0.95)

    def test_from_env_reads_overrides(self):
        config = RuntimeConfig.from_env(
            {
                "BASE_MODEL_ID": "/models/voxtral",
                "ADAPTER_PATH": "/adapters/dora",
                "LOAD_IN_4BIT": "false",
                "DEVICE": "cuda:0",
                "MAX_NEW_TOKENS": "3",
                "DO_SAMPLE": "yes",
                "TEMPERATURE": "0.4",
                "TOP_P": "0.8",
            }
        )

        self.assertFalse(config.load_in_4bit)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.max_new_tokens, 3)
        self.assertTrue(config.do_sample)
        self.assertEqual(config.temperature, 0.4)
        self.assertEqual(config.top_p, 0.8)

    def test_from_env_requires_model_and_adapter(self):
        with self.assertRaisesRegex(ValueError, "BASE_MODEL_ID"):
            RuntimeConfig.from_env({"ADAPTER_PATH": "adapter"})

        with self.assertRaisesRegex(ValueError, "ADAPTER_PATH"):
            RuntimeConfig.from_env({"BASE_MODEL_ID": "model"})

    def test_from_env_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "LOAD_IN_4BIT"):
            RuntimeConfig.from_env(
                {
                    "BASE_MODEL_ID": "model",
                    "ADAPTER_PATH": "adapter",
                    "LOAD_IN_4BIT": "maybe",
                }
            )

        with self.assertRaisesRegex(ValueError, "MAX_NEW_TOKENS"):
            RuntimeConfig.from_env(
                {
                    "BASE_MODEL_ID": "model",
                    "ADAPTER_PATH": "adapter",
                    "MAX_NEW_TOKENS": "many",
                }
            )


if __name__ == "__main__":
    unittest.main()
