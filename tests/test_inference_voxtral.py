import unittest

from mer.inference.voxtral import build_zero_shot_conversation


class VoxtralInferenceTests(unittest.TestCase):
    def test_build_zero_shot_conversation_includes_audio_and_optional_transcript(self):
        conversation = build_zero_shot_conversation(
            audio_path="sample.wav",
            transcript="hello world",
            use_text=True,
            labels=["Angry", "Happy"],
        )

        self.assertEqual(conversation[0]["role"], "system")
        self.assertEqual(conversation[1]["role"], "user")
        content = conversation[1]["content"]
        self.assertEqual(content[0]["type"], "audio_url")
        self.assertIn("Angry, Happy", content[1]["text"])
        self.assertIn("hello world", content[2]["text"])


if __name__ == "__main__":
    unittest.main()
