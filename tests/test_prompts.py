import unittest

from mer.inference import build_emotion_instruction, build_system_user_conversation, build_user_only_conversation


class PromptBuilderTests(unittest.TestCase):
    def test_instruction_mentions_transcript_only_when_requested(self):
        audio_only = build_emotion_instruction(use_text=False)
        audio_text = build_emotion_instruction(use_text=True)

        self.assertIn("From the given audio", audio_only)
        self.assertNotIn("and its transcript", audio_only)
        self.assertIn("and its transcript", audio_text)

    def test_user_only_conversation_adds_transcript_when_present(self):
        conversation = build_user_only_conversation(
            audio_path="sample.wav",
            transcript="hello there",
            use_text=True,
        )

        self.assertEqual(conversation[0]["role"], "user")
        self.assertEqual(conversation[0]["content"][0]["type"], "audio")
        self.assertIn("Transcript:\nhello there", conversation[0]["content"][-1]["text"])

    def test_system_user_conversation_adds_system_prompt(self):
        conversation = build_system_user_conversation(audio_path="sample.wav")

        self.assertEqual([message["role"] for message in conversation], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
