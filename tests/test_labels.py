import unittest

from mer.data import CANONICAL_EMOTIONS, normalize_emotion_name, normalize_prediction_text


class LabelNormalizationTests(unittest.TestCase):
    def test_dataset_label_synonyms_normalize_to_canonical_names(self):
        self.assertEqual(normalize_emotion_name("ang"), "Ang")
        self.assertEqual(normalize_emotion_name("anger"), "Angry")
        self.assertEqual(normalize_emotion_name("happiness"), "Happy")
        self.assertEqual(normalize_emotion_name(" calm "), "Neutral")

    def test_prediction_normalization_uses_exact_and_synonym_matches(self):
        self.assertEqual(normalize_prediction_text("Happy"), "Happy")
        self.assertEqual(normalize_prediction_text("The answer is sadness."), "Sad")
        self.assertEqual(normalize_prediction_text("I think the speaker is angry"), "Angry")

    def test_prediction_normalization_uses_fallback(self):
        self.assertEqual(normalize_prediction_text("unclear emotion"), "Neutral")
        self.assertEqual(normalize_prediction_text("", default="Unknown"), "Unknown")

    def test_canonical_emotions_order_is_stable(self):
        self.assertEqual(CANONICAL_EMOTIONS, ("Angry", "Happy", "Sad", "Neutral"))


if __name__ == "__main__":
    unittest.main()
