from importlib import metadata
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mer.config import RuntimeConfig
from mer.inference.demo import (
    DemoEmotionPredictor,
    DemoPrediction,
    _require_bitsandbytes,
    _select_runtime_device,
    _softmax_scores,
)


class DemoPredictionTests(unittest.TestCase):
    def test_to_dict_is_json_friendly(self):
        prediction = DemoPrediction(
            label="Happy",
            confidence=None,
            audio_path="/tmp/sample.wav",
            transcript_used=True,
            adapter="final_adapter_dora",
            adapter_path="checkpoints/final_adapter_dora",
            base_model="mistralai/Voxtral-Mini-3B-2507",
            raw_text="Happy",
            label_scores={"Angry": 0.1, "Happy": 0.9},
        )

        self.assertEqual(
            prediction.to_dict(),
            {
                "label": "Happy",
                "confidence": None,
                "audio_path": "/tmp/sample.wav",
                "transcript_used": True,
                "adapter": "final_adapter_dora",
                "adapter_path": "checkpoints/final_adapter_dora",
                "base_model": "mistralai/Voxtral-Mini-3B-2507",
                "raw_text": "Happy",
                "label_scores": {"Angry": 0.1, "Happy": 0.9},
            },
        )


class DemoEmotionPredictorTests(unittest.TestCase):
    def test_predict_returns_demo_prediction_without_loading_real_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "final_adapter_dora"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            audio = root / "sample.wav"
            audio.write_bytes(b"fake wav")

            config = RuntimeConfig(
                base_model_id="mistralai/Voxtral-Mini-3B-2507",
                adapter_path=str(adapter),
                load_in_4bit=False,
                device="cpu",
                do_sample=False,
            )
            predictor = DemoEmotionPredictor(config)

            with patch.object(predictor, "_load", return_value=("processor", "model", "cpu")):
                with patch(
                    "mer.inference.demo._predict_label_with_confidence",
                    return_value=("Happy", "Happy", 0.91, {"Angry": 0.09, "Happy": 0.91}),
                ) as predict:
                    result = predictor.predict(audio, transcript="hello")

            self.assertEqual(result.label, "Happy")
            self.assertEqual(result.confidence, 0.91)
            self.assertEqual(result.label_scores, {"Angry": 0.09, "Happy": 0.91})
            self.assertTrue(result.transcript_used)
            self.assertEqual(result.adapter, "final_adapter_dora")
            self.assertEqual(result.base_model, "mistralai/Voxtral-Mini-3B-2507")
            predict.assert_called_once()
            self.assertTrue(Path(result.audio_path).is_absolute())

    def test_predict_requires_existing_audio_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            predictor = DemoEmotionPredictor(
                RuntimeConfig(base_model_id="model", adapter_path=str(adapter))
            )

            with self.assertRaisesRegex(FileNotFoundError, "Audio file"):
                predictor.predict(Path(tmp) / "missing.wav")

    def test_init_requires_adapter_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Adapter path"):
                DemoEmotionPredictor(RuntimeConfig(base_model_id="model", adapter_path=tmp))

    def test_require_bitsandbytes_has_clear_error_when_missing(self):
        missing = metadata.PackageNotFoundError("bitsandbytes")
        with patch("mer.inference.demo.metadata.version", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "4-bit loading requires bitsandbytes"):
                _require_bitsandbytes()

    def test_select_runtime_device_keeps_explicit_device(self):
        self.assertEqual(_select_runtime_device("cpu"), "cpu")
        self.assertEqual(_select_runtime_device("mps"), "mps")

    def test_softmax_scores_normalizes_candidate_scores(self):
        scores = _softmax_scores({"Angry": 2.0, "Happy": 0.0})
        expected = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)

        self.assertAlmostEqual(scores["Angry"], float(expected[0]))
        self.assertAlmostEqual(scores["Happy"], float(expected[1]))
        self.assertAlmostEqual(sum(scores.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
