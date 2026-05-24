import tempfile
import unittest
from pathlib import Path

from mer.data import infer_gender_from_utt, infer_session_from_utt, resolve_iemocap_audio_path


class IemocapDataHelperTests(unittest.TestCase):
    def test_infer_gender_from_utterance_id(self):
        self.assertEqual(infer_gender_from_utt("Ses01F_impro01_F000"), "female")
        self.assertEqual(infer_gender_from_utt("Ses01M_impro01_M000"), "male")
        self.assertEqual(infer_gender_from_utt("bad_id"), "unknown")
        self.assertEqual(infer_gender_from_utt(None), "unknown")

    def test_infer_session_from_utterance_id(self):
        self.assertEqual(infer_session_from_utt("Ses03F_script01_1_F000"), "Ses03")
        self.assertEqual(infer_session_from_utt("bad_id"), "Unknown")

    def test_resolve_iemocap_audio_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "Session1" / "sentences" / "wav" / "Ses01F.wav"
            wav.parent.mkdir(parents=True)
            wav.write_bytes(b"")

            self.assertEqual(resolve_iemocap_audio_path(root, "Session1/sentences/wav/Ses01F.wav"), str(wav))
            self.assertEqual(resolve_iemocap_audio_path(root, "/Session1/sentences/wav/Ses01F.wav"), str(wav))


if __name__ == "__main__":
    unittest.main()
