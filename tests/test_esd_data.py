import tempfile
import unittest
from pathlib import Path

from mer.data import (
    is_english_esd_speaker_path,
    read_esd_transcript,
    resolve_esd_wav_path,
    speaker_id_from_esd_path,
    utterance_id_from_esd_path,
)


class EsdDataHelperTests(unittest.TestCase):
    def test_speaker_and_utterance_parsing(self):
        path = "downloads/esd/0011/Angry/0011_000123.wav"

        self.assertEqual(speaker_id_from_esd_path(path), "0011")
        self.assertEqual(utterance_id_from_esd_path(path), "0011_000123")
        self.assertTrue(is_english_esd_speaker_path(path))
        self.assertFalse(is_english_esd_speaker_path("downloads/esd/0001/Angry/0001_000123.wav"))

    def test_speaker_parsing_falls_back_to_basename(self):
        self.assertEqual(speaker_id_from_esd_path("/tmp/0012_000001.wav"), "0012")

    def test_resolve_esd_wav_path_checks_root_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "audio"
            wav = root / "downloads" / "esd" / "0011" / "Neutral" / "0011_000001.wav"
            wav.parent.mkdir(parents=True)
            wav.write_bytes(b"")

            found = resolve_esd_wav_path(root, "downloads/esd/0011/Neutral/0011_000001.wav")
            self.assertEqual(found, str(wav))

    def test_read_esd_transcript_strips_utterance_id_and_trailing_emotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            speaker_dir = root / "downloads" / "esd" / "0011"
            speaker_dir.mkdir(parents=True)
            transcript_file = speaker_dir / "0011.txt"
            transcript_file.write_text(
                "0011_000001 hello there Neutral\n"
                "0011_000002 I am not the target Happy\n",
                encoding="utf-8",
            )

            wav = speaker_dir / "Neutral" / "0011_000001.wav"
            wav.parent.mkdir()
            wav.write_bytes(b"")

            self.assertEqual(read_esd_transcript(root, wav), "hello there")


if __name__ == "__main__":
    unittest.main()
