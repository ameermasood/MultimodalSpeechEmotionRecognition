import json
import os
import tempfile
import unittest

from mer.training.esd import load_esd_training_records, split_balanced_train_val, transcript_pool_from_records


class TrainingEsdTests(unittest.TestCase):
    def test_load_esd_training_records_filters_and_resolves_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = os.path.join(tmpdir, "meta")
            audio_root = os.path.join(tmpdir, "audio")
            fold_dir = os.path.join(meta_dir, "esd", "fold_2")
            wav_dir = os.path.join(audio_root, "downloads", "esd", "0011", "Angry")
            os.makedirs(fold_dir)
            os.makedirs(wav_dir)

            wav_rel = "downloads/esd/0011/Angry/0011_000001.wav"
            wav_abs = os.path.join(audio_root, wav_rel)
            open(wav_abs, "wb").close()

            with open(os.path.join(audio_root, "downloads", "esd", "0011", "0011.txt"), "w", encoding="utf-8") as f:
                f.write("0011_000001 hello there Angry\n")

            jsonl = os.path.join(fold_dir, "esd_train_fold_2.jsonl")
            with open(jsonl, "w", encoding="utf-8") as f:
                f.write(json.dumps({"wav": wav_rel, "emo": "angry"}) + "\n")
                f.write(json.dumps({"wav": "downloads/esd/0001/Angry/missing.wav", "emo": "angry"}) + "\n")

            records, stats = load_esd_training_records(meta_dir, audio_root, 2, include_transcripts=True)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["audio_path"], wav_abs)
            self.assertEqual(records[0]["label"], "Angry")
            self.assertEqual(records[0]["transcript"], "hello there")
            self.assertEqual(stats["kept"], 1)

    def test_split_balanced_train_val_balances_remaining_training_records(self):
        records = []
        for label, count in [("Angry", 4), ("Happy", 3), ("Sad", 5), ("Neutral", 3)]:
            records.extend({"audio_path": f"{label}-{i}.wav", "label": label} for i in range(count))

        train, val = split_balanced_train_val(records, val_per_class=1, seed=7)

        train_counts = {label: sum(1 for row in train if row["label"] == label) for label in ["Angry", "Happy", "Sad", "Neutral"]}
        val_counts = {label: sum(1 for row in val if row["label"] == label) for label in ["Angry", "Happy", "Sad", "Neutral"]}

        self.assertEqual(set(train_counts.values()), {2})
        self.assertEqual(set(val_counts.values()), {1})

    def test_transcript_pool_ignores_blank_transcripts(self):
        pool = transcript_pool_from_records([{"transcript": "hello"}, {"transcript": "  "}, {"label": "Angry"}])

        self.assertEqual(pool, ["hello"])


if __name__ == "__main__":
    unittest.main()
