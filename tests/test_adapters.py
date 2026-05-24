import json
import tempfile
import unittest
from pathlib import Path

from mer.modeling import (
    adapter_tag_from_path,
    discover_adapters,
    find_adapter_candidates,
    is_dora_adapter,
    resolve_adapter_dir,
    safe_adapter_name,
)


class AdapterHelperTests(unittest.TestCase):
    def test_resolve_adapter_dir_accepts_adapter_or_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "run_a" / "final_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            self.assertEqual(resolve_adapter_dir(adapter), str(adapter.resolve()))
            self.assertEqual(resolve_adapter_dir(root / "run_a"), str(adapter.resolve()))

    def test_discover_adapters_returns_final_adapter_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "run_a" / "final_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "not_adapter").mkdir()

            self.assertEqual(discover_adapters(root), [str(adapter.resolve())])

    def test_find_adapter_candidates_accepts_weight_only_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "weights_only"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"")

            self.assertEqual(find_adapter_candidates(root), [str(adapter.resolve())])

    def test_is_dora_adapter_reads_adapter_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapter_config.json").write_text(json.dumps({"use_dora": True}), encoding="utf-8")

            self.assertTrue(is_dora_adapter(adapter))

    def test_safe_adapter_names_and_tags(self):
        self.assertEqual(safe_adapter_name("run name!*"), "run_name")
        self.assertEqual(adapter_tag_from_path("/tmp/my_run/final_adapter"), "my_run")


if __name__ == "__main__":
    unittest.main()
