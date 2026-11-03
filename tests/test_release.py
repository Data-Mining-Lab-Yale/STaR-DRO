"""Release packaging must refuse private-data/model files outside excluded roots."""
import importlib.util
from pathlib import Path
import unittest
from test_data_io import workspace

spec = importlib.util.spec_from_file_location("package_release", Path(__file__).resolve().parents[1] / "scripts/package_release.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class ReleaseTests(unittest.TestCase):
    def test_only_approved_source_files_are_selected(self):
        with workspace() as root:
            (root / "README.md").write_text("public source", encoding="utf8")
            (root / "unreviewed-notes.md").write_text("not a release file", encoding="utf8")
            (root / "private").mkdir()
            (root / "private/records.arrow").write_bytes(b"private fixture")
            self.assertEqual(package.release_files(root), [root / "README.md"])

    def test_private_file_types_cannot_enter_source_tree(self):
        for suffix in [".arrow", ".parquet", ".jsonl", ".ipynb", ".safetensors", ".pt"]:
            with self.subTest(suffix=suffix), workspace() as root:
                (root / ("accidental" + suffix)).write_bytes(b"fixture")
                with self.assertRaisesRegex(ValueError, "private data/model/credential"):
                    package.release_files(root)
