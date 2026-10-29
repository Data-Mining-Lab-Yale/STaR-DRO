"""STaR-DRO data-interface checks using the single documented format record."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import unittest
import uuid
from star_dro.cli import main
from star_dro.data import encode_record, normalize_record, read_records, validate_record, write_records
from star_dro.utils import file_hash
from star_dro.inference import resolve_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def documented_record():
    blocks = re.findall(r"```json\s*\n(.*?)```", (ROOT/"README.md").read_text(encoding="utf8"), re.S)
    if len(blocks) != 1:
        raise AssertionError("README must contain one input-format record")
    raw = json.loads(blocks[0])
    if "target_text" in raw:
        from star_dro.prompting import build_record
        return build_record(raw, "Extract Topic/Schedule and Courtesy/Thanks annotations with exact evidence. "
                            "Return a JSON object with a results array containing Code, Sub-code, Span.")
    return raw


@contextlib.contextmanager
def workspace():
    parent = ROOT/"outputs"
    parent.mkdir(exist_ok=True)
    path = parent/("test-"+uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        assert path.resolve().parent == parent.resolve()
        shutil.rmtree(path)


class DataInterfaceTests(unittest.TestCase):
    def test_documented_record_matches_runtime_contract(self):
        row = documented_record()
        self.assertIs(validate_record(row), row)
        self.assertTrue(all(a["Span"] in row["source_text"] for a in row["annotations"]))

    def test_json_and_jsonl_normalization(self):
        row = documented_record()
        with workspace() as directory:
            for filename, value in [("record.json", row), ("records.json", [row])]:
                path = directory/filename
                path.write_text(json.dumps(value), encoding="utf8")
                self.assertEqual(read_records(path), [row])
            path = directory/"records.jsonl"
            write_records([row], path)
            self.assertEqual(read_records(path), [row])

    def test_existing_exports_preserve_results_envelope(self):
        row = documented_record()
        answer = json.dumps({"results": row["annotations"]})
        entries = [{"role":"human" if m["role"] == "user" else m["role"], "content":m["content"]} for m in row["messages"]]
        entries.append({"role":"assistant", "content":answer})
        converted = normalize_record({"id":7, "entries":entries})
        self.assertEqual(converted["id"], "7")
        self.assertEqual(converted["annotations"], row["annotations"])
        self.assertEqual(converted["messages"], row["messages"])
        self.assertEqual(converted["output_format"], "results")
        converted = normalize_record({"id":"query-record", "query":row["source_text"], "answer":answer})
        self.assertEqual(converted["output_format"], "results")
        self.assertEqual(converted["annotations"], row["annotations"])

    def test_missing_source_ids_are_stable(self):
        row = documented_record()
        source = {"query":row["source_text"], "answer":json.dumps(row["annotations"])}
        self.assertEqual(normalize_record(source)["id"], normalize_record(source)["id"])

    def test_input_only_preparation_does_not_export_labels(self):
        row = documented_record()
        with workspace() as directory, contextlib.redirect_stdout(io.StringIO()):
            source = directory/"gold.json"
            source.write_text(json.dumps(row), encoding="utf8")
            output = directory/"inputs.jsonl"
            main(["prepare-data", "--input", str(source), "--output", str(output), "--input-only"])
            prepared = read_records(output, require_annotations=False)
            self.assertEqual(prepared[0]["id"], row["id"])
            self.assertNotIn("annotations", prepared[0])
            self.assertEqual(prepared[0]["messages"], row["messages"])

    def test_malformed_and_duplicate_records_raise(self):
        row = documented_record()
        with workspace() as directory:
            path = directory/"data.jsonl"
            path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                read_records(path)
            path.write_text("{broken}\n", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                read_records(path)
        with self.assertRaises(ValueError):
            normalize_record({"query":row["source_text"], "answer":"not JSON"})

    def test_output_envelopes_preserve_annotation_alignment(self):
        class CharacterTokenizer:
            eos_token_id = 999
            def __call__(self, text, **kwargs):
                return {"input_ids":[ord(c) for c in text], "offset_mapping":[(j,j+1) for j in range(len(text))]}
        row = documented_record()
        row.pop("messages")
        row["prompt"] = "Assistant:\n"
        groups = sorted({ann["Code"] for ann in row["annotations"]})
        for envelope in ["list", "results"]:
            row["output_format"] = envelope
            encoded = encode_record(row, CharacterTokenizer(), groups, 4096)
            text = ''.join(chr(t) for t in encoded["input_ids"][:-1])
            answer = json.loads(text[len(row["prompt"]):])
            self.assertEqual(answer["results"] if envelope == "results" else answer, row["annotations"])
            for ann, mask in zip(row["annotations"], encoded["masks"]):
                covered = ''.join(chr(encoded["input_ids"][j+1]) for j, active in enumerate(mask) if active)
                self.assertEqual(json.loads(covered), ann)

    def test_validation_selected_checkpoint_resolution(self):
        with workspace() as directory:
            checkpoint = directory/"epoch-002"
            checkpoint.mkdir()
            (directory/"selection.json").write_text(json.dumps({"checkpoint":"epoch-002"}), encoding="utf8")
            self.assertEqual(resolve_checkpoint(directory), checkpoint.resolve())
            self.assertEqual(resolve_checkpoint(checkpoint), checkpoint)

    def test_checkpoint_selection_cannot_escape_run(self):
        with workspace() as directory:
            (directory/"selection.json").write_text(json.dumps({"checkpoint":"../other-run"}), encoding="utf8")
            with self.assertRaises(ValueError):
                resolve_checkpoint(directory)

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is not installed")
    def test_real_arrow_roundtrip_and_directory_hash(self):
        row = documented_record()
        with workspace() as directory:
            saved = directory/"saved-split"
            write_records([row], saved, storage="arrow")
            self.assertEqual(read_records(saved), [row])
            self.assertEqual(read_records(next(saved.glob("*.arrow"))), [row])
            self.assertEqual(file_hash(saved), file_hash(saved))

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is not installed")
    def test_notebook_arrow_layouts_roundtrip(self):
        from datasets import load_from_disk
        row = documented_record()
        # query/answer requires one complete user turn; preserve task content.
        row["messages"] = [{"role": "user", "content": "\n".join(m["content"] for m in row["messages"])}]
        with workspace() as directory:
            for layout, columns in [("entries", {"id", "entries"}), ("query_answer", {"id", "query", "answer"})]:
                target = directory/layout
                write_records([row], target, storage="arrow", layout=layout)
                self.assertEqual(set(load_from_disk(str(target)).column_names), columns)
                loaded = read_records(target)[0]
                self.assertEqual(loaded["annotations"], row["annotations"])
                self.assertEqual(loaded["messages"], row["messages"])
                self.assertEqual(loaded["output_format"], "results")
                self.assertEqual(read_records(next(target.glob("*.arrow")))[0], loaded)

    def test_notebook_export_rejects_lossy_query_conversion(self):
        row = documented_record()
        row["messages"].insert(0, {"role": "system", "content": "Use the supplied annotation definitions."})
        with workspace() as directory:
            with self.assertRaisesRegex(ValueError, "exactly one user turn"):
                write_records([row], directory/"invalid.jsonl", layout="query_answer")

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is not installed")
    def test_reference_schemas_preserve_context_and_direction(self):
        from datasets import load_from_disk
        spec = importlib.util.spec_from_file_location("prepare_notebook", ROOT/"scripts/prepare_notebook_data.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = documented_record()
        query = "\n".join(m["content"] for m in row["messages"])
        answer = json.dumps({"results": row["annotations"]})
        with workspace() as directory:
            for layout, source in [
                ("entries", {"id": "source-001", "entries": [{"content": query, "role": "human"},
                                                               {"content": answer, "role": "assistant"}]}),
                ("query_answer", {"query": query, "answer": answer, "context": row["source_text"], "TO_PAT_YN": "Y"}),
            ]:
                path = directory/(layout+".json")
                path.write_text(json.dumps(source), encoding="utf8")
                output = directory/layout
                module.prepare(path, output, layout)
                saved = load_from_disk(str(output))
                self.assertEqual(set(saved.column_names), set(source))
                self.assertEqual(saved[0], source)
                self.assertEqual(read_records(output)[0]["annotations"], row["annotations"])


if __name__ == "__main__":
    unittest.main()
