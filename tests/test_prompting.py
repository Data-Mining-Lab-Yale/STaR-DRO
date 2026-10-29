"""End-to-end task-instruction assembly and grounding checks."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from star_dro.data import read_records
from star_dro.prompting import build_record, read_instructions
from test_data_io import workspace

ROOT = Path(__file__).resolve().parents[1]


def source():
    return {"id": "document-01:sentence-01", "target_text": "The office opens at nine.",
            "context": "An opening-hours request.", "metadata": {"direction": "outbound"},
            "annotations": [{"Code": "Topic", "Sub-code": "Schedule", "Span": "opens at nine"}]}


INSTRUCTION = "Extract Topic/Schedule from TARGET TEXT. Return results with Code, Sub-code, Span."


class PromptPreparationTests(unittest.TestCase):
    def test_instruction_and_instance_are_visible_gold_stays_separate(self):
        raw = source()
        row = build_record(raw, INSTRUCTION, {"Topic": ["Schedule"]})
        self.assertEqual(row["messages"], [{"role": "user", "content":
            INSTRUCTION + '\n\nMETADATA:\n{"direction": "outbound"}'
            + "\n\nCONTEXT (for interpretation):\nAn opening-hours request."
            + "\n\nTARGET TEXT (evidence source):\nThe office opens at nine."}])
        self.assertEqual(row["annotations"], raw["annotations"])
        self.assertEqual(row["source_text"], raw["target_text"])
        self.assertEqual(row["output_format"], "results")
        self.assertNotIn(json.dumps(raw["annotations"]), row["messages"][0]["content"])

    def test_completed_markdown_reads_only_the_instruction_block(self):
        with workspace() as directory:
            path = directory / "task.md"
            path.write_text("# Task\n\nUsage guidance.\n\n```text\n" + INSTRUCTION + "\n```\n", encoding="utf8")
            self.assertEqual(read_instructions(path), INSTRUCTION)
            path.write_text("# Missing instruction", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                read_instructions(path)

    def test_template_slots_require_task_definitions(self):
        for name in ["structured_prediction.md", "eppc_miner.md"]:
            with self.subTest(template=name), self.assertRaisesRegex(ValueError, "placeholders"):
                read_instructions(ROOT / "prompts" / name)

    def test_ontology_and_grounding_checked_during_assembly(self):
        with self.assertRaisesRegex(ValueError, "outside the ontology"):
            build_record(source(), INSTRUCTION, {"Topic": ["Different"]})
        raw = source()
        raw["annotations"][0]["Span"] = "unsupported evidence"
        with self.assertRaisesRegex(ValueError, "exact substring"):
            build_record(raw, INSTRUCTION)

    def test_cli_builds_model_ready_jsonl_and_preserves_source(self):
        spec = importlib.util.spec_from_file_location("build_records", ROOT / "scripts/build_records.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with workspace() as directory, contextlib.redirect_stdout(io.StringIO()):
            raw, task, output = directory / "source.json", directory / "task.md", directory / "prepared.jsonl"
            raw.write_text(json.dumps(source()), encoding="utf8")
            task.write_text("# Instruction\n\n```text\n" + INSTRUCTION + "\n```\n", encoding="utf8")
            with patch("sys.argv", ["build_records.py", "--input", str(raw), "--instructions", str(task),
                                     "--output", str(output)]):
                module.main()
            self.assertEqual(read_records(output), [build_record(source(), INSTRUCTION)])
            self.assertEqual(json.loads(raw.read_text(encoding="utf8")), source())
