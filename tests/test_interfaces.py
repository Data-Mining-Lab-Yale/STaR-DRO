"""STaR-DRO schema, configuration and command-interface checks."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from star_dro.cli import parser
from star_dro.data import ANNOTATION_FIELDS
from star_dro.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


class InterfaceTests(unittest.TestCase):
    def test_schema_fields_match_runtime_contract(self):
        record = json.loads((ROOT/"schemas/record.schema.json").read_text(encoding="utf8"))
        self.assertEqual(set(record["properties"]["annotations"]["items"]["required"]), ANNOTATION_FIELDS)
        self.assertEqual(record["required"], ["id"])
        self.assertFalse(record["additionalProperties"])
        self.assertEqual(len(record["oneOf"]), 2)
        gold = json.loads((ROOT/"schemas/training_record.schema.json").read_text(encoding="utf8"))
        self.assertEqual(gold["allOf"][1]["required"], ["annotations"])

    def test_reference_configuration(self):
        cfg = load_config(ROOT/"configs/reference.json")
        self.assertEqual(cfg["training"]["dro_start_epoch"], 2)
        self.assertLessEqual(cfg["training"]["dro_start_epoch"], cfg["training"]["epochs"])
        self.assertGreater(cfg["star_dro"]["alpha"], 1)
        self.assertGreater(cfg["star_dro"]["upper"], 1)

    def test_checkpoint_selection_default_and_choices(self):
        self.assertEqual(load_config()["training"]["checkpoint_selection"], "last")
        self.assertEqual(load_config(ROOT/"configs/reference.json")["training"]["checkpoint_selection"], "last")
        for value, ok in [("validation_ce", True), ("best", False)]:
            text = json.dumps({"training": {"checkpoint_selection": value}})
            with patch("star_dro.utils.Path.read_text", return_value=text):
                if ok:
                    self.assertEqual(load_config("c.json")["training"]["checkpoint_selection"], value)
                else:
                    with self.assertRaises(ValueError):
                        load_config("c.json")

    def test_configuration_rejects_unknown_keys(self):
        with patch("star_dro.utils.Path.read_text", return_value='{"training":{"unrecognized_key":1}}'):
            with self.assertRaises(ValueError):
                load_config("config.json")

    def test_cli_subcommands_and_required_flags(self):
        command_parser = parser()
        action = next(a for a in command_parser._actions if isinstance(a, argparse._SubParsersAction))
        self.assertEqual(set(action.choices), {"prepare-data", "validate-data", "train", "merge", "predict", "evaluate"})
        required = {a.dest for a in action.choices["train"]._actions if a.required}
        self.assertEqual(required, {"model", "train", "valid", "output"})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            command_parser.parse_args([])


if __name__ == "__main__":
    unittest.main()
