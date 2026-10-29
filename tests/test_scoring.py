"""Ontology-driven scoring: label inventories come from the user's ontology or gold."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from star_dro import scoring
from star_dro.metrics import evaluate

ONTOLOGY = {"Topic": ["Schedule", "Location"], "Courtesy": ["Thanks", "Greeting"], "Request": ["Schedule"]}


def ann(code="Topic", sub="Schedule", span="The office opens at nine."):
    return {"Code": code, "Sub-code": sub, "Span": span}


def run(truth, prediction, **kw):
    return evaluate([{"id": "x", "annotations": truth}], [{"id": "x", "prediction": prediction}],
                    ontology=kw.pop("ontology", ONTOLOGY), **kw)


class InventoryTests(unittest.TestCase):
    def test_no_builtin_label_inventory(self):
        self.assertFalse(hasattr(scoring, "Code_set"))
        self.assertFalse(hasattr(scoring, "Sub_Code_set"))

    def test_any_task_ontology_scores_perfect_prediction(self):
        gold = [ann(), ann("Courtesy", "Thanks", "Thank you.")]
        result = run(gold, json.dumps({"results": gold}))
        for key in ("code", "subcode", "code_subcode", "span"):
            self.assertEqual(result["metrics"][key]["f1"], 1.0, key)
        self.assertEqual(result["label_inventory_source"], "ontology")
        self.assertEqual(result["label_inventory"], {"codes": 3, "subcodes": 4})

    def test_inventory_from_gold_when_no_ontology(self):
        gold = [ann("Café", "Menú", "abre a las nueve")]
        result = evaluate([{"id": "x", "annotations": gold}], [{"id": "x", "prediction": json.dumps(gold)}])
        self.assertEqual(result["metrics"]["code"]["f1"], 1.0)
        self.assertEqual(result["label_inventory_source"], "gold_annotations")

    def test_normalization_and_fuzzy_mapping(self):
        alias = ann(" topic ", "SCHEDULE")
        self.assertEqual(run([ann()], json.dumps([alias]))["metrics"]["code"]["f1"], 1.0)
        inv = scoring.LabelInventory({"PartnershipProvider": ["build trust", "Approval/Reinforcement"]})
        self.assertEqual(inv.code("partnership_provider"), "PartnershipProvider")
        self.assertEqual(inv.subcode("Build Trust"), "build trust")
        self.assertEqual(inv.subcode("approval / reinforcement"), "Approval/Reinforcement")
        self.assertEqual(inv.code("PartnershipProvidr"), "PartnershipProvider")  # fuzzy near miss
        self.assertIsNone(scoring.LabelInventory(inv.ontology, fuzzy=False).code("PartnershipProvidr"))
        self.assertIsNone(inv.code("Partnership"))

    def test_indistinguishable_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            scoring.LabelInventory({"Build Trust": ["a"], "build_trust": ["b"]})

    def test_gold_outside_inventory_raises(self):
        with self.assertRaises(ValueError):
            run([ann("Unlisted", "Schedule")], "[]")


class MetricTests(unittest.TestCase):
    def test_set_semantics_and_wrong_parent(self):
        result = run([ann()], json.dumps([ann("Request", "Schedule"), ann("Request", "Schedule")]))
        self.assertEqual(result["metrics"]["code"]["fp"], 1)
        self.assertEqual(result["metrics"]["code"]["fn"], 1)
        self.assertEqual(result["metrics"]["subcode"]["f1"], 1.0)  # Sub-code names are scored independently
        self.assertEqual(result["metrics"]["code_subcode"]["f1"], 0.0)

    def test_hierarchy_violations_are_reported(self):
        result = run([ann()], json.dumps([ann("Courtesy", "Schedule")]))
        self.assertEqual(result["predicted_pairs_outside_hierarchy"], 1)

    def test_unknown_label_policies(self):
        pred = json.dumps([ann(), ann("Invented", "Made up")])
        ignored = run([ann()], pred)
        self.assertEqual(ignored["metrics"]["code"]["precision"], 1.0)
        self.assertEqual(ignored["unknown_predicted_code_labels"], 1)
        counted = run([ann()], pred, unknown_labels="count")
        self.assertEqual(counted["metrics"]["code"]["fp"], 1)
        self.assertEqual(counted["metrics"]["subcode"]["fp"], 1)
        self.assertEqual(counted["metrics"]["code_subcode"]["fp"], 1)
        with self.assertRaises(ValueError):
            run([ann()], pred, unknown_labels="other")

    def test_span_containment_jaccard_threshold_and_one_to_one(self):
        self.assertEqual(run([ann()], json.dumps([ann(span="office opens")]))["metrics"]["span"]["f1"], 1.0)
        for span, expected in [("one two three five", 1.0), ("one two five six", 0.0)]:
            r = run([ann(span="one two three four")], json.dumps([ann(span=span)]))
            self.assertEqual(r["metrics"]["span"]["f1"], expected)
        r = run([ann(span="one two three four")], json.dumps([ann(span="one two five six")]), threshold=0.3)
        self.assertEqual(r["metrics"]["span"]["f1"], 1.0)
        r = run([ann()], json.dumps([ann(), ann()]))
        self.assertEqual((r["metrics"]["span"]["tp"], r["metrics"]["span"]["fp"]), (1, 1))
        self.assertAlmostEqual(r["metrics"]["span"]["f1"], 2/3)

    def test_empty_predicted_span_matches_nothing(self):
        r = run([ann()], json.dumps({"results": [ann(span="")]}))
        self.assertEqual(r["metrics"]["span"]["tp"], 0)

    def test_output_repair_and_envelopes(self):
        text = json.dumps({"results": [ann()]})
        for prediction in [text, json.dumps([ann()]), repr({"results": [ann()]}),
                           "```json\n" + text + "\n```", text + "\nJSON_END", text[:-2]]:
            with self.subTest(prediction=prediction):
                self.assertEqual(run([ann()], prediction)["metrics"]["code"]["f1"], 1.0)
        self.assertEqual(run([ann()], "not JSON")["metrics"]["code"]["fn"], 1)
        self.assertEqual(scoring.parse_output('{"results": ["x", 3, {"Code": "Topic"}]}'), [{"Code": "Topic"}])

    def test_empty_cases_raw_precedence_and_ids(self):
        self.assertEqual(run([], '{"results": []}')["metrics"]["code"]["f1"], 0.0)
        result = evaluate([{"id": "x", "annotations": [ann()]}],
                          [{"id": "x", "prediction": "not JSON", "annotations": [ann()]}], ontology=ONTOLOGY)
        self.assertEqual(result["metrics"]["code"]["fn"], 1)
        self.assertEqual(result["invalid_output_records"], 1)
        gold = [{"id": "a", "annotations": [ann()]}, {"id": "b", "annotations": []}]
        result = evaluate(gold, [{"id": "b", "annotations": []}, {"id": "a", "annotations": [ann()]}], ontology=ONTOLOGY)
        self.assertEqual(result["metrics"]["code"]["f1"], 1.0)
        for bad in ([{"id": "a", "annotations": []}], [{"id": "a", "annotations": []}, {"id": "a", "annotations": []}]):
            with self.assertRaises(ValueError):
                evaluate(gold, bad, ontology=ONTOLOGY)

    def test_grounding_is_exact(self):
        result = evaluate([{"id": "x", "source_text": "one two", "annotations": [ann(span="one two")]}],
                          [{"id": "x", "annotations": [ann(span="ONE TWO")]}], ontology=ONTOLOGY)
        self.assertEqual(result["metrics"]["span"]["f1"], 1.0)
        self.assertEqual(result["grounded_span_fraction"], 0)


class CommandTests(unittest.TestCase):
    def test_cli_scores_with_and_without_ontology(self):
        from star_dro.cli import main
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root/"ontology.json").write_text(json.dumps(ONTOLOGY))
            (root/"gold.json").write_text(json.dumps([{"id": "x", "prompt": "Fictional office text:", "annotations": [ann()]}]))
            (root/"pred.jsonl").write_text(json.dumps({"id": "x", "prediction": json.dumps([ann("topic"), ann("Other", "X")])})+"\n")
            base = ["evaluate", "--gold", str(root/"gold.json"), "--predictions", str(root/"pred.jsonl")]
            main(base + ["--ontology", str(root/"ontology.json"), "--output", str(root/"a.json")])
            main(base + ["--unknown-labels", "count", "--exact-labels", "--output", str(root/"b.json")])
            a, b = (json.loads((root/n).read_text()) for n in ("a.json", "b.json"))
            self.assertEqual(a["metrics"]["code"]["f1"], 1.0)
            self.assertEqual(a["label_inventory_source"], "ontology")
            self.assertEqual(b["metrics"]["code"]["fp"], 1)
            self.assertEqual((b["label_inventory_source"], b["label_matching"]), ("gold_annotations", "normalized"))

    def test_seed_runner_forwards_scoring_options(self):
        spec = importlib.util.spec_from_file_location("run_experiment", Path(__file__).resolve().parents[1]/"scripts/run_experiment.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"experiment"
            calls = []

            def stage(args):
                calls.append(args)
                if args[0] == "train":
                    Path(args[args.index("--output")+1]).mkdir()
                if args[0] == "evaluate":
                    Path(args[args.index("--output")+1]).write_text(json.dumps(
                        {"metrics": {m: {k: 1.0 for k in ("precision", "recall", "f1")}
                                     for m in ("code", "subcode", "code_subcode", "span")}, "invalid_output_rate": 0.0}))
            argv = ["run_experiment.py", "--model", "fixture", "--train", "train", "--valid", "valid", "--test", "test",
                    "--output", str(root), "--seeds", "42", "--unknown-labels", "count", "--span-threshold", "0.5"]
            with patch("sys.argv", argv), patch.object(runner, "main", side_effect=stage), \
                    patch.object(runner, "run_cli", side_effect=stage), patch.object(runner, "require_vllm"), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.run()
            evaluation = next(args for args in calls if args[0] == "evaluate")
            self.assertEqual(evaluation[evaluation.index("--unknown-labels")+1], "count")
            self.assertEqual(evaluation[evaluation.index("--span-threshold")+1], "0.5")
            self.assertEqual([args[0] for args in calls], ["validate-data", "train", "predict", "evaluate"])
            self.assertEqual(json.loads((root/"experiment.json").read_text())["scorer"]["name"], "ontology")


if __name__ == "__main__":
    unittest.main()
