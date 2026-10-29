"""Model-stage contracts using explicit test doubles; GPU execution is separate."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from star_dro.cli import parser
from star_dro import inference, merging
from star_dro.execution import run_cli

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def args(self, root, *extra):
        return parser().parse_args(["predict", "--checkpoint", str(root / "run"),
                                   "--input", str(root / "input.json"),
                                   "--output", str(root / "predictions.jsonl"), *extra])

    def setup_run(self, root, adapter=True):
        checkpoint = root / "run/epoch-002"
        checkpoint.mkdir(parents=True)
        (checkpoint.parent / "selection.json").write_text('{"checkpoint": "epoch-002"}')
        if adapter:
            (checkpoint / "adapter_config.json").write_text('{}')
        (root / "input.json").write_text(json.dumps([{"id": "x", "prompt": "Office hours?"}]))
        return checkpoint

    def test_auto_merge_selection_and_cleanup_on_success_or_failure(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkpoint = self.setup_run(root)
                exported = []
                def merge_stage(command):
                    self.assertEqual(command[:3], ["merge", "--checkpoint", str(checkpoint.resolve())])
                    path = Path(command[command.index("--output") + 1])
                    path.mkdir()
                    (path / "merge.json").write_text('{"safe_merge": true}')
                    exported.append(path)
                def generate(args, selected, model_path, rows, metadata):
                    self.assertTrue(model_path.is_dir())
                    self.assertEqual(selected, checkpoint.resolve())
                    self.assertEqual(rows[0]["id"], "x")
                    self.assertTrue(metadata["safe_merge"])
                    if failed:
                        raise RuntimeError("test generation failure")
                with patch("star_dro.execution.run_cli", side_effect=merge_stage), patch.object(inference, "require_vllm"), patch.object(inference, "_predict_vllm", side_effect=generate):
                    if failed:
                        with self.assertRaisesRegex(RuntimeError, "test generation failure"):
                            inference.predict(self.args(root))
                    else:
                        inference.predict(self.args(root))
                self.assertEqual(len(exported), 1)
                self.assertFalse(exported[0].parent.exists())
                self.assertTrue(checkpoint.is_dir())

    def test_full_checkpoint_skips_adapter_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self.setup_run(root, adapter=False)
            with patch("star_dro.execution.run_cli") as merge_stage, patch.object(inference, "require_vllm"), patch.object(inference, "_predict_vllm") as generate:
                inference.predict(self.args(root))
            merge_stage.assert_not_called()
            self.assertEqual(generate.call_args.args[2], checkpoint.resolve())

    def test_vllm_tokens_sampling_output_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            checkpoint = self.setup_run(root, adapter=False)
            (checkpoint / "generation_config.json").write_text('{"eos_token_id": [2, 9]}')
            args = self.args(root, "--seed", "42", "--max-new-tokens", "32", "--tensor-parallel-size", "2")
            tokenizer = MagicMock(eos_token_id=2)
            tokenizer.return_value = {"input_ids": [101, 5, 6]}
            transformers = SimpleNamespace(AutoTokenizer=MagicMock())
            transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
            text = '{"results": [{"Code": "Topic", "Sub-code": "Schedule", "Span": "Office"}]}'
            vllm = SimpleNamespace(LLM=MagicMock(), SamplingParams=MagicMock())
            engine = vllm.LLM.return_value
            engine.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason="stop")])]
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2, is_bf16_supported=lambda: True))
            with patch.dict("sys.modules", {"torch": torch, "transformers": transformers, "vllm": vllm}):
                inference._predict_vllm(args, checkpoint, checkpoint, [{"id": "x", "prompt": "Office hours?"}], None)
            tokenizer.assert_called_once_with("Office hours?", add_special_tokens=False)
            self.assertEqual(engine.generate.call_args.args[0], [{"prompt_token_ids": [101, 5, 6]}])
            self.assertEqual(vllm.LLM.call_args.kwargs["tensor_parallel_size"], 2)
            self.assertEqual(vllm.LLM.call_args.kwargs["generation_config"], "vllm")
            sampling = vllm.SamplingParams.call_args.kwargs
            self.assertEqual(sampling["temperature"], 0)
            self.assertEqual(sampling["max_tokens"], 32)
            self.assertEqual(sampling["seed"], 42)
            self.assertEqual(sampling["stop"], ["JSON_END"])
            self.assertEqual(sampling["stop_token_ids"], [2, 9])
            result = json.loads(Path(args.output).read_text())
            self.assertEqual(result["id"], "x")
            self.assertEqual(result["prediction"], text)
            self.assertIsNone(result["parse_error"])
            metadata = json.loads(Path(args.output + ".meta.json").read_text())
            self.assertEqual(metadata["backend"], "vllm")
            self.assertEqual(metadata["sampling_params"]["temperature"], 0)

    def test_invalid_prediction_settings_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for extra in (("--device", "cpu"), ("--tensor-parallel-size", "0"),
                          ("--batch-size", "0"), ("--gpu-memory-utilization", "nan"),
                          ("--max-new-tokens", "8192"), ("--stop", "")):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    inference.validate_prediction_options(self.args(root, *extra))

    def test_overlong_prompt_is_rejected_without_loading_vllm_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self.setup_run(root, adapter=False)
            args = self.args(root, "--max-length", "8", "--max-new-tokens", "4")
            tokenizer = MagicMock(eos_token_id=2)
            tokenizer.return_value = {"input_ids": [1] * 5}
            transformer = SimpleNamespace(AutoTokenizer=MagicMock())
            transformer.AutoTokenizer.from_pretrained.return_value = tokenizer
            vllm = SimpleNamespace(LLM=MagicMock(), SamplingParams=MagicMock())
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True))
            with patch.dict("sys.modules", {"torch": torch, "transformers": transformer, "vllm": vllm}), self.assertRaisesRegex(ValueError, "exceeds max-length"):
                inference._predict_vllm(args, checkpoint, checkpoint, [{"id": "x", "prompt": "Office hours?"}], None)
            vllm.LLM.assert_not_called()
            self.assertFalse(Path(args.output + ".meta.json").exists())

    def test_merge_uses_recorded_base_revision_and_exports_standalone(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            checkpoint = self.setup_run(root)
            (checkpoint / "run.json").write_text(json.dumps({"identity": {"model": "recorded-base", "revision": "branch"}, "model_commit": "fixed-commit"}))
            args = parser().parse_args(["merge", "--checkpoint", str(root / "run"), "--output", str(root / "merged")])
            auto_model, tokenizer, adapter = MagicMock(), MagicMock(), MagicMock()
            merged = adapter.from_pretrained.return_value.merge_and_unload.return_value
            merged.save_pretrained.side_effect = lambda path, **kw: Path(path).mkdir()
            torch = SimpleNamespace(bfloat16="bf16", float32="fp32")
            with patch.object(merging, "dependencies", return_value=(torch, auto_model, tokenizer)), patch.dict("sys.modules", {"peft": SimpleNamespace(PeftModel=adapter)}):
                merging.merge(args)
            self.assertEqual(auto_model.from_pretrained.call_args.args, ("recorded-base",))
            self.assertEqual(auto_model.from_pretrained.call_args.kwargs["revision"], "fixed-commit")
            self.assertEqual(auto_model.from_pretrained.call_args.kwargs["device_map"], {"": "cpu"})
            adapter.from_pretrained.return_value.merge_and_unload.assert_called_once_with(safe_merge=True)
            self.assertTrue(merged.save_pretrained.call_args.kwargs["safe_serialization"])
            metadata = json.loads((root / "merged/merge.json").read_text())
            self.assertEqual(metadata["checkpoint"], "epoch-002")
            self.assertEqual(metadata["revision"], "fixed-commit")

    def test_model_stages_are_spawn_safe_subprocesses(self):
        with patch("star_dro.execution.subprocess.run") as execute:
            run_cli(["predict", "--checkpoint", "path with spaces"])
        command = execute.call_args.args[0]
        self.assertEqual(command[1:], ["-m", "star_dro", "predict", "--checkpoint", "path with spaces"])
        self.assertTrue(execute.call_args.kwargs["check"])
        self.assertIn("PYTHONPATH", execute.call_args.kwargs["env"])
        for path in [ROOT / "scripts/star_dro.py", ROOT / "src/star_dro/__main__.py"]:
            with patch("star_dro.cli.main") as main:
                if path.name == "__main__.py":
                    runpy.run_module("star_dro.__main__", run_name="__mp_main__")
                else:
                    runpy.run_path(str(path), run_name="__mp_main__")
                main.assert_not_called()
