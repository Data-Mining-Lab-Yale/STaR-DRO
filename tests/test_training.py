"""Offline model integration on fictional unit-test records, never clinical data."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from test_data_io import workspace

HAS_MODEL = all(importlib.util.find_spec(name) for name in ["torch", "transformers", "peft", "tokenizers"])


@unittest.skipUnless(HAS_MODEL, "torch/transformers/peft/tokenizers required")
class TrainingIntegrationTests(unittest.TestCase):
    def test_local_lora_training_resume_and_merge(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from star_dro.cli import main
        from star_dro import training
        torch.set_num_threads(1)
        with workspace() as directory, contextlib.redirect_stdout(io.StringIO()):
            model_dir = directory / "model"
            vocab = {s: i for i, s in enumerate(["<unk>", "<eos>", "<pad>"] + [chr(i) for i in range(32, 127)] + ["\n"])}
            backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
            backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
            tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>")
            tok.save_pretrained(model_dir)
            torch.manual_seed(4)
            model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=512, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id))
            model.save_pretrained(model_dir)
            for split in ["train", "valid"]:
                rows = [{"id": f"{split}-{i}", "prompt": f"{split} {i}:\n", "annotations":
                         [{"Code": code, "Sub-code": "kind", "Span": "fictional"}]} for i, code in enumerate(["A", "B"])]
                (directory / f"{split}.json").write_text(json.dumps(rows), encoding="utf8")
            cfg = {"training": {"batch_size": 1, "gradient_checkpointing": True},
                   "lora": {"rank": 2, "alpha": 4, "dropout": 0.0}}
            config = directory / "config.json"
            config.write_text(json.dumps(cfg), encoding="utf8")
            base = ["train", "--model", str(model_dir), "--train", str(directory / "train.json"),
                    "--valid", str(directory / "valid.json"), "--config", str(config),
                    "--device", "cpu", "--precision", "float32", "--max-length", "256"]
            full, resumed = directory / "full", directory / "resumed"
            main(base + ["--output", str(full)])
            append = training.append_jsonl

            class Interrupted(Exception):
                pass

            def interrupt_after_checkpoint(path, row):
                if Path(path).name == "train.jsonl" and row["epoch"] == 2:
                    raise Interrupted()
                append(path, row)

            with patch.object(training, "append_jsonl", side_effect=interrupt_after_checkpoint):
                with self.assertRaises(Interrupted):
                    main(base + ["--output", str(resumed)])
            main(base + ["--output", str(resumed), "--resume", str(resumed / "epoch-001")])
            read = lambda p: json.loads(p.read_text(encoding="utf8"))
            self.assertEqual(read(full / "epoch-003/star_dro.json"), read(resumed / "epoch-003/star_dro.json"))
            self.assertEqual(read(full / "selection.json"), read(resumed / "selection.json"))
            for name in ["train.jsonl", "validation.jsonl"]:
                self.assertEqual((full / name).read_text(), (resumed / name).read_text())
            state = read(full / "epoch-003/star_dro.json")
            self.assertEqual(state["steps"], 4)
            # Default selection reports the final epoch of the continuous run.
            self.assertEqual(read(full / "selection.json")["checkpoint"], "epoch-003")
            self.assertEqual(read(full / "selection.json")["criterion"], "last")
            self.assertTrue(read(full / "selection.json")["robust_objective_active"])
            self.assertEqual(read(full / "epoch-001/star_dro.json")["steps"], 0)
            merged_dir = directory / "merged"
            main(["merge", "--checkpoint", str(full), "--output", str(merged_dir),
                  "--device", "cpu", "--precision", "float32"])
            from peft import PeftModel
            from star_dro.inference import resolve_checkpoint
            adapter_model = PeftModel.from_pretrained(LlamaForCausalLM.from_pretrained(model_dir), resolve_checkpoint(full))
            merged_model = LlamaForCausalLM.from_pretrained(merged_dir)
            adapter_model.eval()
            merged_model.eval()
            with torch.no_grad():
                inputs = torch.tensor([[3, 4, 5]])
                torch.testing.assert_close(adapter_model(inputs).logits, merged_model(inputs).logits, rtol=1e-4, atol=1e-5)
            self.assertTrue((merged_dir / "model.safetensors").is_file())
            self.assertFalse((merged_dir / "adapter_config.json").exists())
            self.assertEqual(read(merged_dir / "merge.json")["checkpoint"], read(full / "selection.json")["checkpoint"])
            # Verify that the CLI routes the alternative objective and baseline,
            # rather than testing only their standalone tensor functions.
            cfg["grouping"] = {"groupby": "code_subcode_num_annotations", "granularity": "sample"}
            config.write_text(json.dumps(cfg), encoding="utf8")
            sample = directory / "sample"
            main(base + ["--output", str(sample)])
            self.assertEqual(read(sample / "epoch-003/star_dro.json")["steps"], 4)
            self.assertEqual(read(sample / "run.json")["identity"]["group_inventory"]["groupby"],
                             "code_subcode_num_annotations")
            cfg["training"]["objective"] = "sft"
            config.write_text(json.dumps(cfg), encoding="utf8")
            sft = directory / "sft"
            main(base + ["--output", str(sft)])
            state = read(sft / "epoch-003/star_dro.json")
            self.assertEqual(state["steps"], 0)
            self.assertEqual(state["ema"], [None, None])
            cfg["training"]["checkpoint_selection"] = "validation_ce"
            config.write_text(json.dumps(cfg), encoding="utf8")
            chosen = directory / "chosen"
            main(base + ["--output", str(chosen)])
            ces = [json.loads(line)["completion_ce"] for line in (chosen / "validation.jsonl").read_text().splitlines()]
            selection = read(chosen / "selection.json")
            self.assertEqual(selection["criterion"], "validation_ce")
            self.assertEqual(selection["epoch"], 1 + ces.index(min(ces)))
            cfg["training"]["checkpoint_selection"] = "last"
            cfg["training"]["objective"] = "standard_dro"
            cfg["grouping"]["granularity"] = "annotation"
            config.write_text(json.dumps(cfg), encoding="utf8")
            standard = directory / "standard"
            main(base + ["--output", str(standard)])
            state = read(standard / "epoch-003/standard_dro.json")
            self.assertEqual(state["objective"], "standard_dro")
            self.assertEqual(state["steps"], 4)
            self.assertTrue(read(standard / "selection.json")["checkpoint"].startswith("epoch-"))



if __name__ == "__main__":
    unittest.main()
