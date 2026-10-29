"""STaR-DRO command-line entry points; model dependencies load only when needed."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .data import check_disjoint, load_ontology, read_records, write_records, split_overlap_report
from .metrics import evaluate
from .utils import file_hash, write_json
from .grouping import GroupInventory
from .utils import load_config


def add_vllm_arguments(command):
    command.add_argument("--tensor-parallel-size", type=int, default=1, help="visible GPUs used by vLLM only")
    command.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    command.add_argument("--batch-size", type=int, default=1, help="records per vLLM generate call")
    command.add_argument("--stop", default="JSON_END", help="generation stop string, excluded from output")
    command.add_argument("--enforce-eager", action="store_true", help="disable vLLM CUDA graph capture")
    command.add_argument("--merge-device", default="cpu", help="device for temporary merging: cpu or cuda[:INDEX]")
    command.add_argument("--merge-temp-dir", help="filesystem for temporary full weights; defaults to prediction output parent")


def add_scoring_arguments(command):
    command.add_argument("--span-threshold", type=float, default=0.6, help="token-Jaccard threshold for relaxed span matching")
    command.add_argument("--unknown-labels", choices=["ignore", "count"], default="ignore",
                         help="predicted labels outside the inventory: ignore (manuscript convention) or count as false positives")
    command.add_argument("--exact-labels", action="store_true", help="disable fuzzy label mapping (normalization still applies)")


def parser():
    p = argparse.ArgumentParser(prog="star-dro", description="STaR-DRO: Stateful Tsallis Reweighting")
    p.add_argument("--version", action="version", version="STaR-DRO 0.5.0")
    commands = p.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="normalize JSON/JSONL/Arrow inputs and optionally save Arrow")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--format", choices=["jsonl", "arrow"], default="jsonl")
    prepare.add_argument("--layout", choices=["canonical", "entries", "query_answer"], default="canonical",
                         help="canonical rows or notebook-compatible entries/query-answer columns")
    prepare.add_argument("--ontology")
    prepare.add_argument("--input-only", action="store_true", help="export model inputs without gold annotations")
    valid = commands.add_parser("validate-data", help="validate supported data inputs, ontology, and split separation")
    valid.add_argument("--train", required=True)
    valid.add_argument("--valid")
    valid.add_argument("--test")
    valid.add_argument("--ontology")
    valid.add_argument("--config", help="also validate the configured grouping against the training inventory")
    valid.add_argument("--output")
    valid.add_argument("--allow-split-overlap", action="store_true", help="SMOKE TEST ONLY: preserve overlapping splits and report overlap")
    score = commands.add_parser("evaluate", help="score JSONL predictions against gold annotations")
    score.add_argument("--gold", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--ontology", help="JSON {Code: [Sub-code, ...]} defining the label inventory; "
                       "if omitted, the inventory is taken from the gold annotations")
    add_scoring_arguments(score)
    score.add_argument("--output", required=True)
    train = commands.add_parser("train", help="single-device continuous SFT, STaR-DRO or Standard DRO fine-tuning")
    train.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint")
    train.add_argument("--revision", help="model/tokenizer revision; use a commit hash for reproducibility")
    train.add_argument("--train", required=True)
    train.add_argument("--valid", required=True)
    train.add_argument("--ontology")
    train.add_argument("--config", default=None, help="JSON overrides; default is the STaR-DRO reference config")
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--max-length", type=int, default=8192)
    train.add_argument("--device", default="cuda")
    train.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    train.add_argument("--allow-split-overlap", action="store_true", help="SMOKE TEST ONLY: preserve overlapping splits and record this exception")
    train.add_argument("--resume", help="trusted epoch checkpoint from this output directory")
    merge = commands.add_parser("merge", help="export selected LoRA weights as a standalone safetensors model")
    merge.add_argument("--checkpoint", required=True)
    merge.add_argument("--base-model", help="override the recorded base-model location")
    merge.add_argument("--output", required=True, help="new directory for full merged weights")
    merge.add_argument("--device", default="cpu")
    merge.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    predict = commands.add_parser("predict", help="merge a selected adapter temporarily and generate with vLLM")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--base-model", help="override the base-model location for a LoRA checkpoint")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    predict.add_argument("--max-new-tokens", type=int, default=512)
    predict.add_argument("--max-length", type=int, default=8192)
    predict.add_argument("--seed", type=int, default=0)
    add_vllm_arguments(predict)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "prepare-data":
            ontology = load_ontology(args.ontology)
            rows = read_records(args.input, require_annotations=not args.input_only, ontology=ontology)
            write_records(rows, args.output, args.format, args.input_only, args.layout)
            print(json.dumps({"records": len(rows), "format": args.format,
                              "layout": args.layout, "output": args.output, "input_only": args.input_only}))
        elif args.command == "validate-data":
            ontology = load_ontology(args.ontology)
            splits = {name: read_records(path, ontology=ontology) for name in ["train", "valid", "test"] if (path := getattr(args, name))}
            overlap = split_overlap_report(splits)
            check_disjoint(*splits.values(), allow_overlap=args.allow_split_overlap)
            training_codes = {a["Code"] for row in splits["train"] for a in row["annotations"]}
            if not training_codes:
                raise ValueError("training requires at least one annotated Code")
            report = {"split_overlap": overlap, "allow_split_overlap": args.allow_split_overlap,
                      "evaluation_scope": "smoke_test_with_overlap_allowed" if args.allow_split_overlap else "disjoint_splits"}
            inventory = None
            if args.config:
                cfg = load_config(args.config)
                inventory = GroupInventory.fit(splits["train"], cfg["grouping"]["groupby"], cfg["grouping"]["min_count"])
            for name, rows in splits.items():
                counts = Counter(a["Code"] for row in rows for a in row["annotations"])
                if set(counts)-training_codes:
                    raise ValueError(f"{name}: Codes absent from training: {sorted(set(counts)-training_codes)}")
                if inventory is not None:
                    for row in rows:
                        inventory.encode(row["annotations"])
                report[name] = {"records": len(rows), "annotations": sum(counts.values()),
                                "annotations_by_code": dict(sorted(counts.items())),
                                "sha256": file_hash(getattr(args, name))}
            if inventory is not None:
                report["group_inventory"] = inventory.to_dict()
            if args.output:
                write_json(args.output, report, exclusive=True)
            print(json.dumps(report, indent=2))
            return report
        elif args.command == "evaluate":
            ontology = load_ontology(args.ontology)
            gold = read_records(args.gold, ontology=ontology)
            with open(args.predictions, encoding="utf-8-sig") as handle:
                predictions = [json.loads(line) for line in handle if line.strip()]
            result = evaluate(gold, predictions, args.span_threshold, ontology,
                              unknown_labels=args.unknown_labels, fuzzy=not args.exact_labels)
            result["inputs"] = {"gold_sha256": file_hash(args.gold), "predictions_sha256": file_hash(args.predictions)}
            write_json(args.output, result, exclusive=True)
            print(json.dumps(result, indent=2))
        elif args.command == "train":
            from .training import train
            train(args)
        elif args.command == "merge":
            from .merging import merge
            merge(args)
        elif args.command == "predict":
            from .inference import predict
            predict(args)
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError) as exc:
        p.exit(2, f"STaR-DRO error: {exc}\n")


if __name__ == "__main__":
    main()


