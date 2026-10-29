"""Run STaR-DRO training, selected-checkpoint prediction and scoring across seeds."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from star_dro.cli import main, add_scoring_arguments, add_vllm_arguments
from star_dro.execution import run_cli
from star_dro.inference import require_vllm, validate_prediction_options
from star_dro.utils import load_config, write_json


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ["model", "train", "valid", "test", "output"]:
        parser.add_argument("--"+field, required=True)
    parser.add_argument("--ontology")
    parser.add_argument("--allow-split-overlap", action="store_true", help="SMOKE TEST ONLY: retain supplied overlapping splits and record the exception")
    parser.add_argument("--revision")
    parser.add_argument("--config", default=str(ROOT/"configs/reference.json"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    add_vllm_arguments(parser)
    add_scoring_arguments(parser)
    args = parser.parse_args()
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",")]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be distinct integers")
        validate_prediction_options(args)
        require_vllm()
        cfg = load_config(args.config)
        if args.max_length < 2 or args.max_new_tokens < 1:
            raise ValueError("positive length/generation limits required")
        output = Path(args.output)
        if output.exists():
            raise ValueError("experiment output already exists; choose a new directory")
        # Validate all splits before starting any model run.
        validation = ["validate-data", "--train", args.train, "--valid", args.valid, "--test", args.test,
                      "--config", args.config]
        if args.allow_split_overlap:
            validation += ["--allow-split-overlap"]
        if args.ontology:
            validation += ["--ontology", args.ontology]
        validation_report = main(validation)
        output.mkdir(parents=True)
        write_json(output/"data_validation.json", validation_report)
        write_json(output/"experiment.json", {"model": args.model, "revision": args.revision,
            "seeds": seeds, "config": cfg, "precision": args.precision, "device": args.device,
            "max_length": args.max_length, "max_new_tokens": args.max_new_tokens, "scorer": {"name": "ontology", "span_threshold": args.span_threshold, "unknown_labels": args.unknown_labels, "fuzzy_labels": not args.exact_labels}, "allow_split_overlap": args.allow_split_overlap,
            "evaluation_scope": "smoke_test_with_overlap_allowed" if args.allow_split_overlap else "disjoint_splits",
            "inference": {"backend": "vllm", "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization, "batch_size": args.batch_size,
                "stop": args.stop, "enforce_eager": args.enforce_eager,
                "merge_device": args.merge_device, "merge_temp_dir": args.merge_temp_dir,
                "merged_weights": "temporary"}})
        seed_results = []
        for seed in seeds:
            run_dir = output/f"seed-{seed}"
            command = ["train", "--model", args.model, "--train", args.train, "--valid", args.valid,
                       "--config", args.config, "--output", str(run_dir), "--seed", str(seed),
                       "--device", args.device, "--precision", args.precision, "--max-length", str(args.max_length)]
            if args.allow_split_overlap:
                command += ["--allow-split-overlap"]
            if args.ontology:
                command += ["--ontology", args.ontology]
            if args.revision:
                command += ["--revision", args.revision]
            run_cli(command)
            # Each model stage runs in a new process; training CUDA state is gone
            # before the selected adapter is merged and vLLM initializes.
            command = ["predict", "--checkpoint", str(run_dir), "--input", args.test,
                       "--output", str(run_dir/"predictions.jsonl"), "--device", "cuda",
                       "--precision", args.precision, "--max-length", str(args.max_length),
                       "--max-new-tokens", str(args.max_new_tokens), "--seed", str(seed),
                       "--tensor-parallel-size", str(args.tensor_parallel_size),
                       "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                       "--batch-size", str(args.batch_size), "--stop", args.stop,
                       "--merge-device", args.merge_device]
            if args.merge_temp_dir:
                command += ["--merge-temp-dir", args.merge_temp_dir]
            if args.enforce_eager:
                command += ["--enforce-eager"]
            run_cli(command)
            command = ["evaluate", "--gold", args.test, "--predictions", str(run_dir/"predictions.jsonl"),
                       "--output", str(run_dir/"test-metrics.json"), "--span-threshold", str(args.span_threshold),
                       "--unknown-labels", args.unknown_labels] + (["--exact-labels"] if args.exact_labels else [])
            if args.ontology:
                command += ["--ontology", args.ontology]
            main(command)
            result = json.loads((run_dir/"test-metrics.json").read_text(encoding="utf8"))
            if args.allow_split_overlap:
                result["evaluation_scope"] = "smoke_test_with_overlap_allowed"
                result["split_overlap"] = validation_report["split_overlap"]
                write_json(run_dir/"test-metrics.json", result)
            seed_results.append({"seed": seed, "metrics": result})
        summary = {}
        for metric in ["code", "subcode", "code_subcode", "span"]:
            summary[metric] = {}
            for measure in ["precision", "recall", "f1"]:
                values = [row["metrics"]["metrics"][metric][measure] for row in seed_results]
                summary[metric][measure] = {"mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values)>1 else None, "per_seed": values}
        write_json(output/"summary.json", {"seeds": seeds, "metrics": summary,
                                          "evaluation_scope": "smoke_test_with_overlap_allowed" if args.allow_split_overlap else "disjoint_splits",
                                          "invalid_output_rates": [r["metrics"]["invalid_output_rate"] for r in seed_results]})
        print(json.dumps({"output": str(output), "completed_seeds": seeds}))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"STaR-DRO experiment error: {exc}\n")


if __name__ == "__main__":
    run()
