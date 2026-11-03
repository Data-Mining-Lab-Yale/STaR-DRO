"""Selected-checkpoint merging and deterministic vLLM inference."""
import json
from pathlib import Path
from .data import read_records, render_prompt
from .metrics import parse_prediction
from .utils import environment, file_hash, write_json


def resolve_checkpoint(path):
    """Resolve a run's selection.json or an explicit saved epoch directory."""
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise ValueError("checkpoint must be a local saved STaR-DRO epoch or run directory")
    if (checkpoint/"selection.json").is_file():
        selected = json.loads((checkpoint/"selection.json").read_text(encoding="utf8"))["checkpoint"]
        if not isinstance(selected, str) or not selected:
            raise ValueError("selection.json checkpoint must be a nonempty directory name")
        candidate = (checkpoint/selected).resolve()
        if candidate.parent != checkpoint.resolve() or not candidate.is_dir():
            raise ValueError("selection.json must identify a saved checkpoint in the run directory")
        checkpoint = candidate
    return checkpoint


def require_vllm():
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        raise ImportError("vLLM testing requires requirements.txt or star-dro[inference]; install a CUDA-compatible build")


def validate_prediction_options(args):
    import math
    if args.max_new_tokens < 1 or args.max_length < 2:
        raise ValueError("positive generation budget and max-length >= 2 required")
    if args.max_new_tokens >= args.max_length:
        raise ValueError("max-new-tokens must leave room for a nonempty prompt")
    if args.device != "cuda":
        raise ValueError("vLLM prediction uses --device cuda; select GPUs with CUDA_VISIBLE_DEVICES")
    if args.tensor_parallel_size < 1 or args.batch_size < 1:
        raise ValueError("tensor-parallel-size and batch-size must be positive")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0,1]")
    if args.merge_device != "cpu" and not args.merge_device.startswith("cuda"):
        raise ValueError("merge-device must be cpu or cuda[:INDEX]")
    if not isinstance(args.stop, str) or not args.stop:
        raise ValueError("stop must be a nonempty string")


def _predict_vllm(args, checkpoint, model_path, rows, merge_metadata):
    # All entry points have a __main__ guard. Never start this from python stdin.
    import os
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for this vLLM prediction entry point")
    if args.tensor_parallel_size > torch.cuda.device_count():
        raise ValueError("tensor-parallel-size exceeds the visible CUDA device count")
    if args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("device does not support bfloat16; select --precision float32")
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    # Preserve model-specific end-of-turn/EOS IDs, including Llama's multiple EOS.
    for filename in ("config.json", "generation_config.json"):
        path = model_path / filename
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf8")).get("eos_token_id")
            eos_ids.update(value if isinstance(value, list) else ([] if value is None else [value]))
    sampling_options = {"n": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                        "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                        "repetition_penalty": 1.0, "seed": args.seed, "stop": [args.stop],
                        "stop_token_ids": sorted(eos_ids), "ignore_eos": False,
                        "max_tokens": args.max_new_tokens, "skip_special_tokens": True,
                        "include_stop_str_in_output": False}
    sampling = SamplingParams(**sampling_options)
    engine = None
    output = Path(args.output)
    invalid = 0
    with output.open("x", encoding="utf8") as handle:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            prompts = []
            for row in batch:
                tokens = tokenizer(render_prompt(row, tokenizer), add_special_tokens=False)["input_ids"]
                if not tokens or len(tokens) + args.max_new_tokens > args.max_length:
                    raise ValueError(f"record {row['id']}: prompt plus generation budget exceeds max-length or prompt is empty")
                # Pass token IDs to prevent vLLM adding BOS/chat formatting twice.
                prompts.append({"prompt_token_ids": tokens})
            if engine is None:
                engine = LLM(model=str(model_path), tokenizer=str(model_path), dtype=args.precision,
                             tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_length,
                             gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
                             enforce_eager=args.enforce_eager, generation_config="vllm")
            responses = engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
            if len(responses) != len(batch):
                raise RuntimeError("vLLM returned a different number of responses than requested")
            for row, response in zip(batch, responses):
                if len(response.outputs) != 1:
                    raise RuntimeError("vLLM must return exactly one completion per record")
                completion = response.outputs[0]
                if completion.finish_reason not in {"stop", "length"}:
                    raise RuntimeError(f"vLLM request did not complete: {completion.finish_reason}")
                text = completion.text.strip()
                annotations, error = parse_prediction({"id": row["id"], "prediction": text})
                invalid += int(error is not None)
                handle.write(json.dumps({"id": row["id"], "prediction": text, "annotations": annotations,
                                         "parse_error": error}, ensure_ascii=False) + "\n")
                handle.flush()
    write_json(str(output) + ".meta.json", {"records": len(rows), "invalid_outputs": invalid,
        "input_sha256": file_hash(args.input), "checkpoint": checkpoint.name, "backend": "vllm",
        "merge": merge_metadata, "do_sample": False, "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length, "precision": args.precision, "seed": args.seed,
        "batch_size": args.batch_size, "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization, "enforce_eager": args.enforce_eager,
        "generation_config": "vllm", "sampling_params": sampling_options, "environment": environment()}, exclusive=True)
    print(json.dumps({"records": len(rows), "invalid_outputs": invalid, "backend": "vllm", "output": str(output)}))


def predict(args):
    import tempfile
    from .execution import run_cli
    validate_prediction_options(args)
    output = Path(args.output)
    if output.exists() or Path(str(output) + ".meta.json").exists():
        raise ValueError("prediction output or metadata exists; choose a new output path")
    checkpoint = resolve_checkpoint(args.checkpoint)
    rows = read_records(args.input, require_annotations=False)
    require_vllm()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not (checkpoint / "adapter_config.json").is_file():
        metadata_path = checkpoint / "merge.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf8")) if metadata_path.is_file() else None
        return _predict_vllm(args, checkpoint, checkpoint, rows, metadata)
    # The merge process exits completely before vLLM starts loading weights.
    # Default to the output filesystem, rather than potentially small /tmp.
    temporary_parent = Path(args.merge_temp_dir) if args.merge_temp_dir else output.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="star-dro-merged-", dir=temporary_parent) as temporary:
        model_path = Path(temporary) / "model"
        command = ["merge", "--checkpoint", str(checkpoint), "--output", str(model_path),
                   "--device", args.merge_device, "--precision", args.precision]
        if args.base_model:
            command += ["--base-model", args.base_model]
        run_cli(command)
        metadata = json.loads((model_path / "merge.json").read_text(encoding="utf8"))
        _predict_vllm(args, checkpoint, model_path, rows, metadata)
