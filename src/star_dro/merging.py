"""Merge a selected LoRA adapter into a standalone model for vLLM."""
import json
from pathlib import Path
from .inference import resolve_checkpoint
from .training import dependencies
from .utils import environment, write_json


def merge(args):
    checkpoint = resolve_checkpoint(args.checkpoint)
    output = Path(args.output)
    if output.exists():
        raise ValueError("merge output exists; choose a new directory")
    if output.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("merge output must be outside the source checkpoint")
    if not (checkpoint / "adapter_config.json").is_file():
        raise ValueError("merge requires a LoRA adapter checkpoint; full models can be passed directly to predict")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        raise ValueError("merge device must be cpu or cuda[:INDEX]")
    torch, AutoModel, AutoTokenizer = dependencies()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable for merging; use --device cpu")
    if args.precision == "bfloat16" and args.device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ValueError("merge device does not support bfloat16")
    from peft import PeftModel
    run = json.loads((checkpoint / "run.json").read_text(encoding="utf8"))
    base = args.base_model or run["identity"]["model"]
    revision = run.get("model_commit") or run["identity"].get("revision")
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float32
    # CPU is the default merge location, including for bf16 exports. No quantization
    # is applied: this avoids merging into quantized base weights.
    model = AutoModel.from_pretrained(base, revision=revision, torch_dtype=dtype,
                                      device_map={"": args.device}, low_cpu_mem_usage=True)
    if getattr(model, "is_quantized", False) is True:
        raise ValueError("merge requires the original unquantized base model")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    tokenizer.save_pretrained(output)
    metadata = {"checkpoint": checkpoint.name, "base_model": base, "revision": revision,
                "precision": args.precision, "device": args.device, "safe_merge": True,
                "format": "standalone_safetensors", "environment": environment()}
    write_json(output / "merge.json", metadata, exclusive=True)
    print(json.dumps({"checkpoint": checkpoint.name, "merged_model": str(output)}))
