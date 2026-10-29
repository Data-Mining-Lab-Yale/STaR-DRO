"""Configuration and run-artifact utilities for STaR-DRO."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from .core import Config
from .grouping import canonical_groupby

DEFAULTS = {
    "grouping": {"groupby": "code", "granularity": "annotation", "min_count": 1},
    "star_dro": {"alpha": 1.2, "eta": 0.0006, "rho_start": 0.3, "rho_end": 0.03, "upper": 15.0, "gamma": 0.75,
                 "hardness_scale": "count"},
    "standard_dro": {"eta": 0.1},
    "training": {"objective": "stardro", "epochs": 3, "dro_start_epoch": 2, "batch_size": 3, "learning_rate": 0.0002,
                 "weight_decay": 0.01, "warmup_ratio": 0.03, "max_grad_norm": 1.0, "gradient_checkpointing": True,
                 "checkpoint_selection": "last"},
    "lora": {"enabled": True, "rank": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
}


def load_config(path=None):
    cfg = json.loads(json.dumps(DEFAULTS))
    supplied = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
    if not isinstance(supplied, dict) or set(supplied)-set(cfg):
        raise ValueError("config sections must be grouping, star_dro, standard_dro, training, lora")
    for section, values in supplied.items():
        if not isinstance(values, dict) or set(values)-set(cfg[section]):
            raise ValueError(f"unknown configuration key in {section}")
        cfg[section].update(values)
    Config(num_groups=1, **cfg["star_dro"])
    grouping = cfg["grouping"]
    grouping["groupby"] = canonical_groupby(grouping["groupby"])
    if grouping["granularity"] not in {"sample", "annotation"}:
        raise ValueError("grouping.granularity must be sample or annotation")
    if isinstance(grouping["min_count"], bool) or not isinstance(grouping["min_count"], int) or grouping["min_count"] < 1:
        raise ValueError("grouping.min_count must be a positive integer")
    t, lora = cfg["training"], cfg["lora"]
    if t["objective"] not in {"stardro", "standard_dro", "sft"}:
        raise ValueError("training.objective must be stardro, standard_dro or sft")
    from .standard_dro import StandardDRO
    StandardDRO(1, cfg["standard_dro"]["eta"])
    if t["objective"] == "standard_dro" and grouping["granularity"] != "annotation":
        raise ValueError("the notebook144 Standard DRO baseline requires annotation granularity")
    for key in ["epochs", "dro_start_epoch", "batch_size"]:
        if isinstance(t[key], bool) or not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if t["checkpoint_selection"] not in {"last", "validation_ce"}:
        raise ValueError("training.checkpoint_selection must be last or validation_ce")
    if t["dro_start_epoch"] > t["epochs"]:
        raise ValueError("dro_start_epoch must be within the training run")
    for key in ["learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"]:
        if isinstance(t[key], bool) or not isinstance(t[key], (int, float)) or not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"invalid training.{key}")
    if t["learning_rate"] <= 0 or t["max_grad_norm"] <= 0 or t["warmup_ratio"] >= 1:
        raise ValueError("positive learning rate/gradient norm and warmup_ratio in [0,1) required")
    if not isinstance(t["gradient_checkpointing"], bool) or not isinstance(lora["enabled"], bool):
        raise ValueError("enabled and gradient_checkpointing must be booleans")
    if not isinstance(lora["rank"], int) or lora["rank"] < 1 or lora["alpha"] <= 0 or not 0 <= lora["dropout"] < 1:
        raise ValueError("invalid LoRA configuration")
    if not isinstance(lora["target_modules"], list) or not lora["target_modules"] or any(not isinstance(x, str) or not x for x in lora["target_modules"]):
        raise ValueError("lora.target_modules must be a nonempty list of module names")
    return cfg


def file_hash(path):
    """Hash one file or a saved dataset directory, including relative filenames."""
    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise ValueError("data path does not exist")
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("cannot hash an empty data directory")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    for package in ["star-dro", "torch", "transformers", "peft", "datasets", "pyarrow", "numpy", "tokenizers", "safetensors", "accelerate", "vllm"]:
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
