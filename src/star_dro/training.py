"""Single-device SFT, STaR-DRO and Standard DRO with epoch-boundary resumption."""
import json
import math
import os
from pathlib import Path
import random
from .core import Config, StarDRO
from .standard_dro import StandardDRO
from .data import check_disjoint, collate, encode_record, load_ontology, read_records
from .grouping import GroupInventory
from .utils import append_jsonl, environment, file_hash, load_config, write_json


def dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("install requirements.txt before model training or prediction") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def validate_device(torch, device, precision):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu --precision float32 for CPU execution")
    if device == "cpu" and precision != "float32":
        raise ValueError("CPU runs require --precision float32")
    if device.startswith("cuda") and precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("device does not support bfloat16; select --precision float32")
    return torch.bfloat16 if precision == "bfloat16" else torch.float32


def validation_metrics(model, records, groups, tokenizer, device, granularity="annotation"):
    import torch
    from .torch_loss import annotation_loss
    total_ce, total_tokens = 0.0, 0
    sums, mass, counts = [0.0]*len(groups), [0.0]*len(groups), [0]*len(groups)
    model.eval()
    with torch.no_grad():
        for row in records:
            batch = collate([row], tokenizer.pad_token_id, device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss, _ = annotation_loss(logits, batch["labels"], batch["annotation_masks"], batch["annotation_groups"])
            n = int((batch["labels"][:, 1:] != -100).sum())
            total_ce += float(loss)*n
            total_tokens += n
            ce = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                batch["labels"][:, 1:].reshape(-1), ignore_index=-100, reduction="none").reshape(1, -1)
            if granularity == "annotation":
                for j, g in enumerate(row["groups"]):
                    sums[g] += float(ce[0, batch["annotation_masks"][0, j]].mean())
                    mass[g] += 1
                    counts[g] += 1
            else:
                members = row["memberships"]
                for g in members:
                    sums[g] += float(loss) / len(members)
                    mass[g] += 1 / len(members)
                    counts[g] += 1
    means = [sums[j]/mass[j] if mass[j] else None for j in range(len(groups))]
    observed = [x for x in means if x is not None]
    return {"completion_ce": total_ce/total_tokens,
            "group_mean_ce": dict(zip(groups, means)), "group_counts": dict(zip(groups, counts)),
            "group_effective_mass": dict(zip(groups, mass)), "granularity": granularity,
            "macro_group_ce": sum(observed)/len(observed) if observed else None,
            "worst_group_ce": max(observed) if observed else None}


def train(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("this training entry point supports one process/device")
    if args.max_length < 2:
        raise ValueError("max-length must be at least 2")
    cfg = load_config(args.config)
    ontology = load_ontology(args.ontology)
    raw_train = read_records(args.train, ontology=ontology)
    raw_valid = read_records(args.valid, ontology=ontology)
    overlap = check_disjoint(raw_train, raw_valid, allow_overlap=args.allow_split_overlap)
    inventory = GroupInventory.fit(raw_train, cfg["grouping"]["groupby"], cfg["grouping"]["min_count"])
    groups = list(inventory.names)
    granularity = cfg["grouping"]["granularity"]
    for row in raw_valid:
        inventory.encode(row["annotations"])
    torch, AutoModel, AutoTokenizer = dependencies()
    from transformers import get_cosine_schedule_with_warmup
    from .torch_loss import structured_loss
    dtype = validate_device(torch, args.device, args.precision)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out = Path(args.output)
    resume = Path(args.resume) if args.resume else None
    checkpoint_state = None
    if resume:
        if resume.resolve().parent != out.resolve() or not (resume/"training_state.pt").is_file():
            raise ValueError("resume must identify a complete epoch checkpoint in the output directory")
        # Load only trusted, locally produced training checkpoints (contains RNG state).
        checkpoint_state = torch.load(resume/"training_state.pt", map_location="cpu", weights_only=False)
    elif out.exists():
        raise ValueError("output directory already exists; choose a new path or resume a checkpoint")
    hashes = {"train": file_hash(args.train), "valid": file_hash(args.valid),
              "ontology": file_hash(args.ontology) if args.ontology else None}
    identity = {"model": args.model, "revision": args.revision, "seed": args.seed,
                "max_length": args.max_length, "precision": args.precision,
                "config": cfg, "groups": groups, "group_inventory": inventory.to_dict(), "data_sha256": hashes, "allow_split_overlap": args.allow_split_overlap}
    if resume:
        previous = json.loads((out/"run.json").read_text(encoding="utf8"))
        if previous["identity"] != identity:
            raise ValueError("resume model, seed, config, precision, groups and data hashes must match run.json")
        if checkpoint_state["epoch"] >= cfg["training"]["epochs"]:
            raise ValueError("checkpoint already completed the configured training run")
        if not resume.name.startswith("epoch-"):
            raise ValueError("resume requires an epoch checkpoint")
        later = [p for p in out.glob("epoch-*") if p.is_dir() and p.name > resume.name and (p/"training_state.pt").exists()]
        if later:
            raise ValueError("resume from the latest completed epoch to avoid overwriting later checkpoints")
    tok = AutoTokenizer.from_pretrained(str(resume) if resume else args.model,
                                       **({} if resume else {"revision": args.revision}), use_fast=True)
    if not tok.is_fast or tok.eos_token_id is None:
        raise ValueError("a fast tokenizer with EOS token is required")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    train_records = [encode_record(row, tok, inventory, args.max_length) for row in raw_train]
    valid_records = [encode_record(row, tok, inventory, args.max_length) for row in raw_valid]
    lora = cfg["lora"]
    if resume and not lora["enabled"]:
        model = AutoModel.from_pretrained(resume, torch_dtype=dtype)
    else:
        model = AutoModel.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype)
    if lora["enabled"]:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA requires peft; install requirements.txt") from exc
        if resume:
            model = PeftModel.from_pretrained(model, resume, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=lora["rank"],
                lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"]))
    model.to(args.device)
    model.config.use_cache = False
    t = cfg["training"]
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                           lr=t["learning_rate"], weight_decay=t["weight_decay"])
    steps_per_epoch = math.ceil(len(train_records)/t["batch_size"])
    total_steps = steps_per_epoch*t["epochs"]
    scheduler = get_cosine_schedule_with_warmup(opt, math.ceil(t["warmup_ratio"]*total_steps), total_steps)
    standard = t["objective"] == "standard_dro"
    state_filename = "standard_dro.json" if standard else "star_dro.json"
    adversary = (StandardDRO(len(groups), cfg["standard_dro"]["eta"]) if standard else
                 StarDRO(Config(num_groups=len(groups), **cfg["star_dro"])))
    step, start_epoch, best = 0, 1, float("inf")
    if resume:
        opt.load_state_dict(checkpoint_state["optimizer"])
        scheduler.load_state_dict(checkpoint_state["scheduler"])
        random.setstate(checkpoint_state["python_rng"])
        torch.set_rng_state(checkpoint_state["torch_rng"])
        if torch.cuda.is_available() and checkpoint_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint_state["cuda_rng"])
        saved_adversary = json.loads((resume/state_filename).read_text(encoding="utf8"))
        adversary = StandardDRO.from_state_dict(saved_adversary) if standard else StarDRO.from_state_dict(saved_adversary)
        step, start_epoch, best = checkpoint_state["step"], checkpoint_state["epoch"]+1, checkpoint_state["best_ce"]
        # Keep only records from completed epochs when recovering from partial work.
        for filename in ["train.jsonl", "validation.jsonl"]:
            path = out/filename
            if path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
                path.write_text(''.join(json.dumps(row)+"\n" for row in rows if row["epoch"] < start_epoch), encoding="utf8")
    else:
        out.mkdir(parents=True)
        write_json(out/"run.json", {"identity": identity, "environment": environment(),
            "model_commit": getattr(model.config, "_commit_hash", None),
            "tokenizer_commit": tok.init_kwargs.get("_commit_hash"),
            "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
            "total_parameters": sum(v.numel() for v in model.parameters()),
            "device": args.device, "checkpoint_selection": cfg["training"]["checkpoint_selection"],
            "allow_split_overlap": args.allow_split_overlap, "split_overlap": overlap,
            "evaluation_scope": "smoke_test_with_overlap_allowed" if args.allow_split_overlap else "disjoint_splits"})
    for epoch in range(start_epoch, t["epochs"]+1):
        order = list(range(len(train_records)))
        random.shuffle(order)
        model.train()
        opt.zero_grad(set_to_none=True)
        for start in range(0, len(order), t["batch_size"]):
            batch = collate([train_records[i] for i in order[start:start+t["batch_size"]]], tok.pad_token_id, args.device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            active = t["objective"] in {"stardro", "standard_dro"} and epoch >= t["dro_start_epoch"]
            # Epoch 1 and the SFT baseline use the same neutral completion-token
            # objective; sample normalization begins with robust sample mode.
            loss, _ = structured_loss(logits, batch, granularity if active else "annotation",
                                      adversary if active else None, step/max(total_steps-1, 1))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss; training stopped before the optimizer update")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
            applied_lr = opt.param_groups[0]["lr"]
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            append_jsonl(out/"train.jsonl", {"step": step, "epoch": epoch, "loss": float(loss.detach()),
                "learning_rate": applied_lr, "star_dro_active": active and not standard,
                "standard_dro_active": active and standard,
                "q": adversary.q, "ema": adversary.ema,
                "star_dro": adversary.last_update if active and not standard else None,
                "standard_dro": adversary.last_update if active and standard else None})
        metrics = {"epoch": epoch, **validation_metrics(model, valid_records, groups, tok, args.device, granularity)}
        append_jsonl(out/"validation.jsonl", metrics)
        improved = metrics["completion_ce"] < best
        if improved:
            best = metrics["completion_ce"]
        # "last" reports the final epoch of the continuous run (the manuscript protocol);
        # "validation_ce" reports the epoch with the lowest validation completion CE.
        # With validation_ce, a robust run can select an epoch before DRO activation.
        selected = improved if t["checkpoint_selection"] == "validation_ce" else True
        checkpoint = out/f"epoch-{epoch:03d}"
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)
        write_json(checkpoint/state_filename, adversary.state_dict())
        write_json(checkpoint/"run.json", json.loads((out/"run.json").read_text(encoding="utf8")))
        torch.save({"epoch": epoch, "step": step, "best_ce": best, "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   checkpoint/"training_state.pt.tmp")
        (checkpoint/"training_state.pt.tmp").replace(checkpoint/"training_state.pt")
        if selected:
            write_json(out/"selection.json", {"checkpoint": checkpoint.name, "epoch": epoch,
                                              "criterion": t["checkpoint_selection"],
                                              "validation_completion_ce": metrics["completion_ce"],
                                              "robust_objective_active": (t["objective"] != "sft" and epoch >= t["dro_start_epoch"])})
        print(json.dumps(metrics))
