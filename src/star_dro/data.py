"""STaR-DRO data loading, normalization, chat formatting, and token masks."""
import hashlib
import json
from pathlib import Path
from .grouping import GroupInventory

ANNOTATION_FIELDS = {"Code", "Sub-code", "Span"}


def validate_annotations(value, ontology=None):
    if not isinstance(value, list):
        raise ValueError("annotations must be a JSON list (use [] for no annotations)")
    for index, ann in enumerate(value):
        if not isinstance(ann, dict) or set(ann) != ANNOTATION_FIELDS:
            raise ValueError(f"annotation {index}: require exactly Code, Sub-code, Span")
        if any(not isinstance(x, str) or not x.strip() for x in ann.values()):
            raise ValueError(f"annotation {index}: all values must be nonempty strings")
        if ontology is not None and (ann["Code"] not in ontology or ann["Sub-code"] not in ontology[ann["Code"]]):
            raise ValueError(f"annotation {index}: Code/Sub-code pair is outside the ontology")


def validate_record(row, require_annotations=True, ontology=None):
    if not isinstance(row, dict):
        raise ValueError("each JSONL record must be an object")
    allowed = {"id", "messages", "prompt", "annotations", "source_text", "output_format"}
    if set(row) - allowed:
        raise ValueError(f"unsupported fields: {sorted(set(row)-allowed)}")
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        raise ValueError("id must be a nonempty string")
    if ("messages" in row) == ("prompt" in row):
        raise ValueError("provide exactly one of messages or prompt")
    if "prompt" in row:
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError("prompt must be a nonempty rendered string")
    else:
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty chat history")
        for i, message in enumerate(messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError(f"message {i}: require exactly role and content")
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"message {i}: unsupported role")
            if not isinstance(message["content"], str) or not message["content"].strip():
                raise ValueError(f"message {i}: content must be nonempty")
        if messages[-1]["role"] != "user":
            raise ValueError("the final input message must have role user")
    if row.get("output_format", "list") not in {"list", "results"}:
        raise ValueError("output_format must be list or results")
    if require_annotations and "annotations" not in row:
        raise ValueError("training, validation and gold test records require annotations")
    if "annotations" in row:
        validate_annotations(row["annotations"], ontology)
    if "source_text" in row:
        if not isinstance(row["source_text"], str):
            raise ValueError("source_text must be a string")
        for ann in row.get("annotations", []):
            if ann["Span"] not in row["source_text"]:
                raise ValueError("annotation Span must be an exact substring of source_text")
    return row


def decode_annotations(value):
    """Decode a JSON list or a results-wrapped list without fuzzy repair."""
    if isinstance(value, str):
        value = json.loads(value)
    output_format = "list"
    if isinstance(value, dict) and set(value) == {"results"}:
        value = value["results"]
        output_format = "results"
    if not isinstance(value, list):
        raise ValueError("answer must be a JSON list or an object containing only results")
    annotations = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("answer annotations must be objects")
        item = dict(item)
        if "Subcode" in item and "Sub-code" not in item:
            item["Sub-code"] = item.pop("Subcode")
        annotations.append(item)
    validate_annotations(annotations)
    return annotations, output_format


def normalize_record(row, require_annotations=True, ontology=None):
    """Accept canonical records or existing entries/query-answer exports."""
    if not isinstance(row, dict):
        raise ValueError("each data record must be an object")
    row = {key: value for key, value in row.items() if value is not None}
    if "entries" in row:
        if "messages" in row or "prompt" in row or "query" in row:
            raise ValueError("entries cannot be combined with another model-input field")
        entries = row["entries"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("entries must be a nonempty list of role/content objects")
        messages = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
                raise ValueError("each entry needs role and string content")
            role = "user" if entry.get("role") == "human" else entry.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError("entry role must be human, user, system, or assistant")
            messages.append({"role": role, "content": entry["content"]})
        answer = None
        if messages[-1]["role"] == "assistant":
            answer = messages.pop()["content"]
        converted = {"messages": messages}
        if answer is not None:
            converted["annotations"], converted["output_format"] = decode_annotations(answer)
    elif "query" in row:
        if "messages" in row or "prompt" in row:
            raise ValueError("query cannot be combined with another model-input field")
        if not isinstance(row["query"], str):
            raise ValueError("query must be a string")
        converted = {"messages": [{"role": "user", "content": row["query"]}]}
        if "answer" in row:
            converted["annotations"], converted["output_format"] = decode_annotations(row["answer"])
    else:
        return validate_record(row, require_annotations, ontology)
    # Legacy numeric IDs are preserved as strings; missing IDs derive from inputs.
    converted["id"] = str(row["id"]) if "id" in row else "input-" + input_fingerprint(converted)
    if "source_text" in row:
        converted["source_text"] = row["source_text"]
    return validate_record(converted, require_annotations, ontology)


def read_records(path, require_annotations=True, ontology=None):
    """Load JSONL, a JSON record/list, a saved HF Dataset, or HF Arrow stream."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data input not found: {path}")
    if path.is_dir() or path.suffix.lower() == ".arrow":
        try:
            from datasets import Dataset, DatasetDict, load_from_disk
        except ImportError as exc:
            raise ImportError("Arrow inputs require datasets; install requirements.txt or star-dro[data]") from exc
        data = load_from_disk(str(path)) if path.is_dir() else Dataset.from_file(str(path))
        if isinstance(data, DatasetDict):
            raise ValueError("select one saved split directory, such as DATASET_DIRECTORY/train")
        raw = enumerate(data, 1)
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("JSON input must be one record or an array of records")
        raw = enumerate(value, 1)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        def lines():
            with path.open(encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        try:
                            yield line_number, json.loads(line)
                        except ValueError as exc:
                            raise ValueError(f"{path.name}:{line_number}: invalid JSON") from exc
        raw = lines()
    else:
        raise ValueError("input must be .jsonl, .ndjson, .json, .arrow, or a saved Dataset directory")
    rows, ids = [], set()
    for index, value in raw:
        try:
            row = normalize_record(value, require_annotations, ontology)
            if row["id"] in ids:
                raise ValueError("duplicate id within split")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path.name}:record {index}: {exc}") from exc
        ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: empty dataset")
    return rows


def read_jsonl(path, require_annotations=True, ontology=None):
    """Compatibility alias; format detection is handled by read_records."""
    return read_records(path, require_annotations, ontology)


def make_arrow_dataset(rows, features=None):
    """Build an HF Dataset with a deterministic, content-derived fingerprint.

    Hash canonical JSON, not interpreter-specific pickled Arrow objects. This
    also avoids datasets/dill fingerprint failures on some Python 3.14 builds.
    """
    import pyarrow as pa
    from datasets import Dataset, DatasetInfo
    fingerprint = hashlib.sha256(json.dumps({"rows": rows, "features": features.to_dict() if features is not None else None},
                                            ensure_ascii=False, sort_keys=True,
                                            allow_nan=False, separators=(",", ":")).encode("utf8")).hexdigest()
    table = pa.Table.from_pylist(rows, schema=features.arrow_schema if features is not None else None)
    return Dataset(table, info=DatasetInfo(features=features) if features is not None else None,
                   fingerprint=fingerprint)


def write_records(rows, path, storage="jsonl", input_only=False, layout="canonical"):
    """Persist normalized records; Arrow conversion is explicit, never automatic."""
    path = Path(path)
    if path.exists():
        raise ValueError("data output already exists; choose a new path")
    rows = [{key: value for key, value in row.items() if not (input_only and key == "annotations")} for row in rows]
    if layout not in {"canonical", "entries", "query_answer"}:
        raise ValueError("layout must be canonical, entries, or query_answer")
    if layout != "canonical":
        exported = []
        for row in rows:
            if "prompt" in row:
                raise ValueError("notebook layouts require messages; rendered prompts cannot be safely unrendered")
            answer = row.get("annotations")
            if answer is not None:
                if row.get("output_format", "list") == "results":
                    answer = {"results": answer}
                answer = json.dumps(answer, ensure_ascii=False)
            if layout == "entries":
                entries = [{"role": "human" if m["role"] == "user" else m["role"], "content": m["content"]}
                           for m in row["messages"]]
                if answer is not None:
                    entries.append({"role": "assistant", "content": answer})
                exported.append({"id": row["id"], "entries": entries})
            else:
                if len(row["messages"]) != 1 or row["messages"][0]["role"] != "user":
                    raise ValueError("query_answer requires exactly one user turn; put the complete instruction in that turn")
                item = {"id": row["id"], "query": row["messages"][0]["content"]}
                if answer is not None:
                    item["answer"] = answer
                exported.append(item)
        # Notebook layouts have no source_text column. Canonical exports retain
        # it; users can retain the canonical source as the grounding reference.
        rows = exported
    if storage == "arrow":
        try:
            import datasets  # noqa: F401 -- check the optional I/O dependency
        except ImportError as exc:
            raise ImportError("Arrow export requires datasets; install requirements.txt or star-dro[data]") from exc
        columns = sorted({key for row in rows for key in row})
        make_arrow_dataset([{key: row.get(key) for key in columns} for row in rows]).save_to_disk(str(path))
    elif storage == "jsonl":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    else:
        raise ValueError("storage must be jsonl or arrow")


def load_ontology(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not value:
        raise ValueError("ontology must map Code strings to lists of allowed Sub-code strings")
    for code, subcodes in value.items():
        if not isinstance(code, str) or not code.strip() or not isinstance(subcodes, list) or not subcodes:
            raise ValueError("ontology entries require a nonempty Code and Sub-code list")
        if any(not isinstance(s, str) or not s.strip() for s in subcodes) or len(set(subcodes)) != len(subcodes):
            raise ValueError("ontology Sub-codes must be nonempty and unique within each Code")
    return value


def input_fingerprint(row):
    value = row["messages"] if "messages" in row else row["prompt"]
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_overlap_report(splits):
    """Report exact record/input/source overlap without changing any split."""
    report = {}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            a, b = splits[left], splits[right]
            report[f"{left}:{right}"] = {
                "ids": len({row["id"] for row in a} & {row["id"] for row in b}),
                "exact_inputs": len({input_fingerprint(row) for row in a} & {input_fingerprint(row) for row in b}),
                "exact_source_texts": len({row["source_text"] for row in a if row.get("source_text")} &
                                          {row["source_text"] for row in b if row.get("source_text")})}
    return report


def check_disjoint(*splits, allow_overlap=False):
    report = split_overlap_report({str(i): rows for i, rows in enumerate(splits)})
    if not allow_overlap and any(any(counts.values()) for counts in report.values()):
        raise ValueError("splits overlap by id, exact input, or exact source_text")
    return report


def render_prompt(row, tokenizer):
    if "prompt" in row:
        return row["prompt"]
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("messages input requires a tokenizer chat template; use prompt for a rendered input")
    return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)


def encode_record(row, tokenizer, groups, max_length):
    validate_record(row)
    prompt = render_prompt(row, tokenizer)
    parts, spans, gids = [], [], []
    if isinstance(groups, GroupInventory):
        gids, memberships = groups.encode(row["annotations"])
    else:
        if any(ann["Code"] not in groups for ann in row["annotations"]):
            raise ValueError("unseen Code outside the training group inventory")
        gids = [groups.index(ann["Code"]) for ann in row["annotations"]]
        memberships = sorted(set(gids))
    prefix = '{"results": [' if row.get("output_format", "list") == "results" else "["
    suffix = "]}" if row.get("output_format", "list") == "results" else "]"
    cursor = len(prompt) + len(prefix)
    for ann in row["annotations"]:
        if parts:
            cursor += 2
        # Fixed field order makes serialization independent of JSONL key ordering.
        obj = json.dumps({key: ann[key] for key in ["Code", "Sub-code", "Span"]}, ensure_ascii=False)
        spans.append((cursor, cursor + len(obj)))
        parts.append(obj)
        cursor += len(obj)
    completion = prefix + ", ".join(parts) + suffix
    text = prompt + completion
    append_eos = True
    if "messages" in row:
        full = tokenizer.apply_chat_template(row["messages"] + [{"role": "assistant", "content": completion}],
                                             tokenize=False, add_generation_prompt=False)
        if not full.startswith(text):
            raise ValueError("chat template changes the completion prefix; supply a rendered prompt instead")
        append_eos = len(full) == len(text)
        text = full
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), enc["offset_mapping"]
    if len(ids) + int(append_eos) > max_length:
        raise ValueError("record exceeds max_length; split explicitly instead of truncating")
    boundary = len(prompt)
    if any(s < boundary < e for s, e in offsets):
        raise ValueError("token crosses prompt/completion boundary; end prompt with the model's assistant delimiter")
    labels = [tok if s >= boundary and e > s else -100 for tok, (s, e) in zip(ids, offsets)]
    masks = [[s < end and e > start and labels[k] != -100 and k > 0
              for k, (s, e) in enumerate(offsets)] for start, end in spans]
    if append_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires an EOS token")
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
        masks = [mask + [False] for mask in masks]
    masks = [mask[1:] for mask in masks]
    if len(ids) < 2 or not any(label != -100 for label in labels[1:]):
        raise ValueError("record has no predicted completion tokens")
    if any(not any(mask) for mask in masks):
        raise ValueError("annotation has no token coverage")
    return {"input_ids": ids, "labels": labels, "masks": masks, "groups": gids, "memberships": memberships}


def collate(records, pad_id, device):
    import torch
    n = max(len(r["input_ids"]) for r in records)
    a = max(len(r["groups"]) for r in records)
    ids, labels, attn, masks, gids = [], [], [], [], []
    for r in records:
        padding = n - len(r["input_ids"])
        ids.append(r["input_ids"] + [pad_id] * padding)
        labels.append(r["labels"] + [-100] * padding)
        attn.append([1] * len(r["input_ids"]) + [0] * padding)
        masks.append([m + [False] * padding for m in r["masks"]] + [[False] * (n-1)] * (a-len(r["groups"])))
        gids.append(r["groups"] + [-1] * (a-len(r["groups"])))
    b = len(records)
    return {"input_ids": torch.tensor(ids, device=device),
            "labels": torch.tensor(labels, device=device),
            "attention_mask": torch.tensor(attn, device=device),
            "annotation_masks": torch.tensor(masks, dtype=torch.bool, device=device).reshape(b, a, n-1),
            "annotation_groups": torch.tensor(gids, dtype=torch.long, device=device).reshape(b, a),
            "sample_memberships": [r.get("memberships", sorted(set(r["groups"]))) for r in records]}
