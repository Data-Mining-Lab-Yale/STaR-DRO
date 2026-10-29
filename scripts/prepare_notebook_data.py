"""Prepare the reference Arrow schemas from authorized JSON/JSONL; never resplit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from star_dro.data import make_arrow_dataset, normalize_record


def prepare(source, output, layout):
    from datasets import Features, Value
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("output exists; choose a new directory")
    if source.suffix.lower() == ".json":
        rows = json.loads(source.read_text(encoding="utf-8-sig"))
        rows = [rows] if isinstance(rows, dict) else rows
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        with source.open(encoding="utf-8-sig") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError("input must be JSON or JSONL")
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("input must contain record objects")
    if layout not in {"entries", "query_answer"}:
        raise ValueError("layout must be entries or query_answer")
    exported, identities = [], []
    include_id = layout == "entries" or any("id" in row for row in rows)
    for i, row in enumerate(rows, 1):
        if layout == "entries" and "entries" not in row:
            raise ValueError(f"record {i}: entries layout requires entries")
        if layout == "query_answer":
            if not all(isinstance(row.get(k), str) for k in ["query", "answer", "context", "TO_PAT_YN"]):
                raise ValueError(f"record {i}: query, answer, context and TO_PAT_YN must be strings")
            if row["TO_PAT_YN"] not in {"Y", "N"}:
                raise ValueError(f"record {i}: TO_PAT_YN must be Y or N")
        normalized = normalize_record(row)
        identities.append(normalized["id"])
        if layout == "entries":
            item = {"id": normalized["id"], "entries": row["entries"]}
        else:
            item = {k: row[k] for k in ["query", "answer", "context", "TO_PAT_YN"]}
            if include_id:
                item["id"] = normalized["id"]
        exported.append(item)
    duplicates = sum(n - 1 for n in Counter(identities).values() if n > 1)
    if duplicates:
        raise ValueError(f"{duplicates} repeated resolved IDs; supply stable source IDs and audit repeated inputs")
    if layout == "entries":
        features = Features({"id": Value("string"),
                             "entries": [{"content": Value("string"), "role": Value("string")}]})
    else:
        features = Features({key: Value("string") for key in ["query", "answer", "context", "TO_PAT_YN"]})
        if include_id:
            features["id"] = Value("string")
    make_arrow_dataset(exported, features=features).save_to_disk(str(output))
    return {"records": len(exported), "layout": layout, "columns": list(features), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout", choices=["entries", "query_answer"], required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.input, args.output, args.layout), indent=2))
    except (ValueError, OSError, ImportError, TypeError) as exc:
        parser.exit(2, f"Preparation error: {exc}\n")


if __name__ == "__main__":
    main()
