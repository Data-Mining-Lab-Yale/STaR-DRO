"""Combine completed task instructions with source records, preserving split assignments."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from star_dro.data import load_ontology, write_records
from star_dro.prompting import build_record, read_instructions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON/JSONL with id, target_text, context, metadata, annotations")
    parser.add_argument("--instructions", required=True, help="completed Markdown task template or plain-text instruction")
    parser.add_argument("--ontology", help="JSON Code-to-Sub-code mapping")
    parser.add_argument("--output", required=True, help="new canonical JSONL file")
    parser.add_argument("--output-format", choices=["results", "list"], default="results")
    args = parser.parse_args()
    try:
        path = Path(args.input)
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            raw = [raw] if isinstance(raw, dict) else raw
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            raw = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            raise ValueError("source input must be JSON or JSONL")
        if not isinstance(raw, list) or not raw:
            raise ValueError("source input must contain records")
        instructions, ontology = read_instructions(args.instructions), load_ontology(args.ontology)
        rows = [build_record(row, instructions, ontology, args.output_format) for row in raw]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate source id within split")
        write_records(rows, args.output)
        print(json.dumps({"records": len(rows), "output": args.output, "output_format": args.output_format}))
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"Dataset preparation error: {exc}\n")


if __name__ == "__main__":
    main()
