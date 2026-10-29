"""Assemble a task instruction and instance context into model-visible records."""
import json
from pathlib import Path
import re

from .data import validate_record


def read_instructions(path):
    """Read a completed Markdown template's single text block, or plain text."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig").strip()
    if path.suffix.lower() == ".md":
        blocks = re.findall(r"^```text\s*\n(.*?)^```\s*$", text, re.M | re.S)
        if len(blocks) != 1:
            raise ValueError("instruction Markdown must contain exactly one fenced text block")
        text = blocks[0].strip()
    if not text:
        raise ValueError("task instructions must be nonempty")
    slots = re.findall(r"\[(?:Domain expert|Precision-oriented|Task definition|Authoritative label|"
                       r"Boundary condition|Expose task metadata|Machine-parseable|Verification checklist|"
                       r"INSERT)[^\]\n]*\]", text)
    if slots:
        raise ValueError("complete the task-specific placeholders in your instruction template before building records")
    return text


def build_record(raw, instructions, ontology=None, output_format="results"):
    """Keep gold separate while placing the full task and instance in one user turn."""
    if not isinstance(raw, dict):
        raise ValueError("each source record must be an object")
    allowed = {"id", "target_text", "context", "metadata", "annotations"}
    if set(raw) - allowed:
        raise ValueError(f"unsupported source fields: {sorted(set(raw) - allowed)}")
    target, context = raw.get("target_text"), raw.get("context", "")
    metadata = raw.get("metadata", {})
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target_text must be a nonempty string")
    if not isinstance(context, str) or not isinstance(metadata, dict):
        raise ValueError("context must be a string and metadata must be an object")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("task instructions must be nonempty")
    metadata_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
    content = (instructions.strip() + "\n\nMETADATA:\n" + metadata_text
               + "\n\nCONTEXT (for interpretation):\n" + context
               + "\n\nTARGET TEXT (evidence source):\n" + target)
    row = {"id": raw.get("id"), "messages": [{"role": "user", "content": content}],
           "source_text": target, "output_format": output_format}
    if "annotations" in raw:
        row["annotations"] = raw["annotations"]
    return validate_record(row, ontology=ontology)
