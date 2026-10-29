"""Evaluation entry point: ontology-driven scoring plus parsing and grounding diagnostics."""
import json
from .data import validate_annotations
from . import scoring


def parse_prediction(record):
    """Strict parse used for the invalid-output diagnostic and saved prediction records."""
    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
        raise ValueError("each prediction needs an id string")
    if "prediction" in record:
        if not isinstance(record["prediction"], str):
            raise ValueError("prediction must be a string")
        try:
            value = json.loads(record["prediction"])
            if isinstance(value, dict) and set(value) == {"results"}:
                value = value["results"]
            validate_annotations(value)
            return value, None
        except (ValueError, TypeError) as exc:
            return [], str(exc)
    if "annotations" not in record:
        raise ValueError("prediction record needs prediction text or annotations")
    validate_annotations(record["annotations"])
    return record["annotations"], None


def evaluate(gold, predictions, threshold=0.6, ontology=None, unknown_labels="ignore", fuzzy=True):
    """Score Code, Sub-code, joint Code/Sub-code and Span P/R/F1.

    gold: records with id and annotations (and optional source_text).
    predictions: records with id and raw `prediction` text, or parsed `annotations`.
    ontology: {Code: [Sub-code, ...]} defining the label inventory. When omitted,
      the inventory is derived from the gold annotations and reported as such.
    """
    pred_by_id = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("each prediction needs an id string")
        if row["id"] in pred_by_id:
            raise ValueError("duplicate prediction id")
        pred_by_id[row["id"]] = row
    gold_ids = [row["id"] for row in gold]
    if len(set(gold_ids)) != len(gold_ids) or set(gold_ids) != set(pred_by_id):
        raise ValueError("gold/prediction ids must match exactly, with no duplicates")
    source = "ontology" if ontology is not None else "gold_annotations"
    inventory = scoring.LabelInventory(ontology if ontology is not None else
                                       scoring.ontology_from_annotations(gold), fuzzy=fuzzy)
    pairs = []
    invalid_json = annotated = grounded = total_spans = 0
    for row in gold:
        record = pred_by_id[row["id"]]
        parsed, error = parse_prediction(record)
        invalid_json += int(error is not None)
        # Raw text is authoritative; cached annotations never override it.
        pairs.append((row["annotations"], record["prediction"] if "prediction" in record else record["annotations"]))
        annotated += len(parsed)
        if "source_text" in row:
            spans = [ann["Span"] for ann in parsed]
            grounded += sum(span in row["source_text"] for span in spans)
            total_spans += len(spans)
    metrics, diagnostics = scoring.score(pairs, inventory, threshold, unknown_labels)
    return {"records": len(gold), "metrics": metrics,
            "invalid_output_records": invalid_json,
            "invalid_output_rate": invalid_json / len(gold) if gold else 0.0,
            "predicted_annotations": annotated,
            "grounded_span_fraction": grounded / total_spans if total_spans else None,
            **diagnostics,
            "scorer": "ontology", "label_inventory_source": source,
            "label_inventory": {"codes": len(inventory.codes), "subcodes": len(inventory.subcodes)},
            "label_matching": ("normalized_fuzzy" if fuzzy else "normalized"),
            "fuzzy_backend": scoring.backend() if fuzzy else None,
            "unknown_label_policy": unknown_labels, "span_threshold": threshold,
            "span_matching": "token_containment_or_jaccard_greedy_one_to_one"}
