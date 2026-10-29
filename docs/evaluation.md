# Evaluation protocol

Our scorer evaluates structured outputs by record ID. It accepts raw generated JSON text or pre-parsed annotation lists and requires exactly matching gold/prediction ID sets, with unique IDs for each record.

## Label inventory

The scorer has no built-in labels. Pass the task ontology, a JSON object mapping each Code to its valid Sub-codes:

```bash
star-dro evaluate --gold TEST_PATH --predictions predictions.jsonl --ontology ontology.json --output metrics.json
```

Without `--ontology`, the inventory is derived from the gold annotations (`label_inventory_source: gold_annotations`). Gold labels must belong to the inventory; otherwise scoring stops with an error. Two labels that become identical after normalization (for example `Build Trust` and `build_trust`) are rejected rather than merged.

## Code, Sub-code and joint-pair metrics

For each record, predicted and gold labels are reduced to sets; true positives, false positives and false negatives are accumulated across records and converted to micro precision, recall and F1. Sub-code F1 scores Sub-code names independently of their parent, so a correct child under the wrong parent counts for Sub-code but not for the joint Code/Sub-code pair metric. `predicted_pairs_outside_hierarchy` counts predicted pairs that the ontology does not allow.

Predicted labels are matched after Unicode NFKC normalization, case folding, and removal of spacing, underscores and punctuation other than "/". Near misses are then mapped to the closest inventory label when the RapidFuzz QRatio similarity is at least 95 (Codes) or 92 (Sub-codes); difflib is used when RapidFuzz is not installed. `--exact-labels` keeps normalization but disables fuzzy mapping.

Predicted labels that match no inventory label are reported in `unknown_predicted_code_labels` and `unknown_predicted_subcode_labels`. By default (`--unknown-labels ignore`, the convention used for the manuscript) they do not enter the label metrics. `--unknown-labels count` scores each distinct unmatched label in a record as a false positive, which penalizes invented labels in precision.

## Span metric

Spans are compared as lowercase whitespace-token sets. A predicted span matches an unmatched gold span when either token set contains the other or their Jaccard similarity is at least `--span-threshold` (default 0.6). Predictions are matched in output order and each gold span can match once; duplicate predictions remain separate candidates. Empty spans never match. Span matching is independent of label correctness.

## Output parsing and diagnostics

Generated text is parsed as JSON, then as a Python literal, and finally by salvaging complete Code/Sub-code/Span objects from malformed text (fences, truncation, trailing stop strings). Both a bare list and an object whose `results` key holds the list are accepted. Raw `prediction` text takes precedence over cached `annotations`.

The strict-format diagnostic is separate: a valid response is a JSON annotation list, or an object whose sole key `results` holds that list, with exactly the fields Code, Sub-code and Span. `invalid_output_records` counts responses failing this check even if the salvage parser recovered annotations from them. If gold records include `source_text`, `grounded_span_fraction` measures exact, case-sensitive containment of parsed predicted spans in that text.

## Loss diagnostics

Training validation reports raw completion-token CE and raw CE by the configured group inventory. Annotation attribution averages annotation-mean CE within groups. Sample attribution uses overlap-corrected sample-mean CE. `group_counts` records contributing observations; `group_effective_mass` records corrected mass. Macro CE averages observed group means; worst-group CE is their maximum. Missing groups are null. Validation never updates the adversary. These are likelihood diagnostics, not F1 or adversarially weighted losses.

## Reporting an experiment

Record the completed task prompt, ontology, checkpoint-selection criterion, model revision, split hashes, decoding settings, random seed, and scorer options (span threshold, unknown-label policy, label matching) with each result. The experiment runner performs greedy vLLM generation after temporary merging of selected LoRA weights and saves raw responses, parsed annotations, and parsing errors alongside the metrics.

We report independent Sub-code-name F1 and joint Code/Sub-code-pair F1 as separate metrics, making both child-label recognition and hierarchical attachment observable. Preserve the individual seed results before calculating aggregate means and standard deviations. [Experiment workflow](quickstart.md).
