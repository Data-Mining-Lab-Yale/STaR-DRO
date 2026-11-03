# Modular prompt template for structured prediction

Our prompt architecture makes structured prediction explicit, verifiable, and sensitive to task metadata. The six modules below organize the domain knowledge, decision rules, evidence constraints, and output contract used by the model.

| Module | Purpose | What you supply |
| --- | --- | --- |
| **M1 · Hierarchical XML structuring** | Give global instructions and local rules clear scopes | Expert role, accuracy expectations, task definition, label inventory |
| **M2 · Expert-curated disambiguation** | Resolve boundaries between confusable labels | Operational decision criteria from your annotation guidelines |
| **M3 · Reasoning scaffold** | Verify context, labels, and evidence before output | A four-step verification routine aligned with your task |
| **M4 · Metadata-aware logic** | Condition eligible labels on observed attributes | Speaker role, direction, document type, or other control variables |
| **M5 · Output schema contract** | Produce parseable, grounded annotations | Exact field names, valid parent–child pairs, evidence rules, and answer envelope |
| **M6 · Self-validation gate** | Check consistency before emission | A checklist covering parsing, hierarchy, evidence, and coverage |

## Copy and complete the template

The following is the general template developed for our structured prediction framework, presented in editable Markdown. Replace each bracketed slot with your task-specific content. Keep the four verification steps and connect them to your ontology and evidence definition.

```text
<role>
[Domain expert persona and behavioral prior]
</role>

<performance_target>
[Precision-oriented accuracy expectations]
</performance_target>

<task>
[Task definition, input/output specification, multi-label constraint]
</task>

<codes_definitions>
[Authoritative label inventory with operational definitions]
</codes_definitions>

<disambiguation_rules>
[Boundary condition 1: decision criteria for confusable label pair A vs. B]
[Boundary condition 2: positional, specificity, or semantic cues]
[Boundary condition K: role-dependent or context-dependent distinctions]
</disambiguation_rules>

<reasoning_process>
Step 1: Context and metadata analysis
Step 2: Phrase decomposition and label matching
Step 3: Evidence extraction with boundary verification
Step 4: Cross-validation with loop-back conditions
</reasoning_process>

<metadata_control>
[Expose task metadata as explicit control variable, e.g. speaker role or document type]
</metadata_control>

<output_format>
[Machine-parseable hard constraints: schema, verbatim Spans, valid label pairings, and multi-label coverage]
</output_format>

<quality_gate>
[Verification checklist: parseable output, valid parent-child labels, exact evidence, disambiguation rules, and complete coverage]
</quality_gate>
```

The template corresponds to our general six-module architecture in Supplementary Figure S1. For a task-specific instantiation with direction logic, disambiguation rules, and a concrete JSON contract, see the [EPPC Miner prompt](eppc_miner.md).

## Prepare your task instruction

1. Copy this file to a task-specific Markdown file, such as `prompts/my_task.md`.
2. Fill M1 with the complete Code/Sub-code inventory and operational definitions. Use exactly the label spelling in your annotations and ontology JSON.
3. Fill M2 with criteria for the label pairs your annotators find ambiguous. Preserve the four-step M3 verification routine and specify the evidence-bearing target text.
4. Fill M4 with the meaning of each metadata key you will include in your records. For example, explain how a direction flag changes eligible labels.
5. Fill M5 with your output contract. For this repository's structured extraction workflow, specify a JSON object with a `results` array; each object has exactly `Code`, `Sub-code`, and `Span`. Set `Span` to exact text from `TARGET TEXT`, with `CONTEXT` used for interpretation.
6. Fill M6 with the checks the model should perform before returning its final JSON. Keep a single fenced `text` block containing your completed instruction.

**The completed instruction is the prompt used in your experiment.** `scripts/build_records.py` reads that block, appends each record's metadata, context, and target text, and stores the result in `messages[0].content` with role `user`. Gold annotations stay in the separate `annotations` field. The trainer applies the selected model's chat template to these prepared messages.

[Build the dataset step by step](../docs/data_format.md#step-by-step-dataset-preparation).
