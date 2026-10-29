# Data formats and preparation

## Step-by-step dataset preparation

### 1. Define the task instruction and ontology

Copy the [general prompt template](../prompts/structured_prediction.md) or [EPPC Miner instruction](../prompts/eppc_miner.md) to your task-specific Markdown file. Complete the label definitions, disambiguation rules, metadata logic, output schema, and quality gate in its single fenced `text` block. The builder reads this completed block as the task instruction. Use the same instruction for every split in an experiment.

Create an ontology JSON mapping parent Codes to valid Sub-codes. For the nonclinical README example:

```json
{"Topic": ["Schedule"], "Courtesy": ["Thanks"]}
```

Use your task's complete inventory. Operational label definitions belong in the prompt; the ontology mapping supports programmatic validation and defines the label inventory used by the scorer.

### 2. Create source records and assign splits

Prepare JSONL with the fields below. Assign original messages/documents to splits before expanding them into sentence or task records, retaining stable IDs and source-level provenance. Our study uses a 70/15/15 message-level split.

| Source field | Requirement | Role |
| --- | --- | --- |
| `id` | Nonempty, stable, unique string | Join predictions and gold; preserve source identity |
| `target_text` | Nonempty string | Text from which evidence Spans are drawn |
| `context` | Optional string | Surrounding discourse for interpretation |
| `metadata` | Optional JSON object | Direction, speaker role, document type, or other task variables |
| `annotations` | List of Code/Sub-code/Span objects | Gold structured output; `[]` for no annotations |

The [README source example](../README.md#2-organize-annotated-source-records) shows a complete record. For EPPC Miner, include `TO_PAT_YN` in metadata and define its direction mapping in the instruction.

### 3. Assemble the prompt into every record

```bash
python scripts/build_records.py --input data/private/train-source.jsonl --instructions prompts/my_task.md --ontology data/private/ontology.json --output data/private/train.jsonl
```

Repeat for validation and test using the same completed instruction. The builder validates each gold label pair and checks exact evidence grounding, then writes canonical JSONL. It creates this model input:

```text
[Your completed six-module task instruction]

METADATA:
[This record's metadata as JSON]

CONTEXT (for interpretation):
[This record's context]

TARGET TEXT (evidence source):
[This record's target_text]
```

| Prepared field | Value |
| --- | --- |
| `messages[0].role` | `user` |
| `messages[0].content` | Completed instruction + metadata + context + target text |
| `annotations` | Gold annotation list from the source record |
| `source_text` | The exact `target_text`, used for grounding validation |
| `output_format` | `results` by default; `--output-format list` selects a list envelope |
| `id` | Original source-record ID |

**This prepared `messages[0].content` is the prompt used by training and prediction.** The tokenizer applies the selected model's chat template. The trainer serializes `annotations` as the target completion, masks prompt tokens, and computes completion loss. Set the instruction's answer envelope consistently with `output_format`.

If your data already have complete instructions and instance content in `messages`, `entries`, `query`, or a rendered `prompt`, pass them to the shared loader directly. The explicit builder is useful when starting from separate source fields.

### 4. Export the preferred storage format

Use the prepared JSONL directly or convert each split to Arrow:

```bash
star-dro prepare-data --input data/private/train.jsonl --output data/private/train-arrow --format arrow
star-dro prepare-data --input data/private/valid.jsonl --output data/private/valid-arrow --format arrow
star-dro prepare-data --input data/private/test.jsonl --output data/private/test-arrow --format arrow
```

`--layout entries` creates human/assistant exports. `--layout query_answer` creates query/answer exports. The canonical layout retains grounding fields and is the default. The detailed reference schemas and metadata-preserving export option appear below.

### 5. Validate and run

```bash
star-dro validate-data --train data/private/train-arrow --valid data/private/valid-arrow --test data/private/test-arrow --ontology data/private/ontology.json --config configs/reference.json --output outputs/data-summary.json
```

Inspect split separation, Code coverage, group inventory, hierarchy validity, and grounding before training. Keep the completed prompt, ontology, model revision, and split hashes with your experiment configuration. [Run the experiment](quickstart.md).

## Supported storage formats

| Input path | Reader | Requirement |
| --- | --- | --- |
| `.jsonl` or `.ndjson` file | Line-by-line JSON parsing | One record per nonblank line |
| `.json` file | JSON parsing | One record object or an array of records |
| Hugging Face Dataset directory | `datasets.load_from_disk` | Directory created by Dataset.save_to_disk |
| `.arrow` file | `datasets.Dataset.from_file` | Compatible Arrow IPC stream, including Hugging Face dataset shards |

Training, prediction, data validation and gold scoring share this input loader. Arrow access requires the `datasets` dependency in requirements.txt; `pip install -e ".[data]"` installs only the data extras. Loading uses the documented [Hugging Face disk/Arrow APIs](https://huggingface.co/docs/datasets/en/loading#arrow).

Pass one split per path. For a saved DatasetDict, select a split subdirectory such as `DATASET_DIRECTORY/train`; do not pass the multi-split root. Multiple `.arrow` shards should be loaded through their saved Dataset directory so row order and all shards are retained. A lone `.arrow` path loads only that file.

JSON is read directly and does not automatically create an Arrow cache or file. The trainer normalizes records, tokenizes them and constructs tensors in memory. Storage conversion is a separate, explicit operation.

## Canonical record contract

| Field | Type | Requirement and interpretation |
| --- | --- | --- |
| `id` | Nonempty string | Required; unique within files and across dataset splits. Predictions join to gold by this value. |
| `messages` | Nonempty list of message objects | Provide exactly one of messages and prompt. Contains model-visible instructions and context; ends with a user turn. |
| `prompt` | Nonempty string | Complete rendered model input ending immediately before the assistant answer; alternative to messages. |
| `annotations` | List of annotation objects | Required for training, validation and gold scoring. May be omitted for prediction inputs; empty lists are supported. |
| `source_text` | String | Optional evidence source. Gold Spans must be exact, case-sensitive substrings when supplied. |
| `output_format` | `list` or `results` | Optional completion envelope. `list` serializes the annotation list; `results` wraps it in an object whose sole key is results. Defaults to list. |

Additional canonical fields are rejected. Arrow null values in optional columns are treated as absent. Source text and task metadata must appear in messages/prompt to be visible to the model; source_text is a grounding-validation field and is not automatically inserted into the prompt.

The [README source example](../README.md#2-organize-annotated-source-records) is converted into this contract by `build_records.py`. A JSONL record must occupy one line; JSON objects may be formatted across lines inside a `.json` file.

### Message objects

Every message has exactly `role` and `content`. Allowed roles are system, user and assistant; content is a nonempty string. The final input turn is user. The tokenizer renders messages using its chat template and assistant generation prefix. The model's template determines valid role combinations and turn ordering.

Include the full task instructions, approved label definitions and relevant context. Use the completed task instruction and instance fields assembled by `scripts/build_records.py`. Rendered prompts bypass chat templating, use add_special_tokens=False, and must contain the correct control tokens. Gold annotations must not be embedded in the input prompt.

### Annotation objects

Each annotation has exactly three case-sensitive, nonempty string fields:

| Field | Meaning |
| --- | --- |
| Code | Parent label; identifies the group when Code grouping is selected |
| Sub-code | Child label under that Code |
| Span | Evidence supporting the annotation |

A record may contain multiple annotations, including repeated Codes. An optional ontology JSON maps each Code string to a nonempty list of unique allowed Sub-code strings. With --ontology, every gold parent/child pair must occur in that mapping. It is not learned from validation or test labels.

Our clinical study uses nine Codes and 38 Sub-codes. Complete the EPPC template with the approved codebook and operational definitions.

## Existing entries and query/answer exports

The loader also accepts these existing source layouts, in JSON or Arrow:

| Source layout | Normalization |
| --- | --- |
| `id`, `entries` | Entries contain role/content turns. human maps to user; user, system and assistant are retained. A final assistant turn supplies the answer and is removed from model input. Earlier context remains. |
| `id`, `query`, `answer` | Query becomes one user turn; answer supplies the gold annotations. For prediction-only records, answer may be absent. |

Answers may be JSON text or parsed values: either an annotation list or a results-wrapped list. A results wrapper is retained through output_format=results, so training does not silently change its target envelope. The historical Subcode field spelling is normalized to Sub-code. Parsing is strict JSON; malformed gold answers raise errors rather than being repaired or replaced by empty annotations.

Numeric IDs in these source layouts are converted to strings. If a source ID is missing, a deterministic ID is derived from the model-visible input. Ensure IDs are unique across splits; independently numbered split-local row indices should be replaced with stable message identifiers. Additional source export columns are not forwarded to the canonical model input. Canonical records retain strict field validation.

An annotated entries export must end in its target assistant response. For prediction-only entries, end with the user input. Converted canonical records can be inspected on disk before training with prepare-data.

## Explicit preparation and conversion

Normalize a supported source into canonical JSONL:

```bash
star-dro prepare-data --input SOURCE_PATH --output data/private/train.jsonl --format jsonl
```

Convert JSON/JSONL or normalize an existing Arrow dataset into an Arrow-backed directory:

```bash
star-dro prepare-data --input data/private/train.jsonl --output data/private/train-arrow --format arrow
```

Supply `data/private/train-arrow` directly as --train. Repeat separately for validation and test as needed. Conversion preserves record order and IDs and never generates split assignments.

Create prediction-only input by removing gold annotations:

```bash
star-dro prepare-data --input data/private/test.jsonl --output data/private/test-input.jsonl --format jsonl --input-only
```

The input-only option accepts annotated or already input-only records. It retains input fields and IDs and removes annotations from exported canonical rows. An optional --ontology validates any gold annotations before export. Output paths must be new.

## Validation and token alignment

Use validate-data before training. It checks fields, IDs, gold annotations, optional hierarchy/grounding, training Code support and exact split overlap by IDs, inputs and supplied source texts. Records from the same original message must remain in one split; exact checks cannot detect paraphrases, common patients or upstream grouping mistakes.

Groups are fitted on training according to the configured construction, independently of sample/annotation attribution. Validation/test Codes require training support; `validate-data --config CONFIG` additionally checks the selected group inventory. Missing validation observations produce null group CE. Reports contain counts/hashes, not message text. Directory SHA-256 includes relative filenames and contents. [Grouping conventions](grouping.md).

Fast-tokenizer character offsets align each complete annotation object to its generated tokens. Prompt and padding labels are -100. Causal shifting maps input position t to loss position t−1. Uncovered completion tokens remain neutral in annotation mode. Tokens intersecting multiple object masks use the mean multiplier. A token crossing the prompt/completion boundary, an annotation without token coverage, or an overlong record raises an error; nothing is silently truncated or dropped.

Schemas describe [canonical input records](../schemas/record.schema.json), [annotated records](../schemas/training_record.schema.json) and [prediction records](../schemas/predictions.schema.json). Runtime checks also enforce final user turn, whitespace-only strings, uniqueness and evidence grounding.

## Prediction outputs

Each generated record contains id, raw prediction text, parsed annotations and parse_error. The raw prediction is authoritative for scoring when present. Both list and results-wrapped response envelopes are accepted. Invalid generated JSON retains its raw text, yields an empty parsed list and records a non-null parsing error.

The scorer also accepts pre-parsed id/annotations records. Predictions remain JSONL output even when model inputs use Arrow. Gold/prediction IDs must match exactly. See [evaluation.md](evaluation.md) for matching and error-counting rules.

## Reconstruct the reference Arrow layouts

The supplied reference exports use two storage schemas. These are **schemas, not a redistribution of the data**:

```text
Training/validation export:
  id: string
  entries: list<struct<content: string, role: string>>

Evaluation export:
  query: string
  answer: string
  context: string
  TO_PAT_YN: string
```

An `entries` row contains the full task query as a human turn, followed by a final assistant turn containing the gold JSON **as a string**:

```json
{
  "id": "source-message-001:sentence-01",
  "entries": [
    {"content": "Extract Topic/Schedule evidence. Return a results array with Code, Sub-code, Span. Target: The office opens at nine.", "role": "human"},
    {"content": "{\"results\": [{\"Code\": \"Topic\", \"Sub-code\": \"Schedule\", \"Span\": \"The office opens at nine.\"}]}", "role": "assistant"}
  ]
}
```

In the evaluation layout, `query` is the complete model-visible input, `answer` is the gold JSON string, `context` retains source metadata, and `TO_PAT_YN` is direction metadata (`Y`: to patient; `N`: from patient). Metadata columns are not automatically concatenated into `query` by the reader. The institutional task builder must include the target text, surrounding context, direction, task instructions, and codebook in the intended prompt. Preserve the distinction between evidence-bearing target text and context used only for interpretation.

This nonclinical example illustrates the four-column layout; direction is shown only to demonstrate the storage contract:

```json
{
  "query": "Extract Topic/Schedule evidence. Return a results array with Code, Sub-code, Span. Target: The office opens at nine.",
  "answer": "{\"results\": [{\"Code\": \"Topic\", \"Sub-code\": \"Schedule\", \"Span\": \"The office opens at nine.\"}]}",
  "context": "The office opens at nine.",
  "TO_PAT_YN": "Y"
}
```

Create the exact reference column layout from an already assembled JSON/JSONL source:

```bash
python scripts/prepare_notebook_data.py --input data/private/train-entries.jsonl --output data/private/train-arrow --layout entries
python scripts/prepare_notebook_data.py --input data/private/valid-entries.jsonl --output data/private/valid-arrow --layout entries
python scripts/prepare_notebook_data.py --input data/private/test-query-answer.jsonl --output data/private/test-arrow --layout query_answer
```

The script validates gold, uses explicit Hugging Face `Features`, and calls `Dataset.save_to_disk`. `make_arrow_dataset` constructs the Dataset from a PyArrow table with a deterministic SHA-256 content fingerprint, avoiding interpreter-dependent pickle fingerprints. The resulting directory contains Arrow shard(s), `dataset_info.json`, and `state.json`. Use it with `load_from_disk` in the notebook or this release; use `Dataset.from_file` for an individual compatible IPC stream shard. Do not rename JSON to `.arrow` or substitute Parquet/IPC file format for an IPC stream. [Official Arrow APIs](https://huggingface.co/docs/datasets/loading#arrow).

Equivalent Python for `entries` data you have already validated:

```python
import json
from datasets import Features, Value
from star_dro.data import make_arrow_dataset

with open("data/private/train-entries.jsonl", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
features = Features({
    "id": Value("string"),
    "entries": [{"content": Value("string"), "role": Value("string")}],
})
make_arrow_dataset(rows, features=features).save_to_disk("data/private/train-arrow")
```

For the evaluation layout, use a string `Value` for each of `query`, `answer`, `context`, and `TO_PAT_YN`. A stable `id` column is also supported and recommended for joining predictions, especially when distinct source records share identical text. When IDs are absent, the loader derives them from the model input. Duplicate resolved IDs fail explicitly: audit the source records and provide their true identifiers; do not silently deduplicate them or assign fresh row numbers to conceal overlap.

The CLI’s `prepare-data --layout entries` exports canonical messages into the minimal `id`/`entries` contract. `--layout query_answer` exports `id`/`query`/`answer` and requires one complete user turn. These minimal exports omit optional grounding/metadata columns; use `prepare_notebook_data.py` when preserving the four-column evaluation schema is required. Canonical Arrow output retains `source_text` and messages.

Storage conversion never establishes the manuscript split. Row counts can differ from source-message counts after segmentation or task expansion. Maintain the upstream source-message mapping and split manifest in your private workspace and verify message-level separation independently of exact-input checks. No clinical data, split manifest, source-message identifiers, or converted Arrow files belong in a public/reviewer code archive.
