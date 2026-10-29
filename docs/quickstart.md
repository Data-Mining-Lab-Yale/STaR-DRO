# Running a STaR-DRO experiment

## Required inputs

Before running a model, identify:

1. A model ID or local model directory, with a fixed revision when available.
2. Authorized training, validation and test paths in supported JSON/JSONL/Arrow storage.
3. Task instructions and codebook already included in the model-visible prompts.
4. The task ontology JSON path when hierarchical validity is to be checked.
5. A new output directory, explicit seeds and hardware/precision settings.

Begin with the [prompt template](../prompts/structured_prediction.md) and [dataset preparation steps](data_format.md#step-by-step-dataset-preparation). Complete your task instruction, prepare source records, then use `scripts/build_records.py` to place the instruction and instance content into each record's `messages[0].content`. Clinical experiments use authorized study data and task definitions; see [data availability](data_availability.md).

## Environment

From the repository root, create and activate a Python 3.11 environment, install compatible CUDA/PyTorch/vLLM builds together, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_tests.py --require-torch --require-data
star-dro --help
```

Arrow inputs need datasets, included in requirements.txt. JSON-only validation can run from the source checkout with the standard library. Scoring uses only the standard library; RapidFuzz (in requirements.txt) is an optional fuzzy-matching backend. Full training additionally needs torch, transformers and peft for LoRA. Prediction uses vLLM on CUDA. The `inference` extra installs the bounded vLLM 0.11.x integration and merging dependencies; run `python -m pip check` after installation. CPU-only development uses requirements-dev.txt and does not install vLLM. Model access must be configured through the model provider's normal authentication.

## Prepare the prompt and records

Canonical data can be used directly. Existing entries/query-answer exports are normalized by the shared loader. To persist normalized rows, use prepare-data. Choose --format arrow only when an Arrow-backed output is desired; JSON-to-Arrow conversion is not a prerequisite for training.

Saved DatasetDicts require per-split directory paths. Do not combine splits or place the same original message into several splits. No preparation command creates train/validation/test partitions.

## Validate before training

```bash
star-dro validate-data --train TRAIN_PATH --valid VALID_PATH --test TEST_PATH --ontology ONTOLOGY_PATH --config configs/reference.json --output outputs/data-summary.json
```

Replace uppercase paths with your actual files/directories. Omit --ontology if no ontology validation is required. Resolve validation errors before training. Inspect record counts, Code coverage, IDs, source grounding and split hashes. Ensure the prompts specify the same output envelope as output_format; the loader preserves the results envelope of supported answer exports.

## Run the complete sequence

```bash
python scripts/run_experiment.py --model MODEL_ID_OR_DIRECTORY --revision MODEL_COMMIT --train TRAIN_PATH --valid VALID_PATH --test TEST_PATH --ontology ONTOLOGY_PATH --config configs/reference.json --output outputs/stardro-experiment --seeds 0,1,2 --device cuda --precision bfloat16
```

Replace model, revision and paths. For a local model without a revision, omit --revision. Omit --ontology only when pair validation is not needed. Use explicit, distinct seeds for your experiment. Training supports one process/device. The complete runner requires CUDA for vLLM; CPU-only training remains available through the train subcommand. `--tensor-parallel-size` controls vLLM inference GPUs only and does not enable distributed training.

The runner executes:

1. Split validation before model loading.
2. A continuous training job for each seed, with epoch-2 STaR-DRO activation.
3. Resolve that run's selected checkpoint (final epoch by default; see `training.checkpoint_selection`) and merge its LoRA adapter into temporary standalone safetensors weights.
4. Greedy vLLM generation from the merged model, then deletion of the temporary weights. Full-model checkpoints go directly to vLLM.
5. Ontology-based test scoring and per-seed mean/sample-standard-deviation aggregation.

Training and prediction run in fresh Python subprocesses, and automatic merging runs in its own subprocess before the vLLM engine starts. Guarded entry points and spawn workers avoid recursive execution and Python-stdin worker failures.

Gold annotations in the test input are never passed to model generation. The prediction loader reads only messages/prompt after normalization. The test set is not used for checkpoint selection. To inspect prediction-only input separately, export it with prepare-data --input-only.

## Expected artifacts

The experiment root contains experiment.json and summary.json. Each seed-N directory contains run.json, train.jsonl, validation.jsonl, selection.json, completed epoch checkpoints, predictions.jsonl, prediction metadata (including merge provenance and vLLM settings) and test-metrics.json. The runner retains adapters and training state; automatic merged full weights are temporary and removed after successful prediction or a handled failure. Abrupt job termination can leave temporary directories for the operator to remove. Summary metric values are fractions in [0,1], with sample standard deviation when at least two seeds complete. An incomplete experiment does not write a final summary.

Check the selected epoch, model commit, data hashes, software environment, Code inventory, STaR-DRO activation and invalid-output rates. Raw group validation CE and downstream F1 have different meanings; both are retained for inspection.

## Individual stages and interruption

The same operations are available through train, merge, predict and evaluate. `predict` automatically merges adapter checkpoints; use `merge` only to export a standalone full model intentionally. Passing a run root to predict resolves selection.json automatically; passing an epoch directory selects it explicitly. Refer to [usage.md](usage.md) for detailed flags and resumption.

The experiment runner requires a new root and does not automatically resume multiple interrupted runs. Resume a completed epoch with the train subcommand, then execute prediction/scoring for the resumed run and retain the final metrics. Avoid rerunning a whole completed experiment into the same output path.

## Failure conditions

| Condition | Action |
| --- | --- |
| Missing dependency | Install the documented requirements in the active environment. |
| Model authentication/access error | Resolve access to the selected model and revision. |
| Unseen validation/test Code | Verify the intended training inventory and approved split assignment. |
| Invalid gold JSON or parent/child pair | Correct the source annotation against the codebook. |
| Grounding error | Verify exact evidence text in source_text. |
| Unsupported chat template | Supply a supported model/template or a correctly rendered prompt. |
| Input exceeds maximum length | Split the record explicitly with a task-appropriate policy. |
| Output path exists | Choose a new path, or resume a supported training checkpoint. |
| DatasetDict passed as one split | Select the appropriate saved split subdirectory. |

The [reproducibility guide](reproducibility.md) describes the numerical, gradient, Arrow, and model integration checks, together with the saved environment and experiment provenance.

## Select grouping and baseline

Start with `configs/reference.json` (Code, annotation). Use `configs/code_sample.json` to change only attribution, or the other [grouping configurations](grouping.md) to choose semantic/complexity groups. `configs/sft.json` runs the same continuous training recipe without adversary updates. `configs/standard_dro.json` selects the notebook144 exponential-update baseline with eta 0.1 and annotation-count correction. JSON files are partial overrides of shared defaults; record all resolved values from run.json. Compare seed-matched runs on the same authorized split and model revision. Grouping example configs provide starting points for your task's group structure.

## Inspect the completed run

Read `run.json` for the resolved configuration, ordered inventory, rare-group mapping, revision and data hashes. `train.jsonl` includes q, EMA, present-group scale, applied multipliers, entropy, effective group count and numerical support. `validation.jsonl` reports raw completion CE and group CE/counts/effective mass using the selected attribution. `selection.json` states the reported checkpoint and selection criterion. `test-metrics.json` keeps strict parsing, label policy and Span matching explicit.

## vLLM and merge settings

Both `predict` and the multi-seed runner accept:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--tensor-parallel-size` | 1 | GPUs visible to vLLM; training remains single-device |
| `--gpu-memory-utilization` | 0.7 | vLLM GPU memory budget |
| `--batch-size` | 1 | Records submitted per generation call; separate from training batch size |
| `--max-length` | 8192 | Prompt plus output context limit |
| `--max-new-tokens` | 512 | Output budget; pass 1024 for the notebook's test budget |
| `--stop` | JSON_END | Output stop string, omitted from returned text |
| `--enforce-eager` | off | Disable vLLM CUDA graphs when needed |
| `--merge-device` | cpu | Temporary merge device; cpu or cuda[:INDEX] |
| `--merge-temp-dir` | Prediction output parent | Filesystem for temporary standalone weights |

Inference always uses temperature 0, one completion, the model/tokenizer EOS IDs, and the supplied stop string. Prompt token IDs are submitted directly after the shared chat-template rendering, preventing extra BOS tokens. No constrained decoding or fallback to Transformers generation is used.

Merging needs host RAM (or GPU memory with `--merge-device cuda`) for the unquantized base and temporary workspace, and disk space for a full standalone model, even though those weights are later deleted. Choose a filesystem with sufficient capacity using `--merge-temp-dir`. A 70B bf16 export contains roughly 140 GB of weight data; tensor parallelism affects inference, not merge RAM requirements.

Before launching a full experiment, test an existing checkpoint in an allocated CUDA job:

```bash
star-dro predict --checkpoint EXISTING_RUN --input TEST_PATH --output outputs/vllm-smoke/predictions.jsonl --device cuda --precision bfloat16 --max-new-tokens 1024 --tensor-parallel-size 1 --gpu-memory-utilization 0.7
star-dro evaluate --gold TEST_PATH --predictions outputs/vllm-smoke/predictions.jsonl --output outputs/vllm-smoke/metrics.json
```

Use real authorized paths and a new output directory. Add `--ontology ONTOLOGY_PATH` to evaluate against your label inventory. This runs the actual merge, vLLM and scoring path without retraining.

## Supplied-data smoke tests with overlapping splits

Normal validation/training still reject exact cross-split overlap. The optional `--allow-split-overlap` flag on `validate-data`, `train` and `scripts/run_experiment.py` is an explicit smoke-test exception for checking the pipeline with original overlapping files. It does not remove, deduplicate, resplit or alter records. The runner saves a `data_validation.json` overlap report, marks experiment/test/summary outputs as `smoke_test_with_overlap_allowed`, and includes the exception in training identity. Do not report these results as held-out generalization estimates. Obtain an approved disjoint split before a scientific evaluation.

Repeated prompts within a split may represent different source records; retain their true source IDs when known. Row-locator IDs for an uploaded export are not evidence of distinct original messages and never bypass exact-prompt overlap reporting. The data-format guide explains the usual source-message split requirements.
