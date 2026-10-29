# Training, checkpoints, and inference

All commands are run from the repository root. After `pip install -e .`, use `star-dro`; alternatively, use `python scripts/star_dro.py` with the same arguments. The CLI lists subcommands with `--help` and validates arguments before model execution where possible.

## Prepare data

Use separate authorized split paths matching [data_format.md](data_format.md). These may be JSON/JSONL files or saved Arrow-backed dataset directories. The JSONL path layout is:

```text
data/private/train.jsonl
data/private/valid.jsonl
data/private/test.jsonl
data/private/test-input.jsonl
data/private/ontology.json
```

`test-input.jsonl` contains the test IDs and model inputs without annotations. Include all task instructions and contextual metadata in the messages. The repository never derives test prompts from gold annotations. A gold file may be used as prediction input, but the generator reads only its prompt/messages and ignores annotations; the separate input-only file makes this separation easier to inspect.

Validate the splits first:

```bash
star-dro validate-data --train data/private/train.jsonl --valid data/private/valid.jsonl --test data/private/test.jsonl --ontology data/private/ontology.json --output outputs/data-summary.json
```

For an existing saved Arrow split, replace the corresponding JSONL path with its Dataset directory. Use prepare-data only when you want to persist normalized data or explicitly convert storage. JSON input is not automatically written to Arrow. The [complete runbook](quickstart.md) covers normalization and the multi-seed experiment runner.

## Train

```bash
star-dro train --model meta-llama/Llama-3.2-3B-Instruct --train data/private/train.jsonl --valid data/private/valid.jsonl --ontology data/private/ontology.json --config configs/reference.json --output outputs/stardro-seed0 --seed 0 --device cuda --precision bfloat16
```

Use an available model appropriate to your hardware. For reproducibility, pin a model revision with `--revision COMMIT_HASH`. The tokenizer is loaded from the same model/revision. A fast tokenizer, EOS token, and compatible chat template are required for chat-history inputs. Rendered prompts can be used when an appropriate chat template is not available.

The script supports one process and one device. It rejects a distributed WORLD_SIZE greater than one. It masks prompt/padding labels, preserves complete annotations, uses ordinary completion-only training before activation, and updates STaR-DRO once per active training minibatch. It raises errors on overlong records, invalid annotation coverage and nonfinite loss. It never silently truncates a record.

## Run artifacts

```text
outputs/stardro-seed0/
  run.json               Resolved configuration, seed, data hashes, group order,
                         package versions, available model/tokenizer commits
  train.jsonl            Step loss, learning rate, activation, q, EMA, scale,
                         rho, ascent signals and applied group multipliers
  validation.jsonl       Raw completion CE and configured group-attribution diagnostics
  selection.json         Reported checkpoint, criterion and whether DRO was active
  epoch-001/
    ...                  Model or LoRA adapter, tokenizer
    star_dro.json        STaR-DRO configuration, q, EMA and update count
    standard_dro.json    Standard DRO q/logq state instead, for that baseline
    run.json             Run identity needed to load adapters portably
    training_state.pt    Optimizer, scheduler, RNG and completed epoch/step state
  epoch-002/
  epoch-003/
```

Trace files do not contain prompts or clinical message text. Predictions do contain generated text; store private-data run artifacts under the same access controls as the authorized data.

Validation reports completion-token CE and group CE under the selected construction and attribution, together with observation counts, effective mass, macro group CE, and worst observed group CE. Missing groups have null CE and are excluded from macro/worst aggregates. These diagnostics are not adversarially reweighted. Checkpoint selection follows `training.checkpoint_selection`: `last` (default, the manuscript protocol) reports the final epoch; `validation_ce` reports the epoch with the lowest validation completion CE, never test F1. Under `validation_ce` a robust run can select an epoch before DRO activation; `selection.json` records the epoch, the criterion and `robust_objective_active`. See [grouping.md](grouping.md) and [evaluation.md](evaluation.md).

## Epoch-boundary resume

Resume from the latest completed epoch checkpoint with matching data, model, revision, seed, configuration and precision:

```bash
star-dro train --model meta-llama/Llama-3.2-3B-Instruct --train data/private/train.jsonl --valid data/private/valid.jsonl --ontology data/private/ontology.json --config configs/reference.json --output outputs/stardro-seed0 --seed 0 --device cuda --precision bfloat16 --resume outputs/stardro-seed0/epoch-002
```

With a three-epoch target, resuming from epoch 2 continues with epoch 3. The script restores model/adapters, optimizer, scheduler, Python/Torch random states, q and EMA, and trims partial trace records beyond the last completed checkpoint. It does not resume from a position inside a minibatch or reconstruct a partially written checkpoint. Checkpoints created before interruption should be retained intact. Only load trusted training_state.pt files: full training-state restoration uses PyTorch serialization with Python objects.

For exact continuation on the same device, retain the model revision and environment. Backend nondeterminism and changes of hardware/software can still affect numerical trajectories; the script does not promise bitwise equivalence across environments.

## Predict

Read selection.json, choose the saved checkpoint, and generate outputs:

```bash
star-dro predict --checkpoint outputs/stardro-seed0 --input data/private/test-input.jsonl --output outputs/stardro-seed0/predictions.jsonl --device cuda --precision bfloat16 --max-new-tokens 512
```

The command accepts a run root, which resolves selection.json automatically, or an explicit epoch checkpoint. It supports saved full models and LoRA adapters. Adapter checkpoints are merged into temporary standalone weights using the recorded base model and commit, then tested with vLLM. If a local base-model directory moved, provide `--base-model NEW_LOCATION`; preserve the base weights and tokenizer revision.

The merge runs in a child process that exits before vLLM starts. Automatic merging defaults to CPU, preserves the requested export precision, calls `merge_and_unload(safe_merge=True)`, and saves sharded safetensors and the checkpoint tokenizer. Temporary full weights are deleted after prediction or a handled failure. `--merge-temp-dir` chooses their filesystem; by default they are placed beside predictions. The original adapter and training checkpoints are retained.

Generation uses vLLM only, with temperature 0, one completion, EOS IDs and the `JSON_END` stop string. Select visible GPUs with CUDA_VISIBLE_DEVICES and pass `--device cuda`; `--tensor-parallel-size 2` uses two visible GPUs for inference only. `--gpu-memory-utilization` defaults to 0.7 and `--batch-size` defaults to 1. Optional `--enforce-eager` disables CUDA graphs. Prediction requires CUDA; CPU-only training and merging remain supported.

The default context/output budgets remain 8192/512. Pass `--max-new-tokens 1024` to match the working notebook's test budget. `--seed` defaults to 0 for standalone prediction; the experiment runner forwards each training seed. [All inference options](quickstart.md#vllm-and-merge-settings).

Raw output and parsing status are saved for each completed prediction. `predictions.jsonl.meta.json` records the backend, sampling and engine settings, merge provenance, environment, input hash and invalid-output count. Generation itself does not repair or remap labels; subsequent scoring applies output repair and ontology-based label normalization. Prompt plus output budget may not exceed `--max-length`; inputs are not silently truncated. If interrupted, inspect partial predictions and rerun to a new path. Metadata and experiment summaries are written only after their corresponding stages finish successfully.

## Explicit model export

To retain full merged weights intentionally:

```bash
star-dro merge --checkpoint outputs/stardro-seed0 --output outputs/stardro-seed0-merged --device cpu --precision bfloat16
star-dro predict --checkpoint outputs/stardro-seed0-merged --input data/private/test-input.jsonl --output outputs/stardro-seed0/merged-predictions.jsonl --device cuda --precision bfloat16
```

The merge output must be a new directory outside its source checkpoint. `merge.json` records the selected adapter, base location/revision, precision, device and safe-merge setting. This explicit export persists full weights; ordinary `predict` and the multi-seed runner use temporary exports instead. Full-fine-tuning checkpoints already contain standalone weights and require no merge.

## Score

```bash
star-dro evaluate --gold data/private/test.jsonl --predictions outputs/stardro-seed0/predictions.jsonl --ontology data/private/ontology.json --output outputs/stardro-seed0/test-metrics.json
```

See [evaluation.md](evaluation.md) for metric definitions and edge cases. Record the chosen checkpoint, model revision, seed, split hashes, ontology and decoding settings with any reported result. For multiple runs, use separate output directories and report the aggregation rule rather than combining steps from different runs.

For `training.objective: standard_dro`, `train.jsonl` records `standard_dro_active`, q and its annotation-count-corrected multiplier diagnostics. This baseline does not maintain EMA. Its validation group-loss, checkpoint selection, merging, vLLM generation and scoring stages are identical to the other objectives. See [configuration.md](configuration.md#standard-dro-baseline).
