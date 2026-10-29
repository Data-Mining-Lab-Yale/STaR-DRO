# Reproducibility

## Experiment setup

Use Python 3.11 with mutually compatible CUDA/PyTorch/vLLM builds and the dependencies in `requirements.txt`. Our training entry point supports one process/device, LoRA on attention and MLP projections or full fine-tuning, and both sample and annotation attribution. [Configuration reference](configuration.md).

For each experiment, preserve:

1. The completed task prompt and ontology used to prepare every split.
2. Stable source-record IDs and the original message/document split assignment.
3. Training, validation, test, and ontology hashes.
4. The resolved configuration, model/tokenizer revision, random seeds, and software environment.
5. Per-run checkpoints, checkpoint selection, generation settings, and evaluation outputs.

`run.json` records the resolved configuration, seed, data hashes, ordered group inventory, optional rare-group mapping, available model/tokenizer commits, device, parameter counts, and package versions. Pin revisions with `--revision` and retain a dependency freeze alongside the run.

## Prompt and data consistency

Prepare the instruction using our [modular template](../prompts/structured_prediction.md) and apply it consistently across the experiment's splits. `scripts/build_records.py` combines the completed instruction with source metadata, context, and target text; it checks ontology validity and exact evidence grounding. [Dataset preparation](data_format.md#step-by-step-dataset-preparation).

Assign original messages/documents to splits before sentence segmentation or task expansion. Our EPPC Miner study uses a 70/15/15 message-level split with 840 training, 180 validation, and 180 test messages. Retain the private source-message mapping and use `validate-data --config CONFIG` to check exact input/ID separation and configured group support.

## Training and evaluation

The reference recipe uses three continuous epochs with STaR-DRO active from epoch 2. EMA, q, optimizer, and scheduler state persist across epochs. Completed epoch checkpoints retain the model or adapter, tokenizer, optimizer, scheduler, random state, and adversary for resumption.

Checkpoint selection defaults to the final epoch (`training.checkpoint_selection: last`); `validation_ce` selects by validation completion CE. The three-seed runner retains each seed's outputs, merges selected adapters temporarily and generates vLLM predictions from its selected checkpoint, and reports means and sample standard deviations. Code/Sub-code and Span metrics use the policies documented in [evaluation.md](evaluation.md). [Complete runbook](quickstart.md) · [Resumption and prediction](usage.md).

## Software validation

Run the full suite with:

```bash
python scripts/run_tests.py --require-torch --require-data --require-model
ruff check src scripts tests
python -m build
python scripts/package_release.py
```

The suite checks analytic sparse projections, simplex invariants, persistent EMA, overlap correction, relative scaling, multiplier bounds, all five group constructions, both attribution modes, tensor gradients, prompt assembly, grounding, label/Span metrics, and release contents. Real Arrow tests exercise canonical and entries/query-answer layouts, saved Dataset directories, and individual stream shards.

Offline model integration uses a tiny local Llama model and tokenizer with LoRA and unit-test records. It exercises continuous training, epoch-boundary resumption, checkpoint selection, standalone adapter merging with reloaded-logit parity, alternative grouping/sample attribution, and the SFT and Standard DRO baselines. vLLM interface tests use explicit test doubles to check token IDs, EOS/stop behavior, engine arguments, output records, subprocess isolation and cleanup. They do not establish GPU runtime correctness. Run the [checkpoint-only smoke test](quickstart.md#vllm-and-merge-settings) on allocated GPUs before a large experiment. Resumption is checked against the complete uninterrupted training and validation traces.

[Latest test report](validation.json) · [Software validation environment](tested-environment.txt). The CI workflow runs core interfaces on Python 3.10–3.12 and the required tensor, data, model, lint, and package checks on Python 3.11.

Standard DRO has independent hand-calculation, 200-step notebook-update parity, continuation, empty-batch and numerical-recovery checks. The opt-in overlapping-split smoke mode is tested to ensure normal validation still rejects overlap and the explicit exception reports it without changing records.
