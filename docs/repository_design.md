# Repository guide

The repository follows the experiment workflow from task definition to robust training and evaluation.

| Research task | Starting point |
| --- | --- |
| Understand STaR-DRO | [Method and objectives](method.md) |
| Define a structured prediction prompt | [Six-module template](../prompts/structured_prediction.md) |
| Inspect our clinical task instruction | [EPPC Miner prompt](../prompts/eppc_miner.md) |
| Build model-ready records | [Dataset preparation](data_format.md#step-by-step-dataset-preparation) |
| Choose group structure and attribution | [Grouping](grouping.md) |
| Set training parameters | [Configuration](configuration.md) |
| Run training and evaluation | [Experiment runbook](quickstart.md) |
| Inspect outputs or resume a run | [Usage and checkpoints](usage.md) |
| Interpret evaluation metrics | [Evaluation](evaluation.md) |
| Inspect method components in code | [Implementation guide](paper_alignment.md) |
| Verify the software | [Reproducibility](reproducibility.md) |

Keep each experiment's completed prompt, ontology, configuration, model revision, split hashes, and random seeds with its saved run artifacts. This makes group-wise behavior and downstream extraction results easy to inspect and compare.
