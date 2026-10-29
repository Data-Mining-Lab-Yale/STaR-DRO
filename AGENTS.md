# STaR-DRO repository guide for coding assistants

Read README.md, docs/quickstart.md, docs/data_format.md and configs/reference.json before changing or running an experiment. Use the existing CLI in src/star_dro/cli.py and scripts/run_experiment.py; run source-checkout commands through scripts/star_dro.py when the package is not installed.

Data paths, model access, task prompts, ontology and execution hardware must come from the operator's authorized inputs. If required inputs are missing, identify them instead of inventing records or reporting unexecuted results. The README documents one nonclinical source-format record. Use the Markdown prompt templates and scripts/build_records.py to assemble completed task instructions into model-visible records.

Use prepare-data for explicit normalization/export, validate-data before training, train for continuous optimization, merge for explicit standalone export, predict for temporary merging and vLLM inference from the selected checkpoint, and evaluate (with the task ontology) for the documented metrics. JSON inputs are not automatically written to Arrow. Native Arrow paths use the datasets dependency. Keep training, validation and test roles separate; test metrics must not select model checkpoints.

Read docs/paper_alignment.md and docs/grouping.md for the distinction between the general method, group construction and attribution granularity. Preserve persistent EMA, the configured count/uniform scale, sparse Tsallis mapback, bounded excess-only multipliers, and both documented objectives. Fit one ordered group inventory on training and retain it across validation and resumption. Run numerical/interface/gradient/integration tests and release checks after changes; report skips accurately. Never claim clinical replication from fictional unit-test model execution.

Use new output paths for new runs. Epoch resumption restores the documented training state and requires the same experiment identity. Never include restricted data, credentials or generated private-data artifacts in a release archive. Review files with scripts/package_release.py before packaging.

Keep model stages isolated in fresh processes with guarded __main__ entry points. Preserve adapter checkpoints; delete automatic temporary merged weights after prediction. Tests with mocked engines are interface checks, not GPU validation. Follow docs/quickstart.md for an actual checkpoint-only vLLM smoke test.

Standard DRO is a separate notebook144 baseline with exponential q updates and annotation-count correction; do not substitute STaR-DRO with alpha near 1. Default split validation remains strict. The --allow-split-overlap exception is for explicitly documented pipeline smoke tests only, and must remain visible in reports and run identity. Never deduplicate or resplit a supplied dataset silently.
