# Method implementation guide

Our implementation separates the persistent group adversary, group construction, and the supervised objective. This organization supports task-specific group definitions and either sample-level or annotation-level attribution through the same robust update.

| Method component | Implementation | Inspectable state or output |
| --- | --- | --- |
| Overlap-corrected group estimates | `core.StarDRO.update` | Group loss estimates and observation counts |
| Stateful EMA | `core.StarDRO.update` | Persistent group-loss history |
| Relative hardness signal | `core.StarDRO.update` | Present-group scale and ascent signals |
| Tsallis ascent and sparse mapback | `core.backmap` | Adversarial distribution q |
| Bounded excess-only multipliers | `core.StarDRO.update` | Applied group multipliers in [1,U] |
| Five group constructions | `grouping.GroupInventory` | Frozen ordered inventory and optional rare-group mapping |
| Sample objective | `torch_loss.sample_loss` | Membership-averaged weights and sample-normalized CE |
| Annotation objective | `torch_loss.annotation_loss` | Token-routed weights and weighted-token CE |
| Prompt assembly | `prompting.build_record` | Complete model input in `messages[0].content` |
| Completion alignment | `data.encode_record` | Causal labels and annotation masks |
| Continuous optimization | `training.train` | Epoch-2 activation, checkpoints, optimizer and adversary state |
| Adapter merging | `merging.merge` | Selected adapter merged into standalone safetensors, with provenance |
| Test generation | `inference.predict` | Greedy vLLM outputs and generation metadata |
| Structured scoring | `scoring.score`, `metrics.evaluate` | Ontology-driven Code, Sub-code, joint-pair, Span and validity metrics |

The adversary API accepts detached losses and meaningful group memberships for supervised tasks. Structured generation adds the task instruction, chat formatting, annotation serialization, and token attribution layers. [Method equations](method.md) · [Grouping conventions](grouping.md) · [Dataset preparation](data_format.md).

## Experiment configuration

Our reference configuration uses Code groups with annotation attribution, three continuous epochs, LoRA on attention and MLP projections, AdamW, and robust activation at epoch 2. EMA follows a cosine schedule from 0.3 to 0.03. The reference point uses alpha 1.2, eta 0.0006, U 15, and gamma 0.75. [All parameter definitions](configuration.md).

The training entry point runs on one process/device. Checkpoints save the model or adapter, tokenizer, optimizer, scheduler, random state, and persistent adversary. Prediction temporarily merges the selected adapter (final epoch by default) and uses greedy vLLM generation. Training remains single-device; vLLM tensor parallelism is configurable independently. The [runbook](quickstart.md) connects these stages and describes the experiment artifacts.

The optional `standard_dro.StandardDRO` baseline follows the provided notebook144 exponential group update and observed-annotation count correction. It is separate from the STaR-DRO equations above. `configs/standard_dro.json` routes it through the same trainer and evaluation pipeline.
