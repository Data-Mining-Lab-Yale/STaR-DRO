# Configuration

`configs/reference.json` defines the STaR-DRO reference training recipe. The CLI fills omitted fields from the same defaults used by the reference configuration and rejects unknown keys.

## STaR-DRO parameters

| JSON key under `star_dro` | Reference value | Meaning | Experimental settings |
| --- | --- | --- | --- |
| `alpha` | 1.2 | Tsallis order; must exceed one | Reported range 1.1–1.3 |
| `eta` | 0.0006 | Constant adversarial mirror-ascent step size | Reported range 0.0006–0.003 |
| `rho_start` | 0.3 | Starting EMA update coefficient | Cosine schedule starts at 0.3 |
| `rho_end` | 0.03 | Ending EMA update coefficient | Cosine schedule ends at 0.03 |
| `upper` | 15.0 | Multiplier ceiling U; must exceed one | Reported range 10–30; supplementary diagnostics use 15 as a default |
| `gamma` | 0.75 | Excess multiplier curvature; must be positive | Reported default 0.75 |
| `hardness_scale` | count | Present-group EMA scale: `count` or `uniform` | `uniform` matches the active notebook157 scale |

The number of groups is fitted from training data under the selected grouping. The ordered inventory and any rare-group mapping are saved in run.json.

EMA coefficients use cosine progress over the full continuous training run. The adversary starts updating only at `dro_start_epoch`; consequently, when activation is delayed, the coefficient used on the first robust step is the value of the full-run schedule at that step. EMA and q do not reset at subsequent epoch boundaries. Eta remains constant in this entry point.

Choose eta and U with the number and cardinality of groups in mind. Use the reference configuration as a starting point, monitor group-weight diagnostics, and select task-specific settings on validation data.

`hardness_scale: count` divides by the annotation/observation-count-weighted mean of present-group EMA losses. `hardness_scale: uniform` divides by their unweighted mean. Both leave absent-group signals at zero. Use `configs/uniform_scaling.json` to select uniform scaling without changing other defaults. The selected scale is saved in adversary state and diagnostics.

## Training parameters

| Key under `training` | Reference value | Meaning |
| --- | --- | --- |
| `objective` | stardro | Use stardro, standard_dro or sft; SFT never updates the adversary |
| `epochs` | 3 | Total continuous training epochs |
| `dro_start_epoch` | 2 | One-based first epoch with the selected DRO method |
| `batch_size` | 3 | Records per optimizer update on one device |
| `learning_rate` | 0.0002 | AdamW starting learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `warmup_ratio` | 0.03 | Fraction of optimizer steps used for learning-rate warmup |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `gradient_checkpointing` | true | Memory-saving model activation recomputation |
| `checkpoint_selection` | last | `last` reports the final epoch (manuscript protocol); `validation_ce` reports the lowest validation completion CE epoch |

The learning rate follows a cosine schedule after warmup. Each minibatch produces one optimizer update and one active STaR-DRO update. Gradient checkpointing recomputes model activations, while optimizer, scheduler, and adversary state persist through the continuous run.

## Group construction and attribution

`grouping.groupby` chooses `code`, `subcode`, `code_subcode`, `num_annotations`, or `code_subcode_num_annotations`. `grouping.granularity` independently chooses `annotation` (default) or `sample`. `grouping.min_count` is the minimum training-record membership count, default 1. Groups below a higher threshold are pooled only when needed. [Detailed keys, empty cases, and validation behavior](grouping.md).

Files in configs are partial overrides of shared defaults. Set the group construction, attribution, alpha, eta, U, and regularization for your task; run.json records the complete resolved configuration.

## LoRA parameters

| Key under `lora` | Reference value |
| --- | --- |
| `enabled` | true |
| `rank` | 16 |
| `alpha` | 32 |
| `dropout` | 0.05 |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

These target the notebook's attention and MLP projections. Set `enabled` to false to perform full fine-tuning of a model.

## Runtime options

Use `star-dro train --help` for all flags. `--model` may be a model identifier or local model directory; `--revision` can pin a model/tokenizer commit. Default maximum sequence length is 8192. Default precision is bfloat16 on CUDA. CPU runs use `--device cpu --precision float32`. Model loading uses the standard Hugging Face AutoModel interface.

The training entry point supports single-device LoRA or full fine-tuning. Memory requirements depend on model size, sequence length, batch size, and precision. See the [implementation guide](paper_alignment.md) for the training stages, saved state, and checkpoint-selection policy.

Prediction uses CUDA/vLLM; its independent CLI controls are documented in [the runbook](quickstart.md#vllm-and-merge-settings). CPU flags above apply to training. Automatic merging defaults to CPU and exports the selected inference precision. Neither merge nor vLLM options change STaR-DRO hyperparameters or group-loss calculations.

## Standard DRO baseline

`configs/standard_dro.json` selects `training.objective: standard_dro` with `standard_dro.eta: 0.1`. This is the supplied notebook144 baseline: group mean annotation CE drives `q <- softmax(log(q) + eta*L)`, and present-group annotation multipliers are `q_g / p_hat_g / sum(q_present)`, where `p_hat` is the minibatch annotation-count proportion. Absent groups receive multiplier 1. The baseline has no EMA, Tsallis mapback or STaR-DRO clipping. It uses annotation attribution; sample attribution is rejected for this baseline. Its group construction can still be chosen through the existing grouping settings. The smoke experiment uses Code groups, matching the notebook.

The default three-epoch recipe activates either DRO method at epoch 2. Standard DRO q state persists across active epochs and resumption, and is saved in `standard_dro.json`; STaR-DRO keeps `star_dro.json`. SFT and STaR-DRO numerical defaults are unchanged.
