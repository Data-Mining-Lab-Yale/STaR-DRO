# STaR-DRO method

STaR-DRO maintains a distribution over predefined groups to identify persistent relative difficulty, then maps that distribution into bounded extra emphasis on hard groups. The simplex weights are adversarial state; the optimizer-facing multipliers have a neutral lower bound of one.

For alpha > 1, the Tsallis mirror map on the group simplex is

$$
\Psi_\alpha(q)=\frac{1}{\alpha(\alpha-1)}\sum_{g=1}^{G}q_g^\alpha+I_{\Delta_G}(q).
$$

The adversary takes the mirror-ascent step

$$
q_{t+1}=\operatorname*{arg\,max}_{q\in\Delta_G}
\left\{\eta_t\langle a_t,q\rangle-D_{\Psi_\alpha}(q\Vert q_t)\right\}.
$$

The sections below give the executable recurrence for Algorithm 1, including the distinction between adversarial mass q and training multipliers m. Group construction is a task-level choice; [grouping.md](grouping.md) describes the five structured-output constructions used in the manuscript.

## 1. Estimate group difficulty

For observation i with loss u_i and unique group memberships S_i, add u_i/|S_i| and 1/|S_i| to each participating group's loss sum and effective observation mass. Divide the accumulated sum by mass to obtain the minibatch group mean. This overlap correction prevents an observation with many memberships from contributing more total mass solely because it belongs to multiple groups.

In annotation mode, each JSON object is one observation in one group under the configured grouping. Its update signal is detached mean CE over that object’s tokens. Sample mode uses mean completion CE and distinct memberships, with overlap correction. Grouping and attribution are independent; see [grouping.md](grouping.md).

## 2. Preserve a stateful EMA

On the first observation of a group, set its EMA to the estimated mean. On subsequent observations:

```text
L[t,g] = (1-rho[t]) * L[t-1,g] + rho[t] * estimated_loss[t,g]
```

Absent groups retain their EMA. State initializes once and persists across epochs. The trainer keeps one adversary for the continuous run; validation does not update it.

## 3. Rescale the ascent signal

For present groups P, let n_g be their contributing observation counts:

```text
s = sum(n_g * L_g for g in P) / sum(n_g for g in P)
a_g = L_g / max(s, 1e-6)       if g is present
a_g = 0                       otherwise
```

With `hardness_scale=uniform`, replace s with the unweighted mean of EMA losses over present groups; this is the notebook157 option. The default `count` retains the equation above. This drives ascent with relative difficulty. Above the numerical floor, the count-weighted mean of a over present groups is one. In sample multi-membership use, the estimator uses overlap-corrected mass, while the scale uses contributing observation counts. In the annotation trainer those counts coincide.

## 4. Sparse Tsallis mirror ascent

For alpha > 1 and step size eta:

```text
u_g = q_g**(alpha-1) + (alpha-1) * eta * a_g
q_next_g = max(u_g - lambda, 0)**(1/(alpha-1))
```

Bisection finds lambda so q_next sums to one. Thresholding permits exactly zero adversarial mass. Projection includes all groups; an absent group's ascent is zero but its q can change because the simplex threshold is shared. The core uses Python double precision and centers dual coordinates before bisection.

## 5. Bounded excess-only multipliers

For G > 1:

```text
excess_g = clip((G*q_next_g - 1)/(G-1), 0, 1)
m_g = 1 + (U-1) * excess_g**gamma       if present
m_g = 1                                 otherwise
```

Thus m lies in [1,U]. Groups with q at or below uniform receive one; full adversarial concentration gives U. Gamma controls emphasis at intermediate excess. The G=1 case is explicitly neutral. The multiplier reflects persistent relative group difficulty.

## 6. Annotation-level robust objective

Route each annotation group's multiplier to the tokens of that annotation's complete JSON object. Uncovered completion tokens receive one. Overlapping token masks use the mean multiplier. If c_ij is token CE and r_ij marks a valid completion target, the loss is:

```text
loss = sum(d_ij * c_ij) / sum(d_ij * r_ij)
```

Prompt and padding targets are excluded. Gradients pass through CE, not through q, EMA, or the multiplier computation. The annotation-mean update signal and weighted-token denominator define the STaR-DRO objective.

## Implementation map

| Definition in STaR-DRO | Code |
| --- | --- |
| Algorithm 1: overlap-corrected statistics, EMA, scaling, mirror step, multipliers | `StarDRO.update` in `src/star_dro/core.py` |
| Thresholded simplex mapback | `backmap` in `src/star_dro/core.py` |
| Structured-completion annotation attribution | `encode_record` in `src/star_dro/data.py` |
| Annotation-level objective | `annotation_loss` in `src/star_dro/torch_loss.py` |
| Continuous epoch-2 activation and raw validation diagnostics | `src/star_dro/training.py` |

The generative trainer implements all five group constructions and both attribution modes described in the main paper and supplement S4–S5. The same adversary interface supports integration with other supervised losses and task-specific memberships.

## 7. Sample-level robust objective

For memberships S_i, set `m_i = mean(m_g for g in S_i)`. Let `mean_ce_i` be the sample’s mean completion CE. Algorithm 1’s objective is:

```text
loss_sample = sum(m_i * mean_ce_i) / sum(m_i)
```

`sample_loss` implements this sample-normalized objective and sends detached `mean_ce_i` to the persistent adversary. Annotation mode uses weighted-token normalization. The initial SFT stage and SFT baseline use completion-token CE. These denominators determine how completion length affects each objective.

The core accepts arbitrary supervised-task memberships without depending on JSON, tokenization, or a language model:

```python
from star_dro import Config, StarDRO

adversary = StarDRO(Config(num_groups=3))  # create once per run
group_multipliers = adversary.update(
    losses=[0.4, 1.2],
    memberships=[[0, 1], [2]],
    progress=0.5,
)
sample_multipliers = [sum(group_multipliers[g] for g in s) / len(s)
                      for s in [[0, 1], [2]]]
```

For model integration, use `torch_loss.sample_loss` or `annotation_loss` to preserve differentiation through CE and the documented denominators. q and EMA are detached. The [implementation guide](paper_alignment.md) maps the method components to their modules and saved outputs.

## Integration requirements

Construct one StarDRO instance with a fixed ordered group inventory. Pass detached mean completion CE for sample observations or mean annotation CE for annotation observations to StarDRO.update, together with the associated group memberships and training progress. Retain the same state across epochs and update only during training. Use state_dict and from_state_dict for adversary serialization.

The core is device-independent; transferring detached model statistics to it incurs a CPU synchronization per robust minibatch.
