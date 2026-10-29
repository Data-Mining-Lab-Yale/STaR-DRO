# Group construction and attribution

STaR-DRO separates group construction, the update signal, and the granularity at which multipliers act. We support five grouping strategies and both annotation and sample attribution, allowing researchers to match robust supervision to the semantic and structural difficulty of their task.

| `grouping.groupby` | Annotation group key | Example config |
| --- | --- | --- |
| `code` | Code | [reference.json](../configs/reference.json) |
| `subcode` | Sub-code name, shared across parents when the name repeats | [subcode_annotation.json](../configs/subcode_annotation.json) |
| `code_subcode` | Aligned [Code, Sub-code] pair | [code_subcode_annotation.json](../configs/code_subcode_annotation.json) |
| `num_annotations` | NA_k: total annotation objects in this record | [count_sample.json](../configs/count_sample.json) |
| `code_subcode_num_annotations` | [Code, Sub-code, k] | [code_subcode_count_annotation.json](../configs/code_subcode_count_annotation.json) |

Notebook names `num_span` and `code_subcode_num_span` are accepted aliases. Count means annotation objects, including repeated evidence strings, not unique spans. Joint keys use JSON arrays serialized as strings to avoid delimiter collisions. They use aligned annotation pairs, never Cartesian products of labels.

Group names are sorted and fitted on **training only**. `min_count: 1` preserves every observed group. Higher thresholds pool groups supported by fewer than that many training **records** into `__OTHER__`. Repeated annotations count once per record for the threshold. No unused fallback group is appended: an empty group would change G, the uniform baseline, and the multiplier geometry. Validation reuses the frozen inventory. Unseen groups fail explicitly unless training created a pool. Use `validate-data --config CONFIG` to check before training.

## The same completion, two attribution modes

The nonclinical [README source record](../README.md#2-organize-annotated-source-records) has Topic/Schedule and Courtesy/Thanks objects. Its Code membership set is {Topic, Courtesy}; under count grouping both objects belong to NA_2.

| Operation | `granularity: annotation` | `granularity: sample` |
| --- | --- | --- |
| Observation | One annotation object | One complete record |
| Update signal | Mean CE on object tokens | Mean CE on all completion tokens |
| Memberships | One per annotation | Distinct groups represented in the record |
| Estimator contribution | One observation to its group | Loss/ν and mass 1/ν to each group |
| Applied multiplier | Annotation’s group multiplier | Arithmetic mean of membership multipliers |
| Objective | Weighted token CE / weighted token mass | Weighted sample-mean CE / sum of sample weights |
| Wrapper/separators | Neutral weight 1 | Receive the sample weight |

If Topic receives multiplier 3 and Courtesy receives 1, annotation mode weights the Topic object by 3 and Courtesy by 1; the outer results wrapper and between-object separators retain weight 1. Sample mode averages to 2 and weights the whole record by 2 relative to other records. In a batch containing only this record, sample weighting cancels in the normalized objective; annotation weighting can still change its internal gradient allocation.

An object's mask covers its complete JSON representation: keys, label values, evidence, and local punctuation. Fast-tokenizer offsets align tokens with causal shifting. A token intersecting multiple annotation masks receives the mean multiplier. Uncovered completion and end-of-turn tokens retain neutral weight; prompt and padding tokens are masked.

Empty annotation lists retain completion supervision. Under semantic groupings they have no membership and do not update the adversary. Count grouping assigns NA_0 in sample mode; annotation mode has no observation on those records. These empty-case conventions extend the paper’s annotated setting explicitly.

## Interpreting the grouping comparison

We obtain the highest Code F1 with Code/annotation grouping (79.98) and the highest sample-mode Span F1 with annotation count (91.82) on Llama-3.2-3B-Instruct. Sub-code/annotation achieves 66.12 Sub-code F1, compared with Code/annotation's 66.03. These results support Code/annotation as a strong default while showing competitive alternatives across semantic and complexity-based groups. [Grouping results](results.md).

Set `groupby` and `granularity` independently. Example configurations inherit unspecified reference defaults. As G grows, begin with a conservative effective dual step `(alpha-1)*eta` and monitor entropy, effective group count, and active support to maintain selective, stable reweighting.
