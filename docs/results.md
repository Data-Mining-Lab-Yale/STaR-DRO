# Experimental results

We evaluate STaR-DRO on EPPC Miner across Llama, Qwen, and Gemma. We report F1 percentages averaged over three runs. Our comparisons examine semantic label prediction, grounded evidence extraction, group-wise validation loss, and grouping granularity.

## SFT, standard DRO, and STaR-DRO

We compare prompt-engineered SFT, standard group DRO, and STaR-DRO under the same task setting.

| Model | Method | Code F1 | Sub-code F1 | Span F1 |
| --- | --- | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct | SFT | 77.86 | 63.21 | 91.23 |
| | Standard DRO | 75.91 | 61.86 | 91.64 |
| | STaR-DRO | 77.70 | 64.45 | 91.18 |
| Llama-3.2-3B-Instruct | SFT | 78.85 | 64.90 | 91.46 |
| | Standard DRO | 76.74 | 63.55 | 91.87 |
| | STaR-DRO | 79.98 | 66.03 | 91.00 |
| Llama-3.1-8B-Instruct | SFT | 78.61 | 66.52 | 91.43 |
| | Standard DRO | 76.94 | 64.31 | 91.73 |
| | STaR-DRO | 79.77 | 66.80 | 91.42 |
| Llama-3.3-70B-Instruct | SFT | 79.24 | 67.78 | 92.38 |
| | Standard DRO | 80.02 | 68.68 | 91.74 |
| | STaR-DRO | 81.47 | 69.30 | 91.73 |

STaR-DRO improves Sub-code F1 at all four Llama scales and Code F1 at three, while maintaining Span F1 of at least 91.00. At 70B, it reaches 81.47 Code F1 and 69.30 Sub-code F1, strengthening the semantic decisions central to hierarchical extraction.

## Grouping and attribution sensitivity

We compare five grouping criteria under annotation and sample attribution on Llama-3.2-3B-Instruct:

| Grouping | Annotation: Code / Sub-code / Span | Sample: Code / Sub-code / Span |
| --- | --- | --- |
| Code | 79.98 / 66.03 / 91.00 | 78.94 / 65.42 / 91.15 |
| Sub-code | 79.14 / 66.12 / 91.28 | 78.85 / 65.04 / 91.35 |
| Number of annotations | 79.21 / 65.53 / 91.52 | 79.46 / 65.46 / 91.82 |
| Code × Sub-code | 79.20 / 65.43 / 91.21 | 78.60 / 65.74 / 91.03 |
| Code × Sub-code × Number of annotations | 78.82 / 65.38 / 91.17 | 79.10 / 65.22 / 91.22 |

Annotation attribution can emphasize a hard object without assigning the same extra weight to an easier object in the same completion. Sample attribution averages membership multipliers and weights the entire example. The comparison supports Code/annotation as a strong default and shows competitive performance across meaningful semantic and complexity-based groups.

## Prompt architecture and cross-family evidence

We report average zero-shot gains across four Llama models of **+15.90 Code, +13.02 Sub-code, and +17.40 Span F1 points** from our modular prompt. Across eight Llama, Qwen, and Gemma models, STaR-DRO lowers validation cross-entropy in **70 of 72 model–group comparisons** against SFT.

Group-wise validation loss captures token-level improvements throughout the structured completion. Code, Sub-code, and Span F1 capture the resulting discrete decisions on the test set. Together, they characterize how robust optimization affects both structured generation and downstream extraction.
