# Memory-model tuning record

Run: `xi_2027_63d_p50` on the candidate-only Xi graph.

The model comparison uses hard negatives and the same chronological test split.
Node features for the tuned runs are built from the fit-period trades only.
The optional pair-history head uses causal pair count, 5-minute/1-hour/24-hour
decayed counts, a seen flag and pair age.

## Final fixed test comparison

Five seeds were fixed before this comparison. The test window was not used to
choose another configuration afterward.

| variant | MRR mean | MRR SD | AUC mean | recency MRR | MRR gap vs recency |
|---|---:|---:|---:|---:|---:|
| plain, leak-free | 0.341 | 0.019 | 0.689 | 0.433 | −0.093 |
| tuned, pair history + memory | 0.586 | 0.015 | 0.865 | 0.433 | +0.153 |
| memory-off pair history | 0.182 | 0.033 | 0.563 | 0.433 | −0.252 |

The tuned model improves held-out ranking on this Xi artifact. The memory-off
ablation does not reproduce that performance, so this run does not support the
claim that node memory is irrelevant. The improvement should be described as
coming from the combined memory model and causal pair-history head unless a
future matched ablation changes that result.

Artifacts:

- `data/artifacts/xi_2027_63d_p50/final_test_results.csv`
- `data/artifacts/xi_2027_63d_p50/final_test_results_summary.csv`
- `data/artifacts/xi_2027_63d_p50/tuning_results.csv`
- `data/artifacts/xi_2027_63d_p50/tuning_results_summary.csv`

The final test command was `run_final_test.py`; the validation-only command was
`tune_tgnn.py`. Neither command changes the funding-validation evidence chain.
