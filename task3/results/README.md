# Task 3 results (machine-readable outputs and headline tables)

Reproduce: `python -m task3.offline_data`, `task3.offline`, `task3.signals`, `task3.adaptive`, `task3.analysis` (all with `--config task3/configs/offline.yaml`); Step 4 on Kaggle via `notebooks/task3_live.ipynb` (outputs in `kaggle_outputs/task3_live/`). All numbers are on the 800-question held-out split unless stated, averaged over 10 seeded orders.

Per dataset: `summary.md` (Steps 1-2), `signals.md` (Step 3), `adaptive.md` (Step 5), `allocation.csv` (Step 6), `data_checks.json`; figures in `figures/` and `<ds>/figures/`; cases in `cases.md`.

## MATH (Qwen2.5-3B-Instruct)

Scorer: math-verify 0.9.0. Rollout-level accuracy 66.32%, 523 capped, 521 null extractions (kept as NO_ANSWER).

### Compute allocation (self-consistency)

| N | oracle | sc_accuracy | selection_gap | distinct_groups | correct_share_in_sc_selection_failures_pct | new_groups | new_candidates_repeating_pct | oracle_gain_pct |
|---|---|---|---|---|---|---|---|---|
| 1 | 63.59 | 63.59 | 0.00 | 1.00 | nan | nan | nan | nan |
| 2 | 72.34 | 63.59 | 8.75 | 1.38 | 50.00 | 0.38 | 62.26 | 8.75 |
| 4 | 79.00 | 69.00 | 10.00 | 1.93 | 27.94 | 0.55 | 72.39 | 6.66 |
| 8 | 84.39 | 71.61 | 12.78 | 2.77 | 19.62 | 0.84 | 79.08 | 5.39 |
| 16 | 88.28 | 73.15 | 15.12 | 4.02 | 15.50 | 1.25 | 84.34 | 3.89 |
| 32 | 91.38 | 74.02 | 17.35 | 5.90 | 13.44 | 1.89 | 88.22 | 3.10 |

### Adaptive policies (frozen on dev)

| policy | accuracy | mean_cost | mean_rollouts | fixed_SC_acc_at_same_cost |
|---|---|---|---|---|
| agreement (tau=0.625) | 73.74 | 10112.18 | 11.79 | 72.99 |
| quality (tau=0.5, gamma=0.0) | 73.08 | 7565.41 | 8.86 | 72.25 |

## MathVista (Qwen2.5-VL-3B-Instruct)

Scorer: mathvista-official-normalize-v1 (+lowercase letter, RAW groups for unparseable). Rollout-level accuracy 53.64%, 487 capped, 3032 null extractions (kept as NO_ANSWER).

### Compute allocation (self-consistency)

| N | oracle | sc_accuracy | selection_gap | distinct_groups | correct_share_in_sc_selection_failures_pct | new_groups | new_candidates_repeating_pct | oracle_gain_pct |
|---|---|---|---|---|---|---|---|---|
| 1 | 53.86 | 53.86 | 0.00 | 1.00 | nan | nan | nan | nan |
| 2 | 66.75 | 53.86 | 12.89 | 1.45 | 50.00 | 0.45 | 54.86 | 12.89 |
| 4 | 76.15 | 58.70 | 17.45 | 2.04 | 30.01 | 0.59 | 70.71 | 9.40 |
| 8 | 82.90 | 60.79 | 22.11 | 2.78 | 24.40 | 0.74 | 81.40 | 6.75 |
| 16 | 86.96 | 62.46 | 24.50 | 3.72 | 21.31 | 0.94 | 88.21 | 4.06 |
| 32 | 89.50 | 63.20 | 26.30 | 4.90 | 20.05 | 1.18 | 92.65 | 2.54 |

### Adaptive policies (frozen on dev)

| policy | accuracy | mean_cost | mean_rollouts | fixed_SC_acc_at_same_cost |
|---|---|---|---|---|
| agreement (tau=0.625) | 63.06 | 4528.44 | 13.46 | 62.48 |
| quality (tau=0.5, gamma=0.25) | 63.14 | 3967.33 | 12.15 | 62.10 |

## Step 4 live strategies

### Live MATH strategies (50 questions, same 3,072-token output allowance; math-verify 0.9.0, last-literal-boxed-balanced-v2 (supplied-export rule))

| strategy | n_questions | accuracy_pct | final_no_answer_pct | responses_extracted_pct | responses_truncated_pct | mean_input_tokens | mean_output_tokens | output_allowance | mean_time_s | total_time_min | peak_mem_gb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 50 | 66.00 | 4.00 | 96.00 | 4.00 | 108.38 | 738.26 | 3072 | 34.67 | 28.89 | 6.09 |
| parallel | 50 | 62.00 | 10.00 | 86.67 | 14.00 | 325.14 | 1877.58 | 3072 | 86.25 | 71.88 | 5.94 |
| refine | 50 | 62.00 | 0.00 | 95.00 | 5.00 | 870.26 | 1141.72 | 3072 | 53.52 | 44.60 | 6.17 |

Shared initial response accuracy 62.0%; parallel oracle coverage (any of 3 correct) 68.0%; revised answer differs from the initial one in 10.0% of questions.

### Refinement transitions

| transition (initial -> revised) | questions |
|---|---|
| W->C | 0 |
| C->W | 0 |
| C->C | 31 |
| W->W | 19 |

### Paired accuracy differences (1,000 question bootstrap resamples)

| comparison | accuracy diff (pp) | 95% CI |
|---|---|---|
| parallel_minus_single | -4.00 | [-16.0, 8.0] |
| refine_minus_single | -4.00 | [-14.0, 8.0] |
| refine_minus_parallel | 0.00 | [-6.0, 6.0] |
| refine_minus_initial | 0.00 | [0.0, 0.0] |
| parallel_minus_initial | 0.00 | [-6.0, 6.0] |
