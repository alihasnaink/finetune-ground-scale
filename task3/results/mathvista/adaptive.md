## mathvista: adaptive sampling (quality score Z)

### Development selection (200 questions; fixed-32 SC 64.25%, threshold 63.25%)

| policy | accuracy | mean_cost | mean_rollouts |
|---|---|---|---|
| quality: tau=0.5, gamma=0.25 | 63.30 | 4636.41 | 12.59 |
| agreement: tau=0.625 | 63.65 | 5255.57 | 14.04 |
| quality: tau=0.625, gamma=0.0 | 63.65 | 5319.35 | 14.22 |
| quality: tau=0.625, gamma=0.1 | 63.65 | 5470.29 | 14.71 |
| quality: tau=0.625, gamma=0.25 | 63.60 | 5536.22 | 14.97 |
| quality: tau=0.5, gamma=0.5 | 63.90 | 5610.23 | 15.51 |
| agreement: tau=0.75 | 63.80 | 5689.42 | 15.40 |
| quality: tau=0.75, gamma=0.0 | 63.90 | 5765.65 | 15.61 |

Frozen: agreement: tau=0.625, gamma=None; quality: tau=0.5, gamma=0.25

### Held-out evaluation (800 questions, mean over 10 orders)

| policy | accuracy | mean_cost | mean_rollouts | stop@4 (%) | stop@8 (%) | stop@16 (%) | stop@32 (%) | fixed_SC_acc_at_same_cost |
|---|---|---|---|---|---|---|---|---|
| agreement (tau=0.625) | 63.06 | 4528.44 | 13.46 | 58.03 | 8.15 | 2.11 | 31.71 | 62.48 |
| quality (tau=0.5, gamma=0.25) | 63.14 | 3967.33 | 12.15 | 59.67 | 10.54 | 3.80 | 25.99 | 62.10 |

Fixed-N self-consistency (held-out):

| N | accuracy | mean_cost |
|---|---|---|
| 1 | 53.86 | 274.07 |
| 2 | 53.86 | 550.94 |
| 4 | 58.70 | 1101.01 |
| 8 | 60.79 | 2219.25 |
| 16 | 62.46 | 4444.67 |
| 32 | 63.20 | 8897.76 |

### Paired bootstrap accuracy differences, held-out (pp)

| a | b | diff_pp | ci_lo_pp | ci_hi_pp |
|---|---|---|---|---|
| adaptive-agreement | SC@32 | -0.14 | -0.74 | 0.46 |
| adaptive-quality | SC@32 | -0.06 | -0.80 | 0.79 |
| adaptive-agreement | SC@8 | 2.28 | 1.34 | 3.23 |
| adaptive-agreement | SC@16 | 0.60 | -0.20 | 1.36 |
| adaptive-quality | SC@8 | 2.35 | 1.52 | 3.20 |
| adaptive-quality | SC@16 | 0.68 | -0.05 | 1.40 |
| adaptive-quality | adaptive-agreement | 0.08 | -0.39 | 0.54 |

### Early stops (held-out runs = question x order)

| policy | early_stops_pct | early_stop_accuracy | incorrect_early_stops_pct_of_runs | incorrect_early_stops_recovered_by_sc32_pct | n_incorrect_early_stops | questions_with_incorrect_early_stop | correct_early_stops_lost_by_sc32_pct |
|---|---|---|---|---|---|---|---|
| agreement | 68.29 | 79.37 | 14.09 | 11.80 | 1127 | 259 | 2.81 |
| quality | 74.01 | 76.79 | 17.18 | 13.10 | 1374 | 319 | 3.85 |

### Retrospective: stopping vs fraction of correct candidates in the full pool

| policy | pool_bin | runs | mean_stop_n | early_stop_pct | accuracy | sc32_accuracy | mean_cost |
|---|---|---|---|---|---|---|---|
| agreement | 0 | 840 | 25.74 | 23.33 | 0.00 | 0.00 | 8168.22 |
| agreement | (0,.25] | 1560 | 22.39 | 36.22 | 6.99 | 5.90 | 9052.21 |
| agreement | (.25,.5] | 1360 | 19.19 | 48.24 | 56.69 | 53.24 | 6877.22 |
| agreement | (.5,.75] | 1320 | 9.71 | 85.30 | 94.39 | 100.00 | 2649.78 |
| agreement | (.75,1) | 2100 | 4.25 | 100.00 | 99.95 | 100.00 | 899.94 |
| agreement | 1 | 820 | 4.00 | 100.00 | 100.00 | 100.00 | 614.87 |
| quality | 0 | 840 | 23.70 | 30.95 | 0.00 | 0.00 | 7470.38 |
| quality | (0,.25] | 1560 | 19.68 | 46.86 | 8.85 | 5.90 | 7897.63 |
| quality | (.25,.5] | 1360 | 16.01 | 61.62 | 55.51 | 53.24 | 5534.70 |
| quality | (.5,.75] | 1320 | 9.12 | 89.09 | 93.79 | 100.00 | 2379.60 |
| quality | (.75,1) | 2100 | 4.54 | 99.81 | 100.00 | 100.00 | 938.47 |
| quality | 1 | 820 | 4.00 | 100.00 | 100.00 | 100.00 | 614.87 |

### Stopping vs leader share at n=4

| policy | leader_share_at_n=4 | runs | mean_stop_n | accuracy |
|---|---|---|---|---|
| agreement | 0.25 | 811 | 31.46 | 19.85 |
| agreement | 0.50 | 2226 | 24.08 | 46.99 |
| agreement | 0.75 | 2128 | 6.71 | 65.23 |
| agreement | 1.00 | 2835 | 5.04 | 86.42 |
| quality | 0.25 | 811 | 30.30 | 19.98 |
| quality | 0.50 | 2226 | 17.61 | 46.90 |
| quality | 0.75 | 2128 | 9.01 | 65.55 |
| quality | 1.00 | 2835 | 5.04 | 86.42 |
