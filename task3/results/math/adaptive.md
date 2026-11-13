## math: adaptive sampling (quality score K)

### Development selection (200 questions; fixed-32 SC 82.50%, threshold 81.50%)

| policy | accuracy | mean_cost | mean_rollouts |
|---|---|---|---|
| quality: tau=0.5, gamma=0.0 | 81.55 | 5503.25 | 7.26 |
| quality: tau=0.5, gamma=0.1 | 81.70 | 5783.82 | 7.68 |
| quality: tau=0.5, gamma=0.25 | 81.85 | 6124.86 | 8.12 |
| agreement: tau=0.625 | 82.30 | 7139.23 | 9.26 |
| quality: tau=0.625, gamma=0.0 | 82.30 | 7193.02 | 9.34 |
| quality: tau=0.5, gamma=0.5 | 82.30 | 7274.97 | 9.70 |
| quality: tau=0.625, gamma=0.1 | 82.40 | 7336.27 | 9.60 |
| quality: tau=0.625, gamma=0.25 | 82.40 | 7370.89 | 9.67 |

Frozen: agreement: tau=0.625, gamma=None; quality: tau=0.5, gamma=0.0

### Held-out evaluation (800 questions, mean over 10 orders)

| policy | accuracy | mean_cost | mean_rollouts | stop@4 (%) | stop@8 (%) | stop@16 (%) | stop@32 (%) | fixed_SC_acc_at_same_cost |
|---|---|---|---|---|---|---|---|---|
| agreement (tau=0.625) | 73.74 | 10112.18 | 11.79 | 66.71 | 5.42 | 1.41 | 26.45 | 72.99 |
| quality (tau=0.5, gamma=0.0) | 73.08 | 7565.41 | 8.86 | 77.00 | 5.66 | 1.36 | 15.97 | 72.25 |

Fixed-N self-consistency (held-out):

| N | accuracy | mean_cost |
|---|---|---|
| 1 | 63.59 | 667.38 |
| 2 | 63.59 | 1334.99 |
| 4 | 69.00 | 2671.95 |
| 8 | 71.61 | 5342.03 |
| 16 | 73.15 | 10669.20 |
| 32 | 74.03 | 21347.02 |

### Paired bootstrap accuracy differences, held-out (pp)

| a | b | diff_pp | ci_lo_pp | ci_hi_pp |
|---|---|---|---|---|
| adaptive-agreement | SC@32 | -0.29 | -0.68 | 0.08 |
| adaptive-quality | SC@32 | -0.95 | -1.64 | -0.29 |
| adaptive-agreement | SC@8 | 2.12 | 1.22 | 3.09 |
| adaptive-agreement | SC@16 | 0.59 | -0.07 | 1.20 |
| adaptive-quality | SC@8 | 1.46 | 0.81 | 2.20 |
| adaptive-quality | SC@16 | -0.07 | -0.60 | 0.49 |
| adaptive-quality | adaptive-agreement | -0.66 | -1.11 | -0.24 |

### Early stops (held-out runs = question x order)

| policy | early_stops_pct | early_stop_accuracy | incorrect_early_stops_pct_of_runs | incorrect_early_stops_recovered_by_sc32_pct | n_incorrect_early_stops | questions_with_incorrect_early_stop | correct_early_stops_lost_by_sc32_pct |
|---|---|---|---|---|---|---|---|
| agreement | 73.55 | 88.49 | 8.46 | 8.86 | 677 | 164 | 0.71 |
| quality | 84.03 | 82.13 | 15.01 | 13.66 | 1201 | 267 | 1.59 |

### Retrospective: stopping vs fraction of correct candidates in the full pool

| policy | pool_bin | runs | mean_stop_n | early_stop_pct | accuracy | sc32_accuracy | mean_cost |
|---|---|---|---|---|---|---|---|
| agreement | 0 | 690 | 23.12 | 32.90 | 0.00 | 0.00 | 24071.42 |
| agreement | (0,.25] | 1420 | 25.42 | 24.51 | 14.37 | 13.87 | 24653.65 |
| agreement | (.25,.5] | 780 | 22.24 | 36.92 | 77.31 | 78.85 | 19552.99 |
| agreement | (.5,.75] | 740 | 9.11 | 88.11 | 97.97 | 100.00 | 6696.23 |
| agreement | (.75,1) | 1750 | 4.41 | 99.94 | 99.83 | 100.00 | 2620.36 |
| agreement | 1 | 2620 | 4.00 | 100.00 | 100.00 | 100.00 | 1712.91 |
| quality | 0 | 690 | 18.07 | 51.45 | 0.00 | 0.00 | 19216.81 |
| quality | (0,.25] | 1420 | 19.01 | 48.10 | 15.49 | 13.87 | 18960.89 |
| quality | (.25,.5] | 780 | 12.23 | 74.87 | 71.67 | 78.85 | 10980.82 |
| quality | (.5,.75] | 740 | 5.48 | 98.65 | 95.14 | 100.00 | 3938.15 |
| quality | (.75,1) | 1750 | 4.21 | 100.00 | 99.60 | 100.00 | 2498.37 |
| quality | 1 | 2620 | 4.00 | 100.00 | 100.00 | 100.00 | 1712.91 |

### Stopping vs leader share at n=4

| policy | leader_share_at_n=4 | runs | mean_stop_n | accuracy |
|---|---|---|---|---|
| agreement | 0.25 | 1071 | 31.40 | 30.91 |
| agreement | 0.50 | 1570 | 24.62 | 48.15 |
| agreement | 0.75 | 1346 | 4.31 | 71.84 |
| agreement | 1.00 | 4013 | 4.05 | 95.81 |
| quality | 0.25 | 1071 | 29.54 | 30.63 |
| quality | 0.50 | 1570 | 10.77 | 44.90 |
| quality | 0.75 | 1346 | 4.54 | 71.92 |
| quality | 1.00 | 4013 | 4.05 | 95.81 |
