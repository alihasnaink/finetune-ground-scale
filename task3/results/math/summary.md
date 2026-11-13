## math: held-out evaluation split (800 questions), mean over 10 orders

### Accuracy (%)

| method | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|---|
| FirstAns | 63.59 | 63.59 | 63.59 | 63.59 | 63.59 | 63.59 |
| SC | 63.59 | 63.59 | 69.00 | 71.61 | 73.15 | 74.03 |
| MaxLogP | 63.59 | 65.86 | 66.83 | 67.40 | 66.39 | 64.00 |
| MaxSelfCert | 63.59 | 66.47 | 68.61 | 70.22 | 70.61 | 70.62 |
| SelfCert-RWC | 63.59 | 66.47 | 69.66 | 72.78 | 74.51 | 75.59 |
| Shortest | 63.59 | 64.80 | 64.86 | 64.06 | 63.14 | 62.38 |

### Opportunity and failure decomposition (self-consistency selector)

| N | oracle | sc_acc | sampling_fail | selection_fail | distinct_groups | distinct_answered_groups | sc_cost | firstans_acc | firstans_cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 63.59 | 63.59 | 36.41 | 0.00 | 1.00 | 0.98 | 667.38 | 63.59 | 667.38 |
| 2 | 72.34 | 63.59 | 27.66 | 8.75 | 1.38 | 1.35 | 1334.99 | 63.59 | 667.38 |
| 4 | 79.00 | 69.00 | 21.00 | 10.00 | 1.93 | 1.88 | 2671.95 | 63.59 | 667.38 |
| 8 | 84.39 | 71.61 | 15.61 | 12.78 | 2.77 | 2.68 | 5342.03 | 63.59 | 667.38 |
| 16 | 88.28 | 73.15 | 11.72 | 15.12 | 4.02 | 3.89 | 10669.20 | 63.59 | 667.38 |
| 32 | 91.38 | 74.03 | 8.62 | 17.35 | 5.90 | 5.72 | 21347.02 | 63.59 | 667.38 |

### N = 32: sampling vs selection failures by method (%)

| method | accuracy | oracle | sampling_fail | selection_fail | selection_gap |
|---|---|---|---|---|---|
| FirstAns | 63.59 | 91.38 | 8.62 | 27.79 | 27.79 |
| SC | 74.03 | 91.38 | 8.62 | 17.35 | 17.35 |
| MaxLogP | 64.00 | 91.38 | 8.62 | 27.38 | 27.38 |
| MaxSelfCert | 70.62 | 91.38 | 8.62 | 20.75 | 20.75 |
| SelfCert-RWC | 75.59 | 91.38 | 8.62 | 15.79 | 15.79 |
| Shortest | 62.38 | 91.38 | 8.62 | 29.00 | 29.00 |

### Paired accuracy differences, held-out (pp, 95% bootstrap CI)

| N | a | b | diff_pp | ci_lo_pp | ci_hi_pp |
|---|---|---|---|---|---|
| 4 | SC | FirstAns | 5.41 | 4.59 | 6.21 |
| 4 | MaxSelfCert | MaxLogP | 1.79 | 0.86 | 2.74 |
| 4 | SelfCert-RWC | SC | 0.66 | 0.05 | 1.30 |
| 4 | SelfCert-RWC | MaxSelfCert | 1.05 | 0.56 | 1.53 |
| 4 | MaxSelfCert | SC | -0.39 | -1.34 | 0.47 |
| 4 | Shortest | SC | -4.14 | -5.20 | -3.06 |
| 8 | SC | FirstAns | 8.03 | 6.99 | 9.14 |
| 8 | MaxSelfCert | MaxLogP | 2.83 | 1.64 | 4.00 |
| 8 | SelfCert-RWC | SC | 1.16 | 0.55 | 1.75 |
| 8 | SelfCert-RWC | MaxSelfCert | 2.55 | 1.70 | 3.41 |
| 8 | MaxSelfCert | SC | -1.39 | -2.49 | -0.34 |
| 8 | Shortest | SC | -7.55 | -9.12 | -6.21 |
| 32 | SC | FirstAns | 10.44 | 8.85 | 12.09 |
| 32 | MaxSelfCert | MaxLogP | 6.62 | 4.25 | 9.12 |
| 32 | SelfCert-RWC | SC | 1.56 | 0.54 | 2.65 |
| 32 | SelfCert-RWC | MaxSelfCert | 4.96 | 2.84 | 7.21 |
| 32 | MaxSelfCert | SC | -3.40 | -5.76 | -1.16 |
| 32 | Shortest | SC | -11.65 | -14.34 | -9.20 |
