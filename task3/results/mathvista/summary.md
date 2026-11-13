## mathvista: held-out evaluation split (800 questions), mean over 10 orders

### Accuracy (%)

| method | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|---|
| FirstAns | 53.86 | 53.86 | 53.86 | 53.86 | 53.86 | 53.86 |
| SC | 53.86 | 53.86 | 58.70 | 60.79 | 62.46 | 63.20 |
| MaxLogP | 53.86 | 54.76 | 54.27 | 52.69 | 50.78 | 48.12 |
| MaxSelfCert | 53.86 | 55.19 | 55.11 | 55.24 | 55.06 | 54.62 |
| SelfCert-RWC | 53.86 | 55.19 | 56.99 | 60.74 | 62.21 | 63.76 |
| Shortest | 53.86 | 55.35 | 54.99 | 54.50 | 54.39 | 53.69 |
| Vision-Max | 53.86 | 55.27 | 54.48 | 53.89 | 53.58 | 53.62 |
| Vision-RWC | 53.86 | 55.27 | 57.54 | 60.45 | 62.29 | 63.88 |
| VisConf-Max | 53.86 | 56.17 | 56.74 | 56.92 | 57.16 | 57.75 |
| VisConf-RWC | 53.86 | 56.17 | 58.54 | 61.50 | 63.31 | 64.50 |
| TokFusion-Max | 53.86 | 56.01 | 56.60 | 56.04 | 55.91 | 56.25 |
| TokFusion-RWC | 53.86 | 56.01 | 58.50 | 61.24 | 62.96 | 63.88 |

### Opportunity and failure decomposition (self-consistency selector)

| N | oracle | sc_acc | sampling_fail | selection_fail | distinct_groups | distinct_answered_groups | sc_cost | firstans_acc | firstans_cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 53.86 | 53.86 | 46.14 | 0.00 | 1.00 | 0.91 | 274.07 | 53.86 | 274.07 |
| 2 | 66.75 | 53.86 | 33.25 | 12.89 | 1.45 | 1.30 | 550.94 | 53.86 | 274.07 |
| 4 | 76.15 | 58.70 | 23.85 | 17.45 | 2.04 | 1.80 | 1101.01 | 53.86 | 274.07 |
| 8 | 82.90 | 60.79 | 17.10 | 22.11 | 2.78 | 2.44 | 2219.25 | 53.86 | 274.07 |
| 16 | 86.96 | 62.46 | 13.04 | 24.50 | 3.72 | 3.26 | 4444.67 | 53.86 | 274.07 |
| 32 | 89.50 | 63.20 | 10.50 | 26.30 | 4.90 | 4.33 | 8897.76 | 53.86 | 274.07 |

### N = 32: sampling vs selection failures by method (%)

| method | accuracy | oracle | sampling_fail | selection_fail | selection_gap |
|---|---|---|---|---|---|
| FirstAns | 53.86 | 89.50 | 10.50 | 35.64 | 35.64 |
| SC | 63.20 | 89.50 | 10.50 | 26.30 | 26.30 |
| MaxLogP | 48.12 | 89.50 | 10.50 | 41.38 | 41.38 |
| MaxSelfCert | 54.62 | 89.50 | 10.50 | 34.88 | 34.88 |
| SelfCert-RWC | 63.76 | 89.50 | 10.50 | 25.74 | 25.74 |
| Shortest | 53.69 | 89.50 | 10.50 | 35.81 | 35.81 |
| Vision-Max | 53.62 | 89.50 | 10.50 | 35.88 | 35.88 |
| Vision-RWC | 63.88 | 89.50 | 10.50 | 25.62 | 25.62 |
| VisConf-Max | 57.75 | 89.50 | 10.50 | 31.75 | 31.75 |
| VisConf-RWC | 64.50 | 89.50 | 10.50 | 25.00 | 25.00 |
| TokFusion-Max | 56.25 | 89.50 | 10.50 | 33.25 | 33.25 |
| TokFusion-RWC | 63.88 | 89.50 | 10.50 | 25.62 | 25.62 |

### Paired accuracy differences, held-out (pp, 95% bootstrap CI)

| N | a | b | diff_pp | ci_lo_pp | ci_hi_pp |
|---|---|---|---|---|---|
| 4 | SC | FirstAns | 4.84 | 3.94 | 5.65 |
| 4 | MaxSelfCert | MaxLogP | 0.84 | -0.15 | 1.85 |
| 4 | SelfCert-RWC | SC | -1.71 | -2.52 | -0.91 |
| 4 | SelfCert-RWC | MaxSelfCert | 1.88 | 1.19 | 2.53 |
| 4 | MaxSelfCert | SC | -3.59 | -4.79 | -2.39 |
| 4 | Shortest | SC | -3.71 | -4.91 | -2.55 |
| 4 | VisConf-RWC | SC | -0.16 | -0.94 | 0.58 |
| 4 | VisConf-RWC | VisConf-Max | 1.80 | 1.21 | 2.45 |
| 4 | VisConf-Max | MaxSelfCert | 1.62 | 0.56 | 2.83 |
| 4 | Vision-RWC | SC | -1.16 | -1.96 | -0.37 |
| 4 | VisConf-RWC | SelfCert-RWC | 1.55 | 0.77 | 2.40 |
| 4 | TokFusion-Max | VisConf-Max | -0.14 | -0.69 | 0.39 |
| 4 | TokFusion-RWC | VisConf-RWC | -0.04 | -0.44 | 0.38 |
| 8 | SC | FirstAns | 6.93 | 5.77 | 8.03 |
| 8 | MaxSelfCert | MaxLogP | 2.55 | 1.28 | 3.84 |
| 8 | SelfCert-RWC | SC | -0.05 | -0.83 | 0.81 |
| 8 | SelfCert-RWC | MaxSelfCert | 5.50 | 4.24 | 6.64 |
| 8 | MaxSelfCert | SC | -5.55 | -6.99 | -4.01 |
| 8 | Shortest | SC | -6.29 | -7.86 | -4.76 |
| 8 | VisConf-RWC | SC | 0.71 | -0.06 | 1.55 |
| 8 | VisConf-RWC | VisConf-Max | 4.58 | 3.52 | 5.72 |
| 8 | VisConf-Max | MaxSelfCert | 1.69 | 0.26 | 3.21 |
| 8 | Vision-RWC | SC | -0.34 | -1.12 | 0.57 |
| 8 | VisConf-RWC | SelfCert-RWC | 0.76 | -0.07 | 1.65 |
| 8 | TokFusion-Max | VisConf-Max | -0.89 | -1.65 | -0.10 |
| 8 | TokFusion-RWC | VisConf-RWC | -0.26 | -0.70 | 0.19 |
| 32 | SC | FirstAns | 9.34 | 7.46 | 11.08 |
| 32 | MaxSelfCert | MaxLogP | 6.50 | 3.38 | 9.75 |
| 32 | SelfCert-RWC | SC | 0.56 | -0.81 | 2.14 |
| 32 | SelfCert-RWC | MaxSelfCert | 9.14 | 6.14 | 12.01 |
| 32 | MaxSelfCert | SC | -8.57 | -11.54 | -5.52 |
| 32 | Shortest | SC | -9.51 | -12.44 | -6.56 |
| 32 | VisConf-RWC | SC | 1.30 | -0.14 | 2.80 |
| 32 | VisConf-RWC | VisConf-Max | 6.75 | 3.88 | 9.50 |
| 32 | VisConf-Max | MaxSelfCert | 3.12 | 0.25 | 6.38 |
| 32 | Vision-RWC | SC | 0.68 | -0.89 | 2.21 |
| 32 | VisConf-RWC | SelfCert-RWC | 0.74 | -0.62 | 2.24 |
| 32 | TokFusion-Max | VisConf-Max | -1.50 | -3.50 | 0.38 |
| 32 | TokFusion-RWC | VisConf-RWC | -0.62 | -1.38 | 0.00 |
