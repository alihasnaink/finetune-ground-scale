## mathvista: signal analysis at N = 32 (calibration pooled over all 32 rollouts)

### eval split: 800 questions, 634 with mixed correctness; mean spearman(length, correct) on mixed = -0.050

| score | top-bottom quartile, all q (pp) | top-bottom quartile, mixed q (pp) | spearman(score, correct), mixed q | spearman(score, length) |
|---|---|---|---|---|
| L (mean log-prob) | 1.594 | 2.011 | 0.029 | 0.329 |
| K (self-certainty) | 3.484 | 4.397 | 0.048 | 0.349 |
| neg_len (shortness (-length)) | 4.062 | 5.126 | 0.050 | -1.000 |
| V (visual engagement) | 3.781 | 4.771 | 0.041 | -0.686 |
| C (calibrated confidence) | 3.828 | 4.830 | 0.048 | 0.347 |
| Z (VisConf) | 7.391 | 9.326 | 0.089 | -0.275 |
| Ztok (token fusion) | 7.234 | 9.129 | 0.080 | -0.322 |

### all split: 1000 questions, 791 with mixed correctness; mean spearman(length, correct) on mixed = -0.047

| score | top-bottom quartile, all q (pp) | top-bottom quartile, mixed q (pp) | spearman(score, correct), mixed q | spearman(score, length) |
|---|---|---|---|---|
| L (mean log-prob) | 2.225 | 2.813 | 0.038 | 0.325 |
| K (self-certainty) | 3.862 | 4.883 | 0.053 | 0.341 |
| neg_len (shortness (-length)) | 3.850 | 4.867 | 0.047 | -1.000 |
| V (visual engagement) | 2.900 | 3.666 | 0.033 | -0.685 |
| C (calibrated confidence) | 4.112 | 5.199 | 0.053 | 0.338 |
| Z (VisConf) | 7.250 | 9.166 | 0.087 | -0.279 |
| Ztok (token fusion) | 6.925 | 8.755 | 0.078 | -0.332 |

### Confidence-vision relationship (eval)

- within-rollout Pearson(c_t, v_t), per-question mean: mean -0.197, median -0.209, IQR [-0.300, -0.090], 10.3% of questions positive (789 questions, 24296 rollouts)
- by rollout correctness: correct -0.168, incorrect -0.218
- Cov_t(c, v) = Z^token - Z: mean -0.00224; correct rollouts -0.00198, incorrect -0.00260
- within-question Spearman(Z, Z^token): mean 0.917, median 0.933

### Confidence-vision relationship (all)

- within-rollout Pearson(c_t, v_t), per-question mean: mean -0.191, median -0.206, IQR [-0.298, -0.085], 11.8% of questions positive (985 questions, 30380 rollouts)
- by rollout correctness: correct -0.159, incorrect -0.214
- Cov_t(c, v) = Z^token - Z: mean -0.00217; correct rollouts -0.00187, incorrect -0.00253
- within-question Spearman(Z, Z^token): mean 0.917, median 0.936

### Illustrative trajectories (fixed rule; held-out split order, order 0, N = 32)

- A: VisConf-RWC correct, SC wrong: question 643 (gold 'A'; 10/32 correct); top-Z correct rollout 16: answer 'A', T=328, Z=0.0064, Ztok=0.0055, C=0.513, V=0.0124; top-Z incorrect rollout 10: answer 'D', T=308, Z=0.0066, Ztok=0.0061, C=0.497, V=0.0133
- B: SC correct, VisConf-RWC wrong: question 552 (gold '4'; 14/32 correct); top-Z correct rollout 3: answer '4', T=242, Z=0.0635, Ztok=0.0639, C=0.552, V=0.1150; top-Z incorrect rollout 24: answer '2', T=210, Z=0.0752, Ztok=0.0710, C=0.523, V=0.1438
- C: VisConf-Max vs TokFusion-Max disagree: question 220 (gold 'A'; 24/32 correct); top-Z correct rollout 15: answer 'A', T=353, Z=0.0081, Ztok=0.0059, C=0.564, V=0.0143; top-Z incorrect rollout 18: answer 'NO_ANSWER', T=276, Z=0.0073, Ztok=0.0063, C=0.474, V=0.0153

Figure: figures/trajectories.png; token values in trajectories.csv.
