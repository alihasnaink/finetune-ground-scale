## math: signal analysis at N = 32 (calibration pooled over all 32 rollouts)

### eval split: 800 questions, 469 with mixed correctness; mean spearman(length, correct) on mixed = -0.076

| score | top-bottom quartile, all q (pp) | top-bottom quartile, mixed q (pp) | spearman(score, correct), mixed q | spearman(score, length) |
|---|---|---|---|---|
| L (mean log-prob) | 7.828 | 13.353 | 0.134 | 0.225 |
| K (self-certainty) | 11.984 | 20.442 | 0.198 | -0.161 |
| neg_len (shortness (-length)) | 4.562 | 7.783 | 0.076 | -1.000 |

### all split: 1000 questions, 584 with mixed correctness; mean spearman(length, correct) on mixed = -0.095

| score | top-bottom quartile, all q (pp) | top-bottom quartile, mixed q (pp) | spearman(score, correct), mixed q | spearman(score, length) |
|---|---|---|---|---|
| L (mean log-prob) | 6.800 | 11.644 | 0.121 | 0.218 |
| K (self-certainty) | 11.562 | 19.799 | 0.194 | -0.157 |
| neg_len (shortness (-length)) | 5.812 | 9.953 | 0.095 | -1.000 |
