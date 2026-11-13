### Final test by question type

| run | suite | qtype | condition | n_pairs | accuracy_pct | accuracy_ci95 | pair_accuracy_pct | lenient_accuracy_pct | ref_nll |
|---|---|---|---|---|---|---|---|---|---|
| vlm196 | grounding | counting | correct | 32 | 50.00 | 50.0 [37.5, 62.5] | 28.12 | 50.00 | 1.71 |
| vlm196 | grounding | presence | correct | 32 | 89.06 | 89.1 [81.2, 95.3] | 78.12 | 89.06 | 0.25 |
| vlm196 | grounding | counting | complementary | 32 | 12.50 | 12.5 [4.7, 21.9] | 3.12 | 12.50 | 3.76 |
| vlm196 | grounding | presence | complementary | 32 | 10.94 | 10.9 [4.7, 18.8] | 0.00 | 10.94 | 4.25 |
| vlm196 | grounding | counting | mismatched | 32 | 20.31 | 20.3 [10.9, 29.7] | 3.12 | 20.31 | 2.93 |
| vlm196 | grounding | presence | mismatched | 32 | 54.69 | 54.7 [48.4, 60.9] | 12.50 | 54.69 | 1.88 |
| vlm196 | grounding | counting | no_image | 32 | 0.00 | 0.0 [0.0, 0.0] | 0.00 | 0.00 | 5.60 |
| vlm196 | grounding | presence | no_image | 32 | 0.00 | 0.0 [0.0, 0.0] | 0.00 | 48.44 | 7.68 |
| vlm196 | vqa | number | correct | 24 | 48.33 | 48.3 [36.7, 59.8] | 12.50 | 48.33 | 1.75 |
| vlm196 | vqa | other | correct | 36 | 51.53 | 51.5 [38.7, 63.6] | 27.78 | 51.53 | 1.86 |
| vlm196 | vqa | yes/no | correct | 36 | 67.78 | 67.8 [58.1, 76.9] | 33.33 | 67.78 | 0.65 |
| vlm196 | vqa | number | complementary | 24 | 15.83 | 15.8 [6.9, 25.6] | 0.00 | 15.83 | 4.46 |
| vlm196 | vqa | other | complementary | 36 | 10.00 | 10.0 [3.6, 17.4] | 0.00 | 10.00 | 4.78 |
| vlm196 | vqa | yes/no | complementary | 36 | 47.22 | 47.2 [36.7, 57.6] | 11.11 | 47.22 | 1.57 |
| vlm196 | vqa | number | mismatched | 24 | 19.38 | 19.4 [9.0, 31.0] | 4.17 | 19.38 | 3.07 |
| vlm196 | vqa | other | mismatched | 36 | 19.44 | 19.4 [9.7, 30.6] | 8.33 | 19.44 | 3.70 |
| vlm196 | vqa | yes/no | mismatched | 36 | 59.03 | 59.0 [53.8, 65.0] | 8.33 | 59.03 | 1.23 |
| vlm196 | vqa | number | no_image | 24 | 0.00 | 0.0 [0.0, 0.0] | 0.00 | 0.00 | 5.77 |
| vlm196 | vqa | other | no_image | 36 | 0.00 | 0.0 [0.0, 0.0] | 0.00 | 0.00 | 12.92 |
| vlm196 | vqa | yes/no | no_image | 36 | 0.00 | 0.0 [0.0, 0.0] | 0.00 | 50.00 | 8.36 |
| vlm49 | grounding | counting | correct | 32 | 54.69 | 54.7 [42.2, 67.2] | 31.25 | 54.69 | 1.63 |
| vlm49 | grounding | presence | correct | 32 | 90.62 | 90.6 [82.8, 96.9] | 81.25 | 90.62 | 0.19 |
| vlm49 | grounding | counting | complementary | 32 | 14.06 | 14.1 [6.2, 23.4] | 3.12 | 14.06 | 3.36 |
| vlm49 | grounding | presence | complementary | 32 | 9.38 | 9.4 [3.1, 17.2] | 0.00 | 9.38 | 3.43 |
| vlm49 | grounding | counting | mismatched | 32 | 17.19 | 17.2 [9.4, 25.0] | 0.00 | 17.19 | 2.61 |
| vlm49 | grounding | presence | mismatched | 32 | 48.44 | 48.4 [42.2, 53.1] | 3.12 | 48.44 | 1.62 |
| vlm49 | grounding | counting | no_image | 32 | 17.19 | 17.2 [9.4, 26.6] | 0.00 | 17.19 | 1.80 |
| vlm49 | grounding | presence | no_image | 32 | 50.00 | 50.0 [50.0, 50.0] | 0.00 | 50.00 | 0.85 |
| vlm49 | vqa | number | correct | 24 | 41.46 | 41.5 [28.3, 54.6] | 12.50 | 41.46 | 2.05 |
| vlm49 | vqa | other | correct | 36 | 42.36 | 42.4 [29.7, 55.1] | 22.22 | 42.36 | 2.25 |
| vlm49 | vqa | yes/no | correct | 36 | 70.14 | 70.1 [61.5, 78.8] | 36.11 | 70.14 | 0.68 |
| vlm49 | vqa | number | complementary | 24 | 20.62 | 20.6 [10.2, 32.5] | 4.17 | 20.62 | 3.48 |
| vlm49 | vqa | other | complementary | 36 | 8.19 | 8.2 [2.8, 14.7] | 0.00 | 8.19 | 4.59 |
| vlm49 | vqa | yes/no | complementary | 36 | 44.03 | 44.0 [34.0, 54.2] | 8.33 | 44.03 | 1.24 |
| vlm49 | vqa | number | mismatched | 24 | 17.92 | 17.9 [7.5, 29.6] | 4.17 | 17.92 | 2.71 |
| vlm49 | vqa | other | mismatched | 36 | 14.31 | 14.3 [6.9, 21.7] | 0.00 | 14.31 | 3.82 |
| vlm49 | vqa | yes/no | mismatched | 36 | 57.64 | 57.6 [53.5, 62.4] | 2.78 | 57.64 | 1.21 |
| vlm49 | vqa | number | no_image | 24 | 27.50 | 27.5 [17.3, 37.7] | 0.00 | 27.50 | 2.15 |
| vlm49 | vqa | other | no_image | 36 | 9.17 | 9.2 [3.2, 15.7] | 0.00 | 9.17 | 3.85 |
| vlm49 | vqa | yes/no | no_image | 36 | 56.25 | 56.2 [52.9, 60.0] | 0.00 | 56.25 | 1.08 |
