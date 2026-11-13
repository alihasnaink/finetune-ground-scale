### GSM8K answer style: truncation, extraction and paired transitions vs baseline

| condition | n | flex_acc_pct | strict_acc_pct | truncated_pct | flex_extract_fail_pct | strict_extract_fail_pct | gen_tokens_mean | gen_tokens_median | gen_tokens_p95 | both_correct | gained_vs_baseline | lost_vs_baseline | both_wrong | lost_truncated | lost_extract_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1319 | 43.37 | 0.00 | 0.83 | 0.00 | 100.00 | 225.99 | 202.00 | 421.00 |  |  |  |  |  |  |
| full_ft_mid | 1319 | 57.32 | 0.00 | 1.21 | 0.00 | 100.00 | 146.62 | 126.00 | 245.10 | 443.00 | 313.00 | 129.00 | 434.00 | 3.00 | 0.00 |
| full_ft_final | 1319 | 64.44 | 0.00 | 0.61 | 0.00 | 100.00 | 138.23 | 125.00 | 226.00 | 470.00 | 380.00 | 102.00 | 367.00 | 0.00 | 0.00 |
| l2sp_final | 1319 | 64.97 | 0.00 | 0.61 | 0.00 | 100.00 | 138.96 | 126.00 | 229.10 | 474.00 | 383.00 | 98.00 | 364.00 | 0.00 | 0.00 |
| lora4_final | 1319 | 60.12 | 0.00 | 0.91 | 0.00 | 100.00 | 141.94 | 125.00 | 230.00 | 444.00 | 349.00 | 128.00 | 398.00 | 3.00 | 0.00 |
| lora16_final | 1319 | 62.24 | 0.00 | 0.68 | 0.08 | 100.00 | 139.37 | 124.00 | 235.10 | 452.00 | 369.00 | 120.00 | 378.00 | 1.00 | 1.00 |
| oplora_final | 1319 | 62.17 | 0.00 | 0.38 | 0.08 | 100.00 | 137.29 | 126.00 | 228.00 | 454.00 | 366.00 | 118.00 | 381.00 | 0.00 | 1.00 |
