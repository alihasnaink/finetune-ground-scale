### GSM8K answer style: truncation, extraction and paired transitions vs baseline

| condition | n | flex_acc_pct | strict_acc_pct | truncated_pct | flex_extract_fail_pct | strict_extract_fail_pct | gen_tokens_mean | gen_tokens_median | gen_tokens_p95 | both_correct | gained_vs_baseline | lost_vs_baseline | both_wrong | lost_truncated | lost_extract_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1319 | 43.37 | 0.00 | 0.83 | 0.00 | 100.00 | 225.99 | 202.00 | 421.00 |  |  |  |  |  |  |
| full_ft_mid | 1319 | 57.32 | 0.00 | 1.21 | 0.00 | 100.00 | 146.62 | 126.00 | 245.10 | 443.00 | 313.00 | 129.00 | 434.00 | 3.00 | 0.00 |
| full_ft_final | 1319 | 64.44 | 0.00 | 0.61 | 0.00 | 100.00 | 138.23 | 125.00 | 226.00 | 470.00 | 380.00 | 102.00 | 367.00 | 0.00 | 0.00 |
