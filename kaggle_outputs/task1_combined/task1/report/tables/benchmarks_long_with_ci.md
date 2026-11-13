### Scores with lm-eval stderr and paired-bootstrap 95% CI of the change vs baseline

| condition | metric | score_pct | stderr_pct | delta_pp | delta_ci95_lo | delta_ci95_hi | primary |
|---|---|---|---|---|---|---|---|
| baseline | GSM8K flex EM | 43.37 | 1.37 |  |  |  | True |
| baseline | HellaSwag acc_norm | 47.32 | 0.50 |  |  |  | True |
| baseline | BoolQ acc | 64.37 | 0.84 |  |  |  | True |
| baseline | ARC-Easy acc_norm | 56.06 | 1.02 |  |  |  | True |
| baseline | GSM8K strict EM | 0.00 | 0.00 |  |  |  | False |
| baseline | HellaSwag acc | 37.56 | 0.48 |  |  |  | False |
| baseline | ARC-Easy acc | 60.90 | 1.00 |  |  |  | False |
| full_ft_mid | GSM8K flex EM | 57.32 | 1.36 | 13.95 | 10.99 | 17.13 | True |
| full_ft_mid | HellaSwag acc_norm | 45.69 | 0.50 | -1.63 | -2.22 | -1.09 | True |
| full_ft_mid | BoolQ acc | 60.80 | 0.85 | -3.58 | -5.63 | -1.59 | True |
| full_ft_mid | ARC-Easy acc_norm | 62.04 | 1.00 | 5.98 | 4.50 | 7.58 | True |
| full_ft_mid | GSM8K strict EM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| full_ft_mid | HellaSwag acc | 37.05 | 0.48 | -0.51 | -0.94 | -0.06 | False |
| full_ft_mid | ARC-Easy acc | 65.61 | 0.97 | 4.71 | 3.24 | 6.19 | False |
| full_ft_final | GSM8K flex EM | 64.44 | 1.32 | 21.08 | 18.20 | 24.18 | True |
| full_ft_final | HellaSwag acc_norm | 46.06 | 0.50 | -1.26 | -1.85 | -0.71 | True |
| full_ft_final | BoolQ acc | 61.28 | 0.85 | -3.09 | -5.05 | -1.07 | True |
| full_ft_final | ARC-Easy acc_norm | 63.47 | 0.99 | 7.41 | 5.85 | 8.96 | True |
| full_ft_final | GSM8K strict EM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| full_ft_final | HellaSwag acc | 37.37 | 0.48 | -0.19 | -0.60 | 0.23 | False |
| full_ft_final | ARC-Easy acc | 66.16 | 0.97 | 5.26 | 3.79 | 6.86 | False |
| l2sp_final | GSM8K flex EM | 64.97 | 1.31 | 21.61 | 18.42 | 24.87 | True |
| l2sp_final | HellaSwag acc_norm | 46.50 | 0.50 | -0.82 | -1.34 | -0.28 | True |
| l2sp_final | BoolQ acc | 67.06 | 0.82 | 2.69 | 0.79 | 4.40 | True |
| l2sp_final | ARC-Easy acc_norm | 63.22 | 0.99 | 7.15 | 5.72 | 8.88 | True |
| l2sp_final | GSM8K strict EM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| l2sp_final | HellaSwag acc | 37.56 | 0.48 | 0.00 | -0.38 | 0.40 | False |
| l2sp_final | ARC-Easy acc | 66.29 | 0.97 | 5.39 | 3.96 | 6.99 | False |
| lora4_final | GSM8K flex EM | 60.12 | 1.35 | 16.76 | 13.50 | 20.02 | True |
| lora4_final | HellaSwag acc_norm | 47.40 | 0.50 | 0.08 | -0.40 | 0.52 | True |
| lora4_final | BoolQ acc | 66.39 | 0.83 | 2.02 | 0.21 | 3.82 | True |
| lora4_final | ARC-Easy acc_norm | 63.22 | 0.99 | 7.15 | 5.81 | 8.59 | True |
| lora4_final | GSM8K strict EM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| lora4_final | HellaSwag acc | 37.57 | 0.48 | 0.00996 | -0.34 | 0.38 | False |
| lora4_final | ARC-Easy acc | 65.45 | 0.98 | 4.55 | 3.16 | 6.02 | False |
| lora16_final | GSM8K flex EM | 62.24 | 1.34 | 18.88 | 15.77 | 22.06 | True |
| lora16_final | HellaSwag acc_norm | 47.56 | 0.50 | 0.24 | -0.26 | 0.74 | True |
| lora16_final | BoolQ acc | 72.60 | 0.78 | 8.23 | 6.57 | 9.88 | True |
| lora16_final | ARC-Easy acc_norm | 64.06 | 0.98 | 8.00 | 6.57 | 9.51 | True |
| lora16_final | GSM8K strict EM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| lora16_final | HellaSwag acc | 37.87 | 0.48 | 0.31 | -0.06 | 0.68 | False |
| lora16_final | ARC-Easy acc | 67.05 | 0.96 | 6.14 | 4.71 | 7.66 | False |
| oplora_final | GSM8K flex EM | 62.17 | 1.34 | 18.80 | 15.54 | 22.14 | True |
| oplora_final | HellaSwag acc_norm | 47.39 | 0.50 | 0.07 | -0.40 | 0.56 | True |
| oplora_final | BoolQ acc | 72.08 | 0.78 | 7.71 | 5.99 | 9.39 | True |
| oplora_final | ARC-Easy acc_norm | 63.13 | 0.99 | 7.07 | 5.68 | 8.59 | True |
| oplora_final | GSM8K strict EM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| oplora_final | HellaSwag acc | 37.84 | 0.48 | 0.28 | -0.10 | 0.66 | False |
| oplora_final | ARC-Easy acc | 66.33 | 0.97 | 5.43 | 4.00 | 6.94 | False |
