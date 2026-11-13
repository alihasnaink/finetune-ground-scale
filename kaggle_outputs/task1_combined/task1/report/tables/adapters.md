### Adapter runs: size, OPLoRA basis/export overhead, merged (FP16 export) vs. unmerged logits on 4 training sequences (256 tokens) (adapter effect = unmerged vs. starting model, for scale), and the fraction of ||ΔW||² in W0's top-k output/input singular subspaces over all adapted projections (OPLoRA constraint check)

| run | r | alpha | protect_rank | adapter_params | adapter_mib_fp32 | basis_mib | basis_preprocess_s | export_merge_s | merge_mean_abs_dlogit | merge_max_abs_dlogit | merge_top1_agree | adapter_effect_mean_abs_dlogit | rho_out_mean_all | rho_out_max_all | rho_in_mean_all | rho_in_max_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora4 | 4 | 8 |  | 2523136 | 9.625 |  |  | 3.634 | 0.00686 | 0.2266 | 0.9987 | 1.349 | 0.09219 | 0.1791 | 0.1158 | 0.2683 |
| lora16 | 16 | 32 |  | 10092544 | 38.5 |  |  | 3.499 | 0.0068 | 0.1875 | 0.9987 | 1.345 | 0.09314 | 0.1668 | 0.1087 | 0.1647 |
| oplora | 16 | 32 | 128 | 10092544 | 38.5 | 308 | 31.51 | 3.036 | 0.00677 | 0.1211 | 1 | 1.343 | 1.7e-09 | 1.59e-08 | 2.01e-08 | 3.49e-08 |
