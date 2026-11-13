# Task 1 evidence bundle

Runs: full_ft, lora4, lora16, oplora
Evaluation conditions: baseline, full_ft_mid, full_ft_final, lora4_final, lora16_final, oplora_final
Training data sha256: d46b7f0b10222bc997d14431a4d1a2605682d59117eeb54dcb4734076a8b9265

## Where each required piece of evidence lives
| Manual requirement | Files |
|---|---|
| Benchmark results (all conditions x 4 benchmarks, Δ in pp) | report/tables/benchmarks_primary.*, benchmarks_all.*, benchmarks_long_with_ci.*, figures/benchmark_delta.png |
| GSM8K extraction / truncation vs. correctness | report/tables/gsm8k_behavior.*, figures/gsm8k_length_hist.png, eval/*/samples_gsm8k_cot.jsonl |
| Learning curves (SFT loss, lr) | report/figures/training_loss.png, training_lr_gradnorm_scale.png, runs/*/train_log.csv |
| Regularisation / displacement | report/tables/displacement.* (incl. Ω, λΩ), l2sp_objective.*, displacement_by_layer.csv, displacement_by_module.csv, figures/l2sp_penalty.png, runs/*/displacement_*.json |
| L2-SP logs (Ω, λΩ, total loss) and reference overhead | runs/l2sp/train_log.csv (omega, lambda_omega, total_loss, sft_grad_norm, reg_grad_norm), runs/l2sp/train_summary.json (overhead_*) |
| LoRA / OPLoRA: adapter params, basis storage + preprocessing, export overhead, merged vs. unmerged logits, OPLoRA constraint verification | report/tables/adapters.*, adapter_subspace_by_module.*, adapter_subspace_per_projection.csv, runs/<adapter run>/merge_check_final.json, adapter_subspace_final.json + .csv, train_summary.json (overhead_*) |
| Imported results of earlier notebooks | data/imported_from.json |
| Resources (params, time, memory, checkpoint size) | report/tables/resources.*, runs/*/train_summary.json, eval/*/metrics.json (timings) |
| Weight-space + activation diagnostics | report/tables/diagnostics_*.*, figures/diagnostics.png, diagnostics/*.csv |
| Qualitative (168 responses, 42-response annotation subset) | report/qualitative/responses_by_question.md, all_responses.csv, annotation_sheet.csv |
| Data selection / probe set / prompts | data/data_stats.json, selected_examples.csv, excluded_examples.csv, probe_set.json, qualitative_prompts.json |
| Config + environment | runs/*/config.yaml, env/*.json |

FP32 down_proj deltas used by the diagnostics (task1_artifacts/deltas/<run>/<stage>.pt) and the
FP16 inference exports (task1_artifacts/exports/<run>/<stage>/, for adapter runs including the
exported FP32 factors adapter_factors.pt, projected for OPLoRA) are kept in the Kaggle output and
are deliberately not part of this zip.

The annotation columns (ann_*) in annotation_sheet.csv are intentionally blank: they are to be
filled by hand. Heuristic columns (json_valid, distinct_4gram_ratio, ...) are automatic aids.

## All files
- data/data_stats.json
- data/excluded_examples.csv
- data/imported_from.json
- data/probe_set.json
- data/qualitative_prompts.json
- data/selected_examples.csv
- diagnostics/activation_full_ft.csv
- diagnostics/activation_lora16.csv
- diagnostics/activation_lora4.csv
- diagnostics/activation_oplora.csv
- diagnostics/activation_summary_full_ft.csv
- diagnostics/activation_summary_lora16.csv
- diagnostics/activation_summary_lora4.csv
- diagnostics/activation_summary_oplora.csv
- diagnostics/probe_check.json
- diagnostics/weight_space_full_ft.csv
- diagnostics/weight_space_lora16.csv
- diagnostics/weight_space_lora4.csv
- diagnostics/weight_space_oplora.csv
- env/env_full_ft.json
- env/env_lora16.json
- eval/baseline/lm_eval_results.json
- eval/baseline/metrics.json
- eval/baseline/qualitative.jsonl
- eval/baseline/samples_arc_easy.jsonl
- eval/baseline/samples_boolq.jsonl
- eval/baseline/samples_gsm8k_cot.jsonl
- eval/baseline/samples_hellaswag.jsonl
- eval/full_ft_final/lm_eval_results.json
- eval/full_ft_final/metrics.json
- eval/full_ft_final/qualitative.jsonl
- eval/full_ft_final/samples_arc_easy.jsonl
- eval/full_ft_final/samples_boolq.jsonl
- eval/full_ft_final/samples_gsm8k_cot.jsonl
- eval/full_ft_final/samples_hellaswag.jsonl
- eval/full_ft_mid/lm_eval_results.json
- eval/full_ft_mid/metrics.json
- eval/full_ft_mid/qualitative.jsonl
- eval/full_ft_mid/samples_arc_easy.jsonl
- eval/full_ft_mid/samples_boolq.jsonl
- eval/full_ft_mid/samples_gsm8k_cot.jsonl
- eval/full_ft_mid/samples_hellaswag.jsonl
- eval/lora16_final/lm_eval_results.json
- eval/lora16_final/metrics.json
- eval/lora16_final/qualitative.jsonl
- eval/lora16_final/samples_arc_easy.jsonl
- eval/lora16_final/samples_boolq.jsonl
- eval/lora16_final/samples_gsm8k_cot.jsonl
- eval/lora16_final/samples_hellaswag.jsonl
- eval/lora4_final/lm_eval_results.json
- eval/lora4_final/metrics.json
- eval/lora4_final/qualitative.jsonl
- eval/lora4_final/samples_arc_easy.jsonl
- eval/lora4_final/samples_boolq.jsonl
- eval/lora4_final/samples_gsm8k_cot.jsonl
- eval/lora4_final/samples_hellaswag.jsonl
- eval/oplora_final/lm_eval_results.json
- eval/oplora_final/metrics.json
- eval/oplora_final/qualitative.jsonl
- eval/oplora_final/samples_arc_easy.jsonl
- eval/oplora_final/samples_boolq.jsonl
- eval/oplora_final/samples_gsm8k_cot.jsonl
- eval/oplora_final/samples_hellaswag.jsonl
- lane_gpu1.log
- report/figures/benchmark_delta.png
- report/figures/diagnostics.png
- report/figures/displacement_by_layer_module.png
- report/figures/gsm8k_length_hist.png
- report/figures/training_loss.png
- report/figures/training_lr_gradnorm_scale.png
- report/qualitative/all_responses.csv
- report/qualitative/annotation_sheet.csv
- report/qualitative/responses_by_question.md
- report/tables/adapter_subspace_by_module.csv
- report/tables/adapter_subspace_by_module.md
- report/tables/adapter_subspace_per_projection.csv
- report/tables/adapters.csv
- report/tables/adapters.md
- report/tables/benchmarks_all.csv
- report/tables/benchmarks_all.md
- report/tables/benchmarks_long_with_ci.csv
- report/tables/benchmarks_long_with_ci.md
- report/tables/benchmarks_primary.csv
- report/tables/benchmarks_primary.md
- report/tables/diagnostics_R_math_vs_nonmath.csv
- report/tables/diagnostics_R_math_vs_nonmath.md
- report/tables/diagnostics_activation_summary.csv
- report/tables/diagnostics_activation_summary.md
- report/tables/diagnostics_weight_space.csv
- report/tables/diagnostics_weight_space.md
- report/tables/displacement.csv
- report/tables/displacement.md
- report/tables/displacement_by_layer.csv
- report/tables/displacement_by_module.csv
- report/tables/gsm8k_behavior.csv
- report/tables/gsm8k_behavior.md
- report/tables/resources.csv
- report/tables/resources.md
- runs/full_ft/config.yaml
- runs/full_ft/displacement_final.json
- runs/full_ft/displacement_mid.json
- runs/full_ft/export_final.json
- runs/full_ft/export_mid.json
- runs/full_ft/train_log.csv
- runs/full_ft/train_summary.json
- runs/lora16/adapter_subspace_final.csv
- runs/lora16/adapter_subspace_final.json
- runs/lora16/config.yaml
- runs/lora16/displacement_final.json
- runs/lora16/export_final.json
- runs/lora16/merge_check_final.json
- runs/lora16/train_log.csv
- runs/lora16/train_summary.json
- runs/lora4/adapter_subspace_final.csv
- runs/lora4/adapter_subspace_final.json
- runs/lora4/config.yaml
- runs/lora4/displacement_final.json
- runs/lora4/export_final.json
- runs/lora4/merge_check_final.json
- runs/lora4/train_log.csv
- runs/lora4/train_summary.json
- runs/oplora/adapter_subspace_final.csv
- runs/oplora/adapter_subspace_final.json
- runs/oplora/config.yaml
- runs/oplora/displacement_final.json
- runs/oplora/export_final.json
- runs/oplora/merge_check_final.json
- runs/oplora/train_log.csv
- runs/oplora/train_summary.json
