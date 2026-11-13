### Training resources (train time excludes checkpointing/eval)

| run | trainable_params | trainable_pct | train_time_h | seq_tokens_per_s | peak_mem_alloc_gb | peak_mem_reserved_gb | recovery_ckpt_gib | inference_export_gib | skipped_updates | gpu | lr | lambda | overhead_ref_storage | overhead_ref_bytes | overhead_ref_setup_s | overhead_h2d_bytes_setup | overhead_h2d_bytes_per_update | overhead_penalty_time_s | overhead_penalty_ms_per_update | overhead_penalty_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l2sp | 596049920 | 100.000 | 2.428 | 1133.690 | 12.318 | 13.191 | 6.662 | 1.121 | 0 | Tesla T4 | 0.000 | 0.001 | gpu_bf16 | 1192099840 | 2.075 | 1192099840 | 0 | 290.961 | 232.769 | time to compute Omega and add lambda*(theta-theta0) to the gradients (synchronised); included in train_time_s |
