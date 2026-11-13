### Resources (training time = sum of update times; feature caching reported separately)

| item | run | time_min | peak_gpu_gb | trainable_params | note |
|---|---|---|---|---|---|
| feature caching (SigLIP, separate from training) | shared | 2.73 | 1.36 | 0 | 9255 images, 56.6 img/s, 2.59 GiB FP16 |
| image download | shared | 0.60 |  | <NA> | 9255 downloaded |
| train caption_alignment | vlm196 | 43.99 | 1.55 | 1837056 | connector 1,837,056, LoRA 0, 2250 updates |
| train visual_instruction | vlm196 | 33.35 | 1.57 | 6424576 | connector 1,837,056, LoRA 4,587,520, 1500 updates |
| train natural_vqa | vlm196 | 16.25 | 1.60 | 6424576 | connector 1,837,056, LoRA 4,587,520, 750 updates |
| final evaluation (generation + NLL, 4 conditions) | vlm196 | 1.61 | 5.06 | <NA> | gen ms/answer: grounding correct 35, grounding complementary 32, grounding mismatched 32, grounding no_image 107, vqa correct 34, vqa complementary 35, vqa mismatched 36, vqa no_image 120 |
| development evaluations | vlm196 | 1.45 |  | <NA> | 12 evaluations |
| train caption_alignment | vlm49 | 18.31 | 1.48 | 1837056 | connector 1,837,056, LoRA 0, 2250 updates |
| train visual_instruction | vlm49 | 17.50 | 1.48 | 6424576 | connector 1,837,056, LoRA 4,587,520, 1500 updates |
| train natural_vqa | vlm49 | 8.71 | 1.51 | 6424576 | connector 1,837,056, LoRA 4,587,520, 750 updates |
| final evaluation (generation + NLL, 4 conditions) | vlm49 | 0.75 | 2.29 | <NA> | gen ms/answer: grounding correct 21, grounding complementary 17, grounding mismatched 17, grounding no_image 17, vqa correct 24, vqa complementary 24, vqa mismatched 24, vqa no_image 41 |
| development evaluations | vlm49 | 0.62 |  | <NA> | 12 evaluations |
