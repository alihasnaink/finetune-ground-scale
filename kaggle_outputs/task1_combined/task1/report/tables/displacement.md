### Full-model displacement ||θ−θ0||₂ and the L2-SP penalty Ω = ½||θ−θ0||². For runs with lambda_applied = False, λΩ uses the reference λ and is NOT part of their objective (shown only to compare penalty strength).

| run | stage | update | theta_minus_theta0_l2 | omega | lambda | lambda_applied | lambda_omega | sft_loss_last25 | lambda_omega_over_sft |
|---|---|---|---|---|---|---|---|---|---|
| full_ft | mid | 625 | 10.031 | 50.306 | 0.001 | False | 0.050306 | 0.20212 | 0.24889 |
| full_ft | final | 1250 | 10.67 | 56.928 | 0.001 | False | 0.056928 | 0.18783 | 0.30309 |
| l2sp | final | 1250 | 6.2757 | 19.692 | 0.001 | True | 0.019692 | 0.18764 | 0.10495 |
| lora4 | final | 1250 | 9.1235 | 41.619 | 0.001 | False | 0.041619 | 0.21551 | 0.19312 |
| lora16 | final | 1250 | 13.317 | 88.672 | 0.001 | False | 0.088672 | 0.20244 | 0.43801 |
| oplora | final | 1250 | 12.432 | 77.273 | 0.001 | False | 0.077273 | 0.20326 | 0.38017 |
