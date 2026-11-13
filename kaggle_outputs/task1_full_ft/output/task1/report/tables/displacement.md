### Full-model displacement ||θ−θ0||₂ and the L2-SP penalty Ω = ½||θ−θ0||². For runs with lambda_applied = False, λΩ uses the reference λ and is NOT part of their objective (shown only to compare penalty strength).

| run | stage | update | theta_minus_theta0_l2 | omega | lambda | lambda_applied | lambda_omega | sft_loss_last25 | lambda_omega_over_sft |
|---|---|---|---|---|---|---|---|---|---|
| full_ft | mid | 625 | 10.031 | 50.306 | 0.001 | False | 0.050306 | 0.20212 | 0.24889 |
| full_ft | final | 1250 | 10.67 | 56.928 | 0.001 | False | 0.056928 | 0.18783 | 0.30309 |
