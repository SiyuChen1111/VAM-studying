# Hybrid legacy parameter comparison notes

This comparison uses the user-requested hybrid legacy setup.

- `20-29` replay = 20-29 logits + parsed young scale (0.1) + reused 80-89 legacy parameter set.
- `80-89` replay = archived legacy target-supervision config/params.
- Consequence: noise-related parameters are expected to be the same across the two rows because the hybrid replay reuses the 80-89 parameter tensor block.
- The main internal difference should therefore appear in `scale`, not in `ww.noise_ampa`.

| parameter   | parameter_key   |   hybrid_20_29 |   legacy_80_89 |   abs_diff | note                         |
|:------------|:----------------|---------------:|---------------:|-----------:|:-----------------------------|
| Scale       | scale           |      0.1       |      0.2       |        0.1 | different_under_hybrid_setup |
| Noise AMPA  | ww.noise_ampa   |      0.0228012 |      0.0228012 |        0   | shared_under_hybrid_setup    |
| Threshold   | ww.threshold    |      0.49637   |      0.49637   |        0   | shared_under_hybrid_setup    |
| J_ext       | ww.J_ext        |      0.0216963 |      0.0216963 |        0   | shared_under_hybrid_setup    |
| I_0         | ww.I_0          |      0.331516  |      0.331516  |        0   | shared_under_hybrid_setup    |
| Tau AMPA    | ww.tau_ampa     |      2         |      2         |        0   | shared_under_hybrid_setup    |