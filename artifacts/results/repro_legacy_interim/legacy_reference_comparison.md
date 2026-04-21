# Legacy interim reproduction comparison

Reference directory: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/organized/legacy_interim_reference`

## Spread comparison
| age_group   | condition   |   mean_state_space_spread_repro |   mean_state_space_spread_reference |    abs_diff |
|:------------|:------------|--------------------------------:|------------------------------------:|------------:|
| 20-29       | Congruent   |                        0.232316 |                            0.231621 | 0.00069575  |
| 20-29       | Incongruent |                        0.167082 |                            0.165734 | 0.00134805  |
| 80-89       | Congruent   |                        0.481527 |                            0.481482 | 4.53889e-05 |
| 80-89       | Incongruent |                        0.450396 |                            0.450729 | 0.000332862 |

## Image comparison
`figureA2_80_89_rt_distributions.png` is intentionally expanded in this bundle to include the extra 20-29 row, so legacy image-dimension matching is only enforced for A4.

| candidate_exists   | reference_exists   |   candidate_bytes |   reference_bytes |   candidate_width |   candidate_height |   reference_width |   reference_height | same_dimensions   | file                                     |
|:-------------------|:-------------------|------------------:|------------------:|------------------:|-------------------:|------------------:|-------------------:|:------------------|:-----------------------------------------|
| True               | True               |            294386 |            287795 |              3634 |               1534 |              3634 |               1534 | True              | figureA4_interim_trajectory_geometry.png |
