# Urgency rerun stability

Three repeated reruns of the exact urgency setting used in the fixed readout sweep:
`additive_urgency, start=0.8, slope=0.25, floor=0.0, time_steps_factor=2.0, scale=0.1, epochs=5, train_subset=12000, test_subset=24000, lambda_pileup=1.0`.

- Classification: **mildly unstable**
- `model_congruency_rt_gap` range: **0.093610 to 0.177701** (spread `0.084091`)
- `score` range: **0.392876 to 0.416293** (spread `0.023416`)
- `pred_skewness` range: **-0.268204 to 0.245710** (spread `0.513914`)

## Table

| Rerun | Score | Congruency gap | Pred skewness | Quantile score | Coverage score | Error-correct RT |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.416293 | 0.152964 | 0.245710 | 0.020811 | 0.052376 | -0.631605 |
| 2 | 0.400721 | 0.177701 | 0.207981 | 0.010047 | 0.052913 | -0.452584 |
| 3 | 0.392876 | 0.093610 | -0.268204 | 0.010047 | 0.053450 | -0.817097 |

## Interpretation

- All three reruns stayed in the same broad urgency regime: zero ceiling mass, positive congruency gap, negative error-correct RT.
- The branch is therefore not flipping sign within these three reruns, but it still disagrees qualitatively with the stored urgency artifact, which had a negative congruency gap.
- That means the instability is real at the branch/artifact level even if this local 3-run sample does not itself show repeated sign flips.
