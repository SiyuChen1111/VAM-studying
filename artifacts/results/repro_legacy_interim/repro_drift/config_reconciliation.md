# Config reconciliation: stored vs recheck urgency run

## Verdict

- No effective mismatch was found in the declared sweep configuration.
- The stored and recheck urgency runs are manifest-identical on the exposed readout/training fields.
- The mismatch is in the produced artifacts and metrics, not in the recorded nominal config.

## Declared config comparison

| Field | Stored | Recheck | Match |
|---|---:|---:|---|
| urgency_type | additive_urgency | additive_urgency | yes |
| urgency_start | 0.8 | 0.8 | yes |
| urgency_slope | 0.25 | 0.25 | yes |
| urgency_floor | 0.0 | 0.0 | yes |
| choice_temperature | 0.1 | 0.1 | yes |
| train_subset_n | 12000 | 12000 | yes |
| test_subset_n | 24000 | 24000 | yes |
| time_steps_factor | 2.0 | 2.0 | yes |
| scale | 0.1 | 0.1 | yes |
| epochs | 5 | 5 | yes |
| lambda_pileup | 1.0 | 1.0 | yes |

## Output metric drift within urgency

| Metric | Stored | Recheck | Delta |
|---|---:|---:|---:|
| score | 0.475122 | 0.395766 | -0.079356 |
| pred_mean | 0.803539 | 1.055296 | +0.251757 |
| pred_median | 0.890000 | 1.050000 | +0.160000 |
| pred_q95 | 1.620000 | 1.700000 | +0.080000 |
| pred_q99 | 1.770000 | 1.820000 | +0.050000 |
| frac_at_ceiling | 0.000000 | 0.000000 | +0.000000 |
| pred_skewness | -0.027132 | -0.340172 | -0.313040 |
| quantile_score | 0.031575 | 0.017223 | -0.014352 |
| coverage_score | 0.052107 | 0.053450 | +0.001343 |
| model_congruency_rt_gap | -0.184356 | 0.109806 | +0.294162 |
| pred_error_rt | 0.214957 | 0.264149 | +0.049192 |
| pred_correct_rt | 0.810319 | 1.052368 | +0.242049 |
| error_minus_correct_rt | -0.595362 | -0.788219 | -0.192857 |
| learned_threshold | 0.506820 | 0.503011 | -0.003810 |
| learned_noise_ampa | 0.020156 | 0.024213 | +0.004057 |
| learned_J_ext | 0.015488 | 0.018980 | +0.003492 |
| learned_I_0 | 0.325405 | 0.329090 | +0.003685 |

## Prediction artifact comparison

- Stored predictions SHA256: `c62ca9980cec34f20d85891361d3267e8470f8ac38de085e59744926d85b591c`
- Recheck predictions SHA256: `06fd3f6fac9d7fda14ec4fc994c9b3cc431ed44668e5dab67bfe1a1e82faad4c`
- Same binary artifact: **no**

| Array key | Stored shape | Recheck shape | Stored mean | Recheck mean |
|---|---|---|---:|---:|
| pred_rt | [24000] | [24000] | 0.803539 | 1.055296 |
| pred_choice | [24000] | [24000] | 1.506042 | 1.501875 |
| choice_logits | [24000, 4] | [24000, 4] | -3.605147 | -2.762804 |
| decision_times_class | [24000, 4] | [24000, 4] | 2.177464 | 2.022248 |
| decision_indices | [24000, 4] | [24000, 4] | 217.790792 | 202.325125 |
| decision_times | [24000, 4] | [24000, 4] | 2.177907 | 2.023251 |
| winner_idx | [24000] | [24000] | 1.479250 | 1.471375 |
| dv_t | [24000, 222] | [24000, 222] | 0.093223 | 0.271092 |
| urgency_gain | [222] | [222] | 0.112736 | 0.112736 |
| baseline_index | [24000] | [24000] | 208.163167 | 146.300500 |
| urgency_index | [24000] | [24000] | 80.353875 | 105.529583 |
| baseline_threshold | [24000] | [24000] | 0.153200 | 0.371307 |
| traj | [24000, 222, 4] | [24000, 222, 4] | 0.095431 | 0.135482 |
| threshold | [] | [] | 0.506820 | 0.503011 |

## Interpretation

- The nominal urgency setting is truly the same: `additive_urgency`, start `0.8`, slope `0.25`, floor `0.0`, `choice_temperature=0.1`, `scale=0.1`, `time_steps_factor=2.0`, `epochs=5`, `lambda_pileup=1.0`.
- The prediction artifacts differ as binary files and as array summaries, so the drift is downstream of identical recorded settings.
- The most important observed consequence is the qualitative flip in `model_congruency_rt_gap` from negative to positive.