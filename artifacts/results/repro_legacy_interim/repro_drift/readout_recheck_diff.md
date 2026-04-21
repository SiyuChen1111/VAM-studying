# Readout recheck drift summary

This compares the stored `readout_mode_20_29_sweep` results against the Stage-1 recheck under the same manifest settings.

## High-level read

- `soft_hazard` is the most stable mode.
- `baseline` shifts modestly but remains qualitatively the same: ceiling-heavy, positive congruency gap.
- `urgency` is the drifted mode: it stays low-ceiling, but the primary mechanism metric `model_congruency_rt_gap` flips sign from negative to positive.

## baseline

Classification: **stable-ish**

| Metric | Stored | Recheck | Delta | Note |
|---|---:|---:|---:|---|
| pred_error_rt | 2.121333 | 2.196000 | +0.074667 | mechanism-relevant |
| pred_skewness | -0.278479 | -0.338644 | -0.060165 | mechanism-relevant |
| error_minus_correct_rt | 0.535144 | 0.590350 | +0.055206 | mechanism-relevant |
| pred_median | 1.620000 | 1.660000 | +0.040000 | supporting |
| frac_at_ceiling | 0.329083 | 0.361458 | +0.032375 | mechanism-relevant |

## soft_hazard

Classification: **most stable**

| Metric | Stored | Recheck | Delta | Note |
|---|---:|---:|---:|---|
| pred_skewness | -1.322725 | -1.162417 | +0.160308 | mechanism-relevant |
| pred_mean | 0.051999 | 0.050196 | -0.001803 | supporting |
| pred_median | 0.053030 | 0.051297 | -0.001733 | supporting |
| pred_correct_rt | 0.051911 | 0.050187 | -0.001724 | mechanism-relevant |
| pred_error_rt | 0.053401 | 0.052061 | -0.001340 | mechanism-relevant |

## urgency

Classification: **drifted materially**

| Metric | Stored | Recheck | Delta | Note |
|---|---:|---:|---:|---|
| pred_skewness | -0.027132 | -0.340172 | -0.313040 | mechanism-relevant |
| model_congruency_rt_gap | -0.184356 | 0.109806 | +0.294162 | QUALITATIVE_SIGN_FLIP |
| pred_mean | 0.803539 | 1.055296 | +0.251757 | supporting |
| pred_correct_rt | 0.810319 | 1.052368 | +0.242049 | mechanism-relevant |
| error_minus_correct_rt | -0.595362 | -0.788219 | -0.192857 | mechanism-relevant |

## Mechanism-relevant blocker

- Urgency `model_congruency_rt_gap`: **-0.184356 -> 0.109806** (`+0.294162`), a qualitative sign flip.
- Under the repository rubric, this is enough to block winner promotion even though tail/support stay in a similar range.
