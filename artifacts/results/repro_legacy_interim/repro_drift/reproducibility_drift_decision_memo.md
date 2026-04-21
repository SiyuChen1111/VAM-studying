# Reproducibility drift decision memo

## Bottom line

The urgency instability is **real at the artifact level** and the branch is **not currently trustworthy enough for promotion**.

The root cause is best classified as **nondeterministic training/evaluation drift**, with a second contributing issue from **non-canonical evaluation paths**. It is **not** explained by a manifest/config mismatch, and there is no evidence here for a subset mismatch.

## 1. What exactly drifted?

The stored and recheck urgency runs use the same recorded setting:
- `additive_urgency`
- `urgency_start = 0.8`
- `urgency_slope = 0.25`
- `urgency_floor = 0.0`
- `choice_temperature = 0.1`
- `time_steps_factor = 2.0`
- `scale = 0.1`
- `epochs = 5`
- `lambda_pileup = 1.0`
- `train_subset_n = 12000`
- `test_subset_n = 24000`

But the produced urgency metrics differ materially:
- `model_congruency_rt_gap`: **-0.1844 -> +0.1098** (qualitative sign flip)
- `pred_mean`: `0.8035 -> 1.0553`
- `pred_q95`: `1.62 -> 1.70`
- `pred_q99`: `1.77 -> 1.82`
- `pred_skewness`: `-0.0271 -> -0.3402`
- `quantile_score`: `0.0316 -> 0.0172`
- `error_minus_correct_rt`: `-0.5954 -> -0.7882`

Under the mechanism-oriented rubric, the sign flip in `model_congruency_rt_gap` is enough to block winner promotion.

## 2. What is causing the drift?

### Ruled out
- **Config mismatch:** ruled out by direct field-by-field reconciliation.
- **Subset mismatch:** not supported by the available evidence; subset selection is seeded in the sweep scripts.

### Primary cause
- **Nondeterministic training/evaluation:** confirmed.
  - `train_age_groups_efficient.py` uses `DataLoader(..., shuffle=True)` without a fixed generator.
  - `vgg_wongwang_lim.py` samples fresh `torch.randn(...)` noise inside WW dynamics.
  - `compute_stage2_outputs(...)` performs two separate stochastic WW simulations per evaluation.
  - no deterministic backend policy is visible in the inspected path.

### Secondary cause
- **Evaluation-path inconsistency / non-canonical reporting:** confirmed.
  - Stored “best” metrics are training-eval metrics from the training path.
  - Later predictions are recomputed on test logits.
  - Some downstream post-analysis paths reconstruct RT using `decision_times_class.min(dim=1)` and do not preserve urgency-readout semantics.

### Checkpoint sensitivity
- **Unstable checkpoint selection:** plausible secondary contributor.
  - The ranking key uses floating metrics and a hard congruency-gap gate.
  - Small stochastic movement can change which checkpoint is selected.

## 3. Is the urgency branch currently trustworthy enough for promotion?

No.

Three fresh reruns of the exact urgency setting were internally consistent in staying **positive** on `model_congruency_rt_gap` (`+0.1530`, `+0.1777`, `+0.0936`), but that makes the stored negative artifact look **non-reproducible**, not trustworthy. Right now the branch does not provide stable enough evidence for mechanism-level promotion.

## 4. What is the smallest fix needed before optimization can resume?

The smallest credible fix is:

1. **Freeze and fix determinism first.**
   - Set explicit torch seeds before training and before every evaluation/inference pass.
   - Give the training DataLoader a fixed generator.
   - Add a deterministic backend policy where supported.

2. **Use one canonical evaluation path.**
   - Report urgency metrics only through the urgency-aware prediction path (`infer_predictions_from_params(...)` / urgency readout config), not through alternate RT reconstruction.

3. **Save the exact predictions used for model selection.**
   - Remove ambiguity between stored “best” metrics and later recomputed artifacts.

4. **Then rerun exactly one urgency setting as a reproducibility check.**
   - Only resume urgency optimization if the canonical deterministic rerun remains stable.

## Final recommendation

**Freeze and fix determinism first.**

Do not resume urgency search yet.
