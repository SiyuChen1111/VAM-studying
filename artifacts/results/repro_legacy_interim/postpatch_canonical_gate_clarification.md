# Post-patch canonical gate clarification

## Purpose

This note clarifies which artifact is the valid gate after the determinism and canonical-path patching work.

## Clarification

- The stored urgency row in `artifacts/results/repro_legacy_interim/readout_mode_20_29_sweep/readout_sweep_results.csv` is a **legacy pre-canonical reference only**.
- The valid post-patch control gate is the deterministic canonical recheck in `artifacts/results/repro_legacy_interim/readout_mode_20_29_sweep_recheck_postpatch/readout_sweep_results.csv`.
- Subsequent urgency search should be interpreted only relative to that patched canonical control, not relative to the old stored legacy artifact.

## Evidence

### 1. The old stored urgency artifact is not the gate anymore

`artifacts/results/repro_legacy_interim/repro_drift/reproducibility_drift_decision_memo.md` records that the earlier discrepancy was caused by nondeterministic training/evaluation drift plus non-canonical evaluation-path usage, and it explicitly says urgency optimization should wait until determinism is fixed and one canonical evaluation path is used.

The old stored urgency row in `artifacts/results/repro_legacy_interim/readout_mode_20_29_sweep/readout_sweep_results.csv` therefore remains useful as a historical reference, but not as the post-patch decision gate. Its urgency row reports:

- `model_congruency_rt_gap = -0.1843561530`
- `pred_q95 = 1.62`
- `pred_q99 = 1.77`

### 2. The patched canonical recheck is the valid control gate

The deterministic recheck in `artifacts/results/repro_legacy_interim/readout_mode_20_29_sweep_recheck_postpatch/readout_sweep_results.csv` is the canonical post-patch control sweep referenced by `artifacts/results/repro_legacy_interim/urgency_readout_decision_memo.md`.

Its urgency row reports:

- `model_congruency_rt_gap = +0.0552031994`
- `pred_q95 = 1.70`
- `pred_q99 = 1.82`
- `frac_at_ceiling = 0.0`

This is the control result that demonstrates the patched canonical path stayed in the same qualitative urgency regime after the fix: positive congruency gap, zero ceiling mass, and stable urgency-aware evaluation semantics.

### 3. Urgency search proceeded only after the canonical control stabilized

`artifacts/results/repro_legacy_interim/urgency_readout_decision_memo.md` states that determinism patching and urgency canonicalization were completed first, then the fixed-horizon control recheck stayed in the same qualitative urgency regime, and only then was the constrained urgency sweep (`artifacts/results/repro_legacy_interim/urgency_parameter_20_29_sweep_v2/urgency_sweep_results.csv`) used for ranking.

That sweep contains tied top candidates with:

- `score = 0.6033560038`
- `model_congruency_rt_gap = +0.0342101455`
- `pred_q95 = 1.62`
- `pred_q99 = 1.74`
- `frac_at_ceiling = 0.0`

So the correct interpretation is:

1. legacy stored urgency artifact = historical pre-canonical reference only;
2. post-patch deterministic canonical recheck = valid gate;
3. urgency parameter search = downstream of that gate, not a replacement for it.
