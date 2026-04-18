# Minimal conflict-capture probe summary

This is a **diagnostic mechanism probe** layered on top of `dynamic_selection_phase1`.

Previous heterogeneity result: `HETEROGENEITY-NOT-SUPPORTED`

## Explicit comparison
- **Current baseline**: `dynamic_selection_phase1` with no conflict-capture term (`capture_strength=0.0`).
- **Heterogeneity probe result**: `HETEROGENEITY-NOT-SUPPORTED`; subject-level scale variation alone did not rescue the locked flanker diagnostics.
- **Minimal conflict-capture probe result**: `CAPTURE-PROBE-NOT-SUPPORTED` with best bounded setting `capture_strength_0.30`.

## Bounded probe design
- only one effective free mechanism degree of freedom was probed: `capture_strength`
- fixed timing constants: `capture_midpoint_s=0.05` and `capture_tau_s=0.03`
- Stage-2 readout remained `baseline` and no urgency branch was introduced

## Best probe
- probe tag: `capture_strength_0.30`
- capture_strength: `0.3`
- verdict: `CAPTURE-PROBE-NOT-SUPPORTED`

## Metric deltas vs baseline
| Metric | Before distance (baseline→human) | After distance (capture-probe→human) | Improved |
|---|---:|---:|:---:|
| earliest_incongruent_caf | 0.254246 | 0.254246 | no |
| first_delta | 0.0433043 | 0.0433819 | no |
| incongruent_error_minus_correct_rt | 0.599547 | 0.599218 | yes |
| incongruent_conditional_tail | 0.396963 | 0.396963 | no |

## Execution
- Runner: `python code/scripts/run_minimal_conflict_capture_probe.py --mode full --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe`
- Analysis: `python code/scripts/analyze_minimal_conflict_capture_probe.py --input_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe`

This memo is a diagnostic mechanism probe summary and should not be treated as confirmatory evidence for a new model family.