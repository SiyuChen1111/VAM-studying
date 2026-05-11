# Gate Decision: DMC apply_to=all_trials — Congruent-Trial Diagnostic

Date: 2026-05-06

## Question

Does extending DMC modulation to congruent trials (flanker=target) close the ~25pp response-agreement gap observed in the `incongruent_only` baseline?

## Method

- Smoke run: `train_dmc_var_ww_smoke.py --apply_to all_trials`
- Parameters: auto_strength=0.3, selection_strength=0.4 (same as best prior run a3_s4)
- 1024-trial behavior-balanced subset, 15 epochs WW training

## Results

| Metric | a3_s4 (incongruent_only) | a3_s4_all (all_trials) |
|--------|--------------------------|------------------------|
| Overall accuracy | 0.850 | 0.253 |
| Overall resp_agree | 0.751 | 0.245 |
| Incongruent resp_agree | 0.855 | 0.248 |
| Congruent resp_agree | 0.637 | 0.242 |

Applying DMC to congruent trials destroyed performance on BOTH trial types, reducing accuracy to chance level (0.25 for 4-class). The marginal choice distribution remained uniform — the model wasn't biased, just random.

## Mechanism

On congruent trials, flanker = target = class X. The DMC modulation applies:
- `flanker_mult`: 1 + auto(pulse) - selection(gate) → applied to class X
- `target_mult`: 1 - 0.5*auto(pulse) + 0.25*selection(gate) → also applied to class X

Net effect on class X late in the trial (selection_gate ≈ 1):
  multiplier ≈ (1 - 0.4) * (1 + 0.1) ≈ 0.66

The correct class evidence is reduced by ~34% while competing classes remain at baseline. The WW race then selects randomly among all classes.

## Verdict

**DMC_ON_ALL_KILL** — extending DMC modulation to congruent trials is not just unhelpful; it's catastrophic. The DMC mechanism fundamentally requires flanker ≠ target to function. When they're the same class, the boost-then-suppress pattern becomes pure suppression of the correct answer.

## Implication

The congruent-trial response-agreement gap (~64% vs ~85% on incongruent) cannot be closed by extending DMC. The gap is structural and requires a different approach — either:
1. A fundamentally different modulation scheme for congruent trials
2. A different accumulator/readout design that doesn't conflate flanker and target evidence
3. Accepting the congruent-trial gap as a known limitation of this architecture

This diagnostic is now closed. No further DMC parameter sweeps on congruent trials are warranted.
