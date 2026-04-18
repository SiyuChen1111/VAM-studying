# Heterogeneity probe summary

This is a **diagnostic heterogeneity probe** on `dynamic_selection_phase1` (no new mechanism).

**Weighting**: trial-weighted (n=404725 selected trials)

## Selection rules
- eligibility: >=20 incongruent trials, >=3 incongruent error trials, >=10 unique RT values
- metrics: earliest incongruent CAF, incongruent error-minus-correct RT, RT skewness
- ranking: mean absolute within-age-group z-score (extreme_score)
- strata: at least one subject from lower and upper halves of earliest incongruent CAF rank
- tie-breaks: higher incongruent error count, then lexicographic user_id

Subject selection was deterministic and used earliest incongruent CAF, incongruent error-minus-correct RT, and **RT skewness** as the hybrid strata metrics.

## Selection / exclusion accounting
- 20-29: pool=`train_plus_test_fallback_due_to_test_only_insufficiency`, pool_users=12, test_only_users=3, selected=4, excluded=0
- 80-89: pool=`train_plus_test_fallback_due_to_test_only_insufficiency`, pool_users=4, test_only_users=1, selected=4, excluded=0

## Age-group centers and bounded scale grids
- 20-29: center=0.103017, grid=[0.08301724672317505, 0.09301724672317506, 0.10301724672317505, 0.11301724672317505, 0.12301724672317506], boundary_hits=4
- 80-89: center=0.093117, grid=[0.07311717748641967, 0.08311717748641968, 0.09311717748641968, 0.10311717748641967, 0.11311717748641968], boundary_hits=4

## Target mechanisms
- earliest incongruent CAF
- first delta quantile
- incongruent error-minus-correct RT
- incongruent conditional tail (q95 of incongruent-error RT)

## Contextual comparison outputs
- `reaggregated/contextual_metrics.csv` compares full age-group human aggregate vs full age-group baseline simulation vs reaggregated selected-subject simulation.
- `reaggregated/reaggregated_metrics.csv` contains the primary selected-subject human vs selected-subject simulation comparison.

## Success bar
**Verdict**: `HETEROGENEITY-NOT-SUPPORTED`

| Metric | Before distance (baseline→human) | After distance (single-subject→human) | Improved |
|---|---:|---:|:---:|
| earliest_incongruent_caf | 0.0870209 | 0.0870209 | no |
| first_delta | 0.0367337 | 0.0211957 | yes |
| incongruent_error_minus_correct_rt | 0.520383 | 0.5116 | yes |
| incongruent_conditional_tail | 0.994 | 0.994 | no |

## Deviation note
- Test-only selection could not satisfy the required 4 subjects per age group with the available prepared test splits, so the workflow used a documented pooled train+test fallback for subject selection and simulation.

This memo reports the verdict encoded by `reaggregated/success_bar.json` only.