# Supervisor Update — 2026-04-08

## Scope

This note gives a concise update on the VGG + Wong-Wang / age-group modeling line after the recent readout, selector, prototype, and error-regime experiments.

## What we tried

### 1. Main WW age-group line
- Built the shared Stage-1 VGG + age-specific Stage-2 Wong-Wang pipeline for `20-29` and `80-89`.
- Main persistent issue: the fitted model was too accurate and too fast relative to human behavior.

### 2. Response-supervision correction
- Switched Stage-2 choice supervision from `target_labels` to `response_labels`.
- This made the objective more scientifically valid and improved congruency behavior.
- Limitation: the selected solution still tended to drift toward an overly idealized regime.

### 3. Readout-only rescue attempts
- Tested `soft_hazard` and `urgency` on matched `20-29` smoke runs.
- Result: neither rescued the RT-distribution problem.

### 4. Selector and eval redesign
- Added behavior-balanced eval subsets and behavior-focused checkpoint ranking.
- Result: the selector became informative, but it still did not change the final WW winner.

### 5. Lightweight WW objective tweak
- Tested `error_ordering`.
- Result: the selected WW checkpoint stayed unchanged and the model still had no useful error-conditioned RT structure.

### 6. Structured accumulator-RNN prototype
- Built and tested a VGG-backed accumulator prototype.
- Result: trajectories were interpretable and some statistics improved, but RT scale and response agreement failed too strongly.

### 7. Error-regime chain inside WW
- Shifted to RT-distribution-shape losses and joint shape+noise experiments.
- The most informative candidate was:
  - `cdf_wasserstein + fixed_noise_ampa = 0.06`
- This was the **first candidate with nonzero model errors** and defined `error_minus_correct_rt`.

## Main current conclusion

The local WW question has now been answered more clearly:

> WW can be pushed into an **error regime**, but the resulting regime is still **not human-like**.

Specifically, the best error-regime candidate is still:
- too fast relative to human RT
- too heavy-tailed
- directionally wrong on error-vs-correct RT relative to the current matched human data

Later tiny corrections (threshold correction, weak mean/median anchor) did not fix that mismatch.

## Most useful current artifact

The best scientific summary of the latest branch is:

- `results/organized/handoff/error_regime_experiment_chain_memo.md`

This memo shows that:
- shape-only losses can move the solution but do not create errors
- stronger internal noise plus shape supervision can create an error regime
- but the resulting regime remains miscalibrated relative to human behavior

## Practical recommendation

The best immediate next move is:

> **pause local WW patching and consolidate findings**

Rationale:
- readout-only changes failed
- selector-only changes failed
- behavior-balanced eval made the selector informative but did not change the winner
- one lightweight WW objective tweak failed
- the first structured prototype was informative but not viable
- the best WW error-regime candidate is scientifically useful, but not a final solution

## What has been cleaned up for GitHub sharing

- Added a dated root-level `logs.md` that reconstructs the full experiment chain.
- Added this supervisor update note.
- Kept the canonical mainline checkpoints/results needed for reproducibility.
- Removed exploratory smoke checkpoint trees and trivial status-marker files that were reproducible from scripts and fully summarized in existing memos.

## Suggested discussion point

The key research question now is no longer “how to make the model commit errors,” but:

> **how to make the model commit errors in a human-like RT regime**.

That may require either:
- a stronger next-line structured model, or
- a reframing of what should be optimized as the primary behavioral target.
