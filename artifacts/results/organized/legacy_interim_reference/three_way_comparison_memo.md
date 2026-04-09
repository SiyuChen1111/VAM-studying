# Three-way comparison memo

## Scope
This memo compares the three currently available branches:

1. **20-29 full-data branch** — interim best completed-scale result from the original full young-group Stage 2 run.
2. **20-29 matched-subject branch** — completed-scale result from the subject-count-matched young-group control branch (scale 0.1).
3. **80-89 final branch** — formal completed Stage 2 result.

The goal of this memo is to clarify what changed after introducing the matched-subject control branch, and what did **not** change.

## Current comparison table

| branch | status | scale | score | model_mean_rt | human_mean_rt | model_accuracy | human_accuracy | model_congruency_rt_gap | human_congruency_rt_gap | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20-29 full-data | interim_best_completed_scale | 0.1 | 0.5753 | 0.4870 | 0.6048 | 1.0000 | 0.9343 | 0.0814 | 0.0397 | scale 4/5, epoch 01/20 |
| 80-89 final | final | 0.2 | 0.6448 | 0.4331 | 0.8383 | 1.0000 | 0.9615 | 0.0950 | 0.0830 | formal Stage 2 result |
| 20-29 matched | completed_scale_0.1 | 0.1 | 0.6280 | 0.5350 | 0.6180 | 0.9999 | 0.9660 | 0.0895 | 0.0585 | completed scale 0.1 |

## Main findings

### 1. Subject-count matching clearly improved tractability
The matched 20-29 control branch reduced the young-group training burden dramatically.

- Full-data 20-29 Stage 2: roughly **9–20 minutes per epoch**
- Matched 20-29 Stage 2: roughly **~1.7–1.9 minutes per epoch** through the completed scale 0.1 run

This confirms that the original full-data young branch was slowed substantially by its much larger subject and trial count.

### 2. Matching did **not** eliminate the core modeling pathology
Even after matching the young group to the old-group subject-count structure, the model still shows:

- near-ceiling model accuracy,
- inflated model congruency RT gap relative to human behavior,
- and an overall decision regime that remains somewhat too efficient.

This means the current mismatch is **not** explained solely by the larger young-group sample size.

### 3. The shared pathology appears to be model-level, not just data-volume-level
Across all three branches, the model tends to be:

- too accurate,
- too efficient,
- and likely too fast relative to human behavior.

The 80-89 branch still shows the strongest temporal mismatch, but the matched 20-29 branch demonstrates that reducing young-group data volume does not automatically restore a fully human-like decision regime.

At the same time, the matched branch does improve fit quality relative to the original full-data young branch:

- full-data 20-29 score = **0.5753**
- matched 20-29 score = **0.6280**
- full-data 20-29 predicted mean RT = **0.487 s**
- matched 20-29 predicted mean RT = **0.535 s**

So matching does matter, but it does not solve the entire problem.

## Interpretation

The matched-subject control branch strengthens the current interpretation:

> The main problem is not just that the 20-29 branch had more subjects and therefore trained more slowly. The more important issue is that the current shared Stage 1 + Stage 2 fitting setup tends to drive the system into an overly efficient decision regime: highly accurate, conflict-sensitive, but insufficiently human-like in its temporal structure.

This is an important research result because it reduces the plausibility of a simple “sample-size confound only” explanation.

## What this does and does not show

### What it supports now
- Matching subject count is a useful and valid control branch.
- The sample-size imbalance mattered a lot for runtime.
- The sample-size imbalance does **not** appear to be the sole reason for the behavioral mismatch.

### What it does not yet prove
- It does not yet settle the final young-vs-old comparison.
- It does not yet identify the final parameter-level explanation.
- It does not yet replace the need for the formal behavior → parameter → mechanism analysis once the matched branch finishes more fully.

## Recommended next step
Continue the matched 20-29 branch beyond scale 0.1 so we can determine whether the current completed-scale result remains the branch best, and then extend the comparison from a strong control-branch interim result to a more stable multi-scale comparison.
