# Experiment log summary

## Scope

This file summarizes the major experimental branches explored in the last two days for the LIM / Flanker age-group project.

## 1. Cached-logit Stage 2 baseline

- Correct afternoon logits were identified and used as the valid Stage 1 artifacts.
- `20-29` and `80-89` Stage 2 runs were made observable with unbuffered logs and explicit MPS usage.
- Main finding: the pipeline could run, but behavior often drifted toward unrealistically high accuracy and imperfect RT distribution shape.

## 2. Subject-count-matched young control branch

- A matched `20-29` branch was created to mirror the `80-89` subject-count structure.
- Matched train users: `182, 899, 1478`
- Matched test user: `677`
- This dramatically reduced training cost and made controlled comparisons more feasible.

## 3. Response-label supervision correction

- Stage 2 choice supervision was changed from `target_labels` to `response_labels`.
- This improved the scientific validity of the training objective.
- Main finding: early checkpoints became more human-like, but later checkpoints often drifted back toward near-ceiling ground-truth accuracy.

## 4. Behavior-optimal checkpointing

- A new checkpoint selection logic was introduced.
- It now prioritizes:
  1. RT shape
  2. response agreement
  3. congruency gap
  4. mean / median RT
  5. ground-truth accuracy (weakly)
- Main finding: this made checkpoint selection more aligned with the research goal, even when it did not change the best scale itself.

## 5. RT-shape objective experiments

- Several targeted experiments were run:
  - increased time horizon / RT-shape-focused selection
  - quantile tail loss
  - anti-pileup loss
- Main finding: none of these objective-only changes clearly solved the human-like RT distribution problem.

## 6. `noise_ampa` mechanism probe

- A clean three-level `noise_ampa` probe was run on the matched `20-29` branch.
- Main finding: `noise_ampa` clearly changes behavior, but changing it alone does not solve the RT distribution shape problem.
- Interpretation: sampling noise matters, but it is not the whole answer.

## 7. Accumulator-RNN line

- Prototype v1 and v2 were implemented.
- v1 was unstable and not sufficiently accumulation-like.
- v2 was more stable but still failed to produce plausible RT dynamics.
- Current conclusion: this line is promising conceptually, but not yet competitive with the main WW-based path.

## 8. Current best high-level decision

The current most defensible line remains:

- **WW main path**
- `response_labels` supervision
- behavior-optimal checkpoint selection

The next meaningful advance should come from improving how RT is **read out** from the accumulation process, while preserving the accumulation-style mechanism.
