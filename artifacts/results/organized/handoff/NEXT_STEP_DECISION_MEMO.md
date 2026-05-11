# Next-step decision memo

> **Historical status note (2026-05-02):** This memo records an earlier local recommendation from before the bounded successor-screening bundle under `artifacts/results/rt_model_next_step/` was executed. It should not be read as the current repo-wide next-step decision. The current authoritative successor-screening outcome is `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`, which concludes `NO_SUCCESSOR_BRANCH_CLEARED_GATES` and requires a **new planning phase** before any further successor-branch design work.

## Purpose

This memo summarizes the main experimental lines we tried, what each one taught us, and which line should be kept as the current main path for the next stage.

## Line 1 — Wong-Wang main path with response-label supervision

### What we changed
- switched Stage 2 choice supervision from `target_labels` to `response_labels`
- added response-aware logging and current-best checkpoint selection
- ran matched `20-29` and `80-89`

### What we learned
- this line can produce much more human-like response behavior than the older target-supervision setup
- `response_agreement` became meaningful
- congruency gap often becomes much closer to the human target
- however, the model still tends to drift back toward overly high ground-truth accuracy

### Decision
**Keep as current main path.**

### Why
This is the only line that currently preserves the intended evidence-accumulation mechanism while still producing interpretable behavior-level progress.

---

## Line 2 — RT-shape-only objective tweaks

### Variants tried
- increased `time_steps_factor`
- enabled stronger RT-shape-focused checkpointing
- small quantile tail loss
- anti-pileup / objective-layer probes

### What we learned
- simply extending the RT horizon did not fix the distribution shape
- a small quantile tail loss was better than the pure horizon trick, but still did not outperform the simpler behavior-focused baseline
- anti-pileup loss also failed to outperform the best simpler configuration

### Decision
**Deprioritize as primary path.**

### Why
These interventions are informative as diagnostics, but by themselves they do not solve the core problem.

---

## Line 3 — Fixed `noise_ampa` probe

### What we changed
- kept the main behavior-oriented training configuration
- froze `noise_ampa` at a higher value

### What we learned
- `noise_ampa` clearly affects the model behavior
- it can reduce the over-idealized accuracy regime in some checkpoints
- but increasing noise alone does not automatically produce human-like RT distributions

### Decision
**Keep as mechanism probe, but not as current optimization path.**

### Why
It remains useful for testing the mechanistic role of sampling noise, which is central to the proposal, but it is not yet the best route to behavior-level improvement.

---

## Line 4 — `accumulator-RNN` prototype v1

### What we changed
- replaced the Wong-Wang backend with a first recurrent accumulator-style prototype

### What we learned
- v1 either became numerically unstable or failed to recover the expected conflict-sensitive timing structure
- its readout and recurrence were still too generic to count as a trustworthy accumulator model

### Decision
**Abandon v1.**

### Why
It is not a strong enough foundation to continue debugging when the current Wong-Wang main path is already more interpretable and more behaviorally useful.

---

## Line 5 — `accumulator-RNN` prototype v2

### What we changed
- made readout more decision-time-consistent
- made recurrent update more accumulation-like

### What we learned
- v2 was more stable than v1
- but RT behavior remained far from realistic
- best result still had grossly implausible RTs and poor congruency timing structure

### Decision
**Pause / abandon as current main path.**

### Why
The idea may still be worth revisiting later, but in its current form it is not competitive with the response-supervision Wong-Wang line.

---

## Current recommendation

### Keep
- **Wong-Wang + response-label supervision + behavior-optimal checkpoint selection**

### Use as secondary diagnostics
- fixed `noise_ampa` probes
- limited objective-layer probes when a very specific behavioral question needs to be answered

### Do not prioritize right now
- pure RT-shape-only loss tuning
- accumulator-RNN v1/v2 replacement line

## Most important unresolved scientific issue

The model can now become more human-like in response behavior and conflict structure, but it still does not naturally produce a strongly human-like right-skewed RT distribution. This means the current best path is behaviorally improved but not yet distributionally satisfactory.

## Recommended next step

Stay on the Wong-Wang main path, keep `response_labels`, keep behavior-optimal checkpointing, and treat `noise_ampa` as the most meaningful next mechanism probe if the explicit scientific goal is to test whether sampling noise shapes RT distribution tails.
