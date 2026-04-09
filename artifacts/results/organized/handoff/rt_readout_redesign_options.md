# RT readout redesign options while preserving accumulation

## Goal

Preserve the accumulation-style mechanism while improving human-like RT distribution shape.

## Why readout is the likely bottleneck

Current evidence suggests the model can partially recover:
- response agreement
- accuracy trends
- congruency effects

but still fails to produce strongly human-like right-skewed RT distributions. This suggests the issue is not only the presence or absence of accumulation, but how RT is extracted from the accumulation trajectories.

## Option 1 — Decision-time-consistent readout

### Idea
Read choice and RT from the accumulator state **at the decision time**, not from a summary over all time points.

### Why keep it
- preserves accumulation logic
- preserves trajectory geometry
- is the smallest change from the current mechanism story

### Limitation
- may still be too rigid if the threshold-crossing rule remains overly hard / bounded

## Option 2 — Softer threshold crossing / hazard-like readout

### Idea
Still use accumulation trajectories, but convert them into a probability of crossing / committing at each time step.

### Why keep it
- preserves the temporal accumulation process
- can generate more realistic RT distributions without discarding mechanism structure

### Why this is promising
- avoids some hard boundary pile-up behavior
- gives a more distributional interpretation of decision time

## Option 3 — Two-stage readout from trajectories

### Idea
Keep the accumulator trajectories but let a small readout layer interpret them into RT / choice.

### Why keep it
- may preserve most of the mechanism while improving flexibility

### Limitation
- easiest route to hidden black-box behavior if unconstrained

## Recommended order

1. **First priority:** decision-time-consistent readout (already partly implemented in prototype v2)
2. **Second priority:** hazard-like / soft-bound readout using the same trajectories
3. **Third priority:** flexible learned readout, but only if the first two fail

## Recommendation

If the goal is to preserve the accumulation mainline while improving RT distributions, the best next step is:

> **Keep the recurrent accumulator and trajectory exactly as the latent mechanism, and redesign only the RT readout to be softer / more distribution-aware.**

This is the cleanest path to a more human-like RT distribution without abandoning the mechanism story.
