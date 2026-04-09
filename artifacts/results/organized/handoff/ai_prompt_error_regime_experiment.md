# Coding Prompt — Drive Human-Like Error Responses and Behavioral Signatures

## Task Prompt for Agent

You are implementing the **next concrete experiment** in this repository.

### Objective
The next goal is **not** to force the model to fully match human RT skew yet.

Instead, the goal is to move the model into a regime where it can produce **human-like erroneous responses**, and then test whether those errors are accompanied by the key behavioral RT signatures from `research_proposal_v4.md`.

### Why this is the next step
Current evidence shows:

- the current WW baseline already has:
  - some RT variability
  - mild right-skew
  - a congruent/incongruent RT effect
- but it still produces:
  - **no model errors** at the selected checkpoint
  - therefore no model error RT distribution
  - therefore no meaningful test of the **error-slower** signature

So the bottleneck is now:

> The model is not yet entering a human-like **error regime**.

Without that, several of the proposal’s key behavioral signatures cannot even be tested properly.

---

## Scientific target

According to `research_proposal_v4.md`, the most relevant behavioral signatures now are:

- stochastic responses
- right-skewed RT
- error-slower
- congruent vs incongruent RT difference
- interpretable accumulation trajectories under these conditions

The immediate priority is:

> Make the model produce **nontrivial errors** while preserving plausible RT behavior.

---

## High-Level Constraints

Keep unchanged where possible:

- matched `20-29` branch
- corrected logits / corrected data mapping
- VGG backbone / current prototype family
- smoke-scale regime only
- behavior-balanced evaluation setup
- trajectory export and analysis capability

Do **not**:

- jump to full-data or `80-89`
- build a new architecture family
- refactor broadly
- optimize for perfect human skew first
- add many objectives at once

Prefer **one minimal intervention** that can move the model into a more human-like error regime.

---

## Core Question

The purpose of this experiment is to answer:

> Can we make the model produce a realistic amount of errors, without destroying RT scale and response behavior, so that the key behavioral signatures in the proposal become testable?

---

## Choose one intervention only

Implement **exactly one** of the following directions, choosing the one that is smallest and cleanest in the current codebase:

### Option A — relax overconfident decision behavior
Introduce a minimal mechanism that prevents the model from becoming effectively perfect too early.

Examples:
- softer choice temperature
- weaker winner dominance
- slightly stronger internal noise
- slightly weaker evidence gain

### Option B — calibrate the decision regime
Introduce one small change that increases the chance of near-threshold competition.

Examples:
- threshold shift
- noise scaling
- evidence scaling
- competition scaling

### Option C — response-focused lightweight training adjustment
Use one very small adjustment to make the model less over-aligned with target correctness and more behavior-like.

Examples:
- keep `response_labels` as main supervision target
- reduce over-strong choice dominance
- very small balancing of RT vs choice terms

### Rule
Choose **only one** intervention.
Do **not** combine several.

---

## Implementation Plan

### 1. Work in the current prototype family
Use the current VGG-based path that is already closest to producing trajectories and RT predictions.

Do **not** switch to a totally new model family.

### 2. Preserve smoke-scale evaluation
Keep using:
- matched `20-29`
- behavior-balanced smoke eval subset
- current artifact style
- trajectory export

### 3. Add just enough logging to verify the key new question
You must explicitly measure:

- number of model errors
- whether model error RT distribution exists
- whether `error_minus_correct_rt` becomes defined
- whether response agreement stays acceptable
- whether RT scale collapses or stays plausible

### 4. Save trajectory artifacts
This remains mandatory.

Save enough to inspect:
- correct vs error trajectories
- congruent vs incongruent trajectories
- winner vs runner-up separation over time

---

## Execution

### 5. Run one baseline + one candidate only

Run:

1. existing baseline anchor
2. one candidate with the single chosen intervention

Suggested directories:
- baseline:
  - existing relevant baseline smoke path
- candidate:
  - new smoke path clearly named by the intervention

Comparison output:
- one comparison directory with figures and summary

Do **not** run multiple candidate variants.

---

## Analysis Requirements

### 6. Reuse existing analysis and extend minimally
Use the current analyzer and only add what is necessary.

Required outputs:
- RT distribution comparison
- correct vs error RT plot
- congruent vs incongruent RT plot
- trajectory summary / plots
- `summary_smoke.md`

### 7. Required reporting
The summary must explicitly state:

- model error count
- human error count
- whether `error_minus_correct_rt` is defined
- whether the model still preserves:
  - plausible RT scale
  - acceptable response agreement
- whether the model now shows:
  - nontrivial errors
  - error-related RT structure
  - proposal-relevant behavioral signatures

---

## Success Criteria

This experiment is successful if it achieves **all** of the following at least minimally:

1. the model produces nontrivial errors
2. `error_minus_correct_rt` becomes defined
3. RT scale remains plausible
4. response agreement does not collapse catastrophically
5. at least some proposal-relevant signatures become more testable:
   - right-skewed RT
   - error-slower or at least defined error-vs-correct RT
   - congruent vs incongruent RT difference
   - interpretable trajectories

If the model only produces errors by collapsing accuracy/response agreement or destroying RT scale, reject it.

---

## Deliverables

Produce:

1. minimal code changes for one chosen intervention
2. one baseline smoke comparison
3. one candidate smoke run
4. RT distribution figures
5. correct vs error figures
6. trajectory artifacts / trajectory summaries
7. `summary_smoke.md`
8. short conclusion:
   - **error regime achieved and worth continuing**
   - or **errors only achieved by breaking the model**
   - or **still no useful error regime**

---

## Verification Requirements

Before finishing:

- run diagnostics on changed files
- run the smoke experiment successfully
- confirm output files exist
- quote concrete metrics from saved outputs
- explicitly state:
  - model error count
  - whether `error_minus_correct_rt` is defined
  - whether RT scale remained plausible
  - whether response agreement degraded
  - whether the model now reproduces more of the proposal’s core behavioral signatures
- separate pre-existing issues from new issues

---

## Important Guidance

- Do not chase perfect human skew yet.
- The immediate goal is to make the model enter a **behaviorally realistic error regime**.
- Once that exists, the proposal’s key RT signatures become meaningfully testable.
- Keep the intervention minimal and interpretable.
