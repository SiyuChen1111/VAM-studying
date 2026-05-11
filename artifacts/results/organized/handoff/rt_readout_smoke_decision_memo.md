# RT Readout Smoke Decision Memo

> **Historical status note (2026-05-02):** This memo records the readout-smoke phase and its local recommendation. It is still useful as evidence that readout-only fixes failed, but it is not the current repo-wide next-step document. Later branch-level status is governed by `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`.

## Scope

This memo synthesizes the saved smoke-test outputs for the `20-29` matched branch under `checkpoints_age_groups_rtreadout/20-29/smoke/`.

It separates:

- **Observed evidence**: metrics quoted directly from saved outputs
- **Interpretation**: what those patterns suggest
- **Recommendation**: the single best next direction

## File status check

Confirmed present:

- `checkpoints_age_groups_rtreadout/20-29/smoke/baseline/metrics_smoke.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/soft_hazard/metrics_smoke.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/B_urgency/metrics_smoke.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_soft_hazard/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_urgency/summary_smoke.md`

Not found in the inspected smoke tree:

- `checkpoints_age_groups_rtreadout/20-29/smoke/C_noisy_readout/`
- any `comparison_baseline_vs_noisy_readout/summary_smoke.md`

So Experiment C cannot be evaluated from the currently saved outputs.

## Observed evidence

### Comparison table

| Metric | Baseline | A: soft_hazard | B: urgency | C: noisy_readout | Human | Evidence source |
|---|---:|---:|---:|---:|---:|---|
| Mean RT | 0.7124 | 0.0529 | 0.7175 | Missing | 0.6256 | `baseline/metrics_smoke.json`, `soft_hazard/metrics_smoke.json`, `comparison_baseline_vs_urgency/summary_smoke.md` |
| Median RT | 0.7100 | 0.0538 | 0.7400 | Missing | 0.6010 | same as above |
| Predicted skewness | -0.1843 | -1.2633 | -0.6799 | Missing | 75.5031 | comparison summaries |
| Tail spread (q95-q50) | 0.2800 | 0.0036 | 0.2200 | Missing | 0.1490 | comparison summaries |
| Error minus correct RT | 0.2801 | 0.0009 | -0.4414 | Missing | -0.0651 | comparison summaries |
| Congruency gap | 0.0944 | 0.0009 | 0.0592 | Missing | 0.0261 | comparison summaries |
| Response agreement | 0.9401 | 0.9318 | 0.9423 | Missing | N/A | comparison summaries |

### Per-experiment observations

#### Baseline

- Baseline keeps reasonable scale but is still slower than human on mean RT: `0.7124` vs `0.6256`.
- Baseline skew is negative (`-0.1843`) while the human reference is strongly right-skewed.
- Baseline error-minus-correct RT is positive (`0.2801`) while the human value is slightly negative (`-0.0651`).
- Baseline congruency gap is too large: `0.0944` vs human `0.0261`.

#### A = soft_hazard

- Mean RT collapses to `0.0529`.
- Tail spread collapses to `0.0036`.
- Congruency gap collapses to `0.0009`.
- Response agreement drops slightly to `0.9318`.
- Saved summary conclusion: `Worth scaling up: False`.

#### B = urgency

- Mean RT stays close to baseline: `0.7175` vs baseline `0.7124`.
- Median RT also stays plausible: `0.7400` vs baseline `0.7100`.
- Response agreement is preserved: `0.9423` vs baseline `0.9401`.
- But skew gets worse: `-0.6799` vs baseline `-0.1843`.
- Error-minus-correct RT gets much worse: `-0.4414` vs baseline `0.2801` and human `-0.0651`.
- Saved summary conclusion: `reject B`.

#### C = noisy_readout

- No saved C outputs were found in the inspected smoke directory.
- Therefore there is no observed evidence for or against noisy readout in this handoff state.

## Interpretation

### 1. Did any of A/B/C improve RT distribution meaningfully?

- **A:** No. It destroys RT scale and tail behavior.
- **B:** No. It preserves scale but worsens skew and error-vs-correct ordering.
- **C:** Unknown from current saved artifacts, because the expected output directory and summaries are missing.

### 2. Which failure mode is now most likely the core bottleneck?

The strongest pattern from saved A/B evidence is that **simple RT readout changes are not fixing the qualitative structure of the RT distribution**.

- A changes timing aggressively and collapses too early.
- B changes timing conservatively and preserves scale, but the shape and error dynamics still move in the wrong direction.

That suggests the main problem is not just “commit slightly earlier/later” or “add a bit of urgency.”

### 3. What bottleneck is most plausible?

Based on the saved outputs, the most plausible bottleneck is:

- **not primarily threshold timing**: A shows that timing-style control alone can catastrophically collapse scale.
- **not yet proven to be readout stochasticity**: C is missing, so there is no saved evidence that noise in the readout helps.
- **most likely a mismatch between accumulation geometry and RT decoding** within the current WW mainline.

In practical terms, the baseline evidence trajectory appears to support good-enough response agreement, but the decoder still produces the wrong skew, wrong error-vs-correct relationship, and too-large congruency gap.

### 4. Is the problem already structural enough to leave WW now?

Not yet, based on the currently saved evidence.

Reason:

- We only have confirmed saved failures for two readout-only variants.
- We do **not** have saved C outputs.
- Baseline still performs reasonably on scale and response agreement, which means the WW backbone is not obviously unusable.

So the evidence is strong enough to reject A and B, but **not strong enough to justify replacing WW immediately**.

## Recommendation

### Single best next direction

**Path 3 — return to the WW mainline with a more targeted experiment on checkpoint selection / RT objective redesign / richer behavior-optimal scoring.**

This is the best-supported next step from the currently saved evidence.

### Why this recommendation

1. **A and B both fail.**
   - A fails the basic RT-scale test.
   - B passes the RT-scale gate but still worsens the key behavioral structure.

2. **C is missing evidence.**
   - There is no saved basis for recommending scale-up of noisy readout.

3. **Baseline remains the strongest fully evidenced option.**
   - It is still wrong on skew/error/congruency structure, but it is the best-supported current model in the saved artifact state.

4. **The next likely gain is from better optimization target selection, not another quick readout patch.**
   - The current scoring already tracks several behaviors, but the observed failures suggest mean/scale preservation is easier than getting the correct qualitative RT geometry.

## Ranked conclusion

### Best candidate

1. **Baseline WW mainline**
   - Not because it is good enough, but because it is the least damaging and best-supported current result.

### Rejected candidates

2. **B: urgency**
   - Rejected because it preserves scale but worsens skew and error-vs-correct behavior.

3. **A: soft_hazard**
   - Rejected because it collapses RTs far too early and destroys tail behavior.

4. **C: noisy_readout**
   - Not ranked as a winner or loser from this handoff state because the expected saved outputs are missing.

### Next recommended direction

**WW mainline, with a targeted objective/checkpoint experiment on the same matched `20-29` setup before any full-data or `80-89` work.**

## Minimal execution plan

1. **Audit the current WW smoke selection logic.**
   - Reuse the existing matched `20-29` smoke subset.
   - Keep the same validated cached logits and response supervision.

2. **Define one tighter WW-side experiment.**
   - Prioritize one of:
     - checkpoint selection refinement using RT-shape/error-ordering first,
     - RT objective redesign that explicitly penalizes wrong error-vs-correct ordering,
     - richer behavior-optimal scoring that weights skew/tail/error structure more heavily than mean RT.

3. **Run only one WW-targeted smoke experiment first.**
   - Do not run `80-89`.
   - Do not scale beyond `20-29` until smoke evidence shows improvement over baseline on the qualitative metrics.

4. **Gate success on these metrics together.**
   - RT scale remains plausible.
   - Skew/tail improve relative to baseline.
   - Error-vs-correct moves toward the human sign/magnitude.
   - Congruency gap moves toward human while response agreement stays acceptable.

5. **Optional follow-up before committing further.**
   - If C outputs exist elsewhere, recover them and revisit this memo before closing the readout-only branch completely.

## Final recommendation in one sentence

With the current saved evidence, the best next move is **not** another quick readout mechanism, but a **WW-mainline smoke experiment focused on checkpoint/objective redesign**, while treating noisy-readout as **missing evidence** until its artifacts are actually available.
