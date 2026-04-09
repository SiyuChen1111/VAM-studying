# Research Log / Timeline

**Last updated:** 2026-04-08  
**Scope:** VGG → VGG+WW → age-group fitting → response-supervision → readout/selector/objective/error-regime experiments  
**Purpose:** A dated, supervisor-facing reconstruction of what was tried, in what order, and why each branch succeeded, stalled, or failed.

---

## How to read this log

- **[dated directly]** means the date is written inside the file itself.
- **[dated by file timestamp]** means the date is inferred from repository file modification time.
- **Interpretations** are grounded in the corresponding `.md` records and saved experiment summaries.
- Where exact day-level ordering is clear but exact wall-clock causality is not, I mark that honestly.

---

## Phase 0 — Initial VGG groundwork

### 2026-03-26 — first explicit VGG drift-rate work [dated directly]

Source:
- `logs.md` (original entry; this file previously contained only this day)

What happened:
- Created notebook-based VGG drift-rate exploration:
  - `vgg_drift_rate_complete.ipynb`
  - `vgg_drift_rate_fixed.ipynb`
- Added helper scripts for downloading / testing VGG16 and fixing feature dimensions.
- Key finding: VGG16 feature shape for the chosen image size was `25088`, not `8192`, and the VGG pipeline had to be corrected accordingly.

Why it mattered:
- This was the starting point for the whole “use VGG as visual evidence front-end” line.

Main limitation / failure at this stage:
- Still a notebook/prototyping stage, not yet the age-group behavioral modeling pipeline.

---

## Phase 1 — Formalizing the VGG+WW direction

### 2026-03-28 — early VGG+WW implementation planning [dated by file timestamp]

Sources:
- `Kar/VGG_WW.py`
- `.trae/documents/vgg_lstm_lim_implementation_plan.md`
- `.trae/documents/vgg_wongwang_lim_implementation_plan.md`

What happened:
- The repo moved from notebook-style VGG drift-rate ideas toward a proper LIM / Flanker modeling implementation.
- Two architectural directions were explicitly on the table:
  1. VGG + Wong-Wang
  2. VGG + LSTM / accumulator-style alternatives

Why it mattered:
- This is the first clean transition from “VGG drift-rate demo” to “real model family design.”

Main limitation / failure at this stage:
- Planning was ahead of empirical validation; no behavioral-age-group evidence yet.

---

## Phase 2 — Age-group data and first official WW training pipeline

### 2026-03-30 — age-group pipeline scaffolded [dated by file timestamp]

Sources:
- `prepare_age_group_data.py`
- `train_age_group_model.py`
- `train_age_group_stage2.py`
- `extract_age_group_logits.py`
- `research_proposal_v4.md`
- `research_plan.md`

What happened:
- Built the basic age-group split pipeline for LIM / Flanker.
- Established the core research framing:
  - shared Stage 1 visual front-end
  - age-specific Stage 2 decision dynamics
  - compare `20-29` vs `80-89`
  - internal noise (`noise_ampa`) as primary mechanism hypothesis

Why it mattered:
- This is when the project became a formal age-related WW decision-dynamics study, not just a VGG proof-of-concept.

Main limitation / failure at this stage:
- The initial training/evaluation line still leaned too much toward idealized accuracy rather than human-like RT behavior.

---

## Phase 3 — First full age-group WW results and early pathology diagnosis

### 2026-03-31 — first age-group post-analysis and formal baseline pathology [dated by file timestamp]

Sources:
- `run_age_group_post_analysis.py`
- `results/age_groups/analysis_summary.md`
- `archive/stage2_deprecated_2026-03-31/DEPRECATION_LOG.md`

What happened:
- First formal age-group Stage-2 outputs were analyzed.
- The now-classic pathology appeared:
  - model accuracy near ceiling
  - RT too fast
  - RT distribution shape not human-like enough

Why it mattered:
- This is the first point where the project had evidence that “plain WW fitting” was behaviorally off even when it ran successfully.

Main limitation / failure:
- Decision regime was too efficient: conflict-sensitive, but too fast and too accurate.
- Some earlier result files/logs from this period were later archived as deprecated/superseded.

---

## Phase 4 — Matched young control branch and interim age-group diagnosis

### 2026-04-01 — matched `20-29` control branch created [dated by file timestamp]

Sources:
- `extract_age_group_logits_fast.py`
- `create_matched_20_29_logits_subset.py`
- `results/age_groups_interim/three_way_comparison_memo.md`
- `generate_interim_age_group_report.py`
- `results/age_groups_interim/current_stage_conclusion_memo.md`

What happened:
- Built a **subject-count-matched** young branch to mirror the `80-89` structure.
- This reduced the young branch to a controlled matched comparison instead of a larger population estimate.
- Interim reports emphasized that both age groups were showing the same broad pathology:
  - too-fast RTs
  - too-high model accuracy
  - better relative congruency structure than absolute temporal realism

Why it mattered:
- This made comparisons much cleaner and reduced sample-size confounds in young-vs-old control analyses.

Main limitation / failure:
- The matched young branch is scientifically useful as a control, but small in subject count (later confirmed as 3 train + 1 test subjects).

---

## Phase 5 — Human-side proposal-aligned analysis and response-supervision transition

### 2026-04-01 to 2026-04-02 — proposal-aligned human summaries + response-supervision correction [dated by file timestamp]

Sources:
- `generate_proposal_aligned_behavior_figures.py`
- `results/organized/proposal_aligned_human_behavior/integrated_current_results_analysis.md`
- `results/age_groups_response_supervision_interim/response_supervision_interim_memo.md`
- `results/age_groups_response_supervision_frozen/frozen_current_best_memo.md`
- `results/organized/handoff/CURRENT_STATUS.md`

What happened:
- Human-side summary figures and notes were organized to anchor the real target behavior.
- The training target for Stage 2 was corrected from:
  - `target_labels`
  to:
  - `response_labels`

Why it mattered:
- This was a major scientific correction: the model should match human responses, not only ground-truth correctness.

Main gain:
- Congruency RT gap became more human-like.

Main limitation / failure:
- Even with response supervision, later checkpoints still drifted back toward near-ceiling accuracy.
- Better conflict structure did **not** automatically produce a human-like RT regime.

---

## Phase 6 — Explicit handoff logging and RT-readout redesign branch

### 2026-04-05 — organized handoff and readout-smoke experiments [dated by file timestamp]

Sources:
- `results/organized/handoff/logs.md`
- `vgg_wongwang_lim.py`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_soft_hazard/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_urgency/summary_smoke.md`
- `results/organized/handoff/rt_readout_smoke_decision_memo.md`

What happened:
- The repo entered a deliberately surgical “RT readout redesign” phase on matched `20-29` smoke.
- Readout-only variants tested:
  - `soft_hazard`
  - `urgency`

Results:
- `soft_hazard`: collapsed RT scale badly.
- `urgency`: preserved scale better, but worsened skew and error-vs-correct structure.

Why it mattered:
- This phase ruled out “just change readout” as the easy fix.

Main conclusion:
- **Readout-only patching failed.**

---

## Phase 7 — Behavior-balanced selector experiments

### 2026-04-06 (morning) — selector made informative, but winner unchanged [dated by file timestamp]

Sources:
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`
- `results/organized/handoff/ww_behavior_balanced_decision_memo.md`

What happened:
- Implemented behavior-balanced eval subset.
- Added checkpoint-tail-focus selection logic.
- Made ranking tradeoffs visible.

What succeeded:
- The selector was no longer blind.
- Ranking frontier became visible.

What failed:
- Final selected checkpoint still did not change.
- `error_minus_correct_rt` remained undefined.

Main conclusion:
- **Selector became informative but did not rescue the WW winner.**

---

## Phase 8 — First lightweight objective tweak and first next-line prototype

### 2026-04-06 (midday) — `error_ordering` and accumulator-RNN branch [dated by file timestamp]

Sources:
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_WW_error_ordering_behavior_balanced/summary_smoke.md`
- `train_age_group_accumrnn.py`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_vs_accumulator_rnn_behavior_balanced/summary_smoke.md`
- `results/organized/handoff/post_error_ordering_decision_memo.md`
- `results/organized/handoff/final_post_prototype_decision_memo.md`

What happened:
- Tried a first WW-side lightweight behavior loss: `error_ordering`.
- Also tested a structured accumulator-RNN prototype.

Results:
- `error_ordering`: did not move the selected WW solution at all.
- Accumulator-RNN prototype:
  - produced interpretable trajectories
  - produced defined `error_minus_correct_rt`
  - but collapsed RT scale and response agreement

Main conclusions:
- **One small WW objective tweak failed.**
- **The first structured next-line prototype was informative, but not viable.**

---

## Phase 9 — Mechanism probes within WW: noise, threshold, competition

### 2026-04-06 (afternoon/evening) — WW local mechanism probes [dated by file timestamp]

Sources:
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_WW_noise_probe_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_WW_threshold_probe_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_WW_competition_probe_behavior_balanced/summary_smoke.md`
- `results/age_groups_full_matched_compare/analysis_summary.md`

What happened:
- Probed three clean Stage-2 knobs:
  1. internal noise
  2. threshold
  3. competition scaling

Results:
- Each could move some secondary metrics.
- None could, by itself, move the model into a useful human-like error regime.

Related age-group comparison work on the same date:
- Added `run_matched_full_age_group_analysis.py`
- produced KDE + trajectory geometry figures comparing matched `20-29` vs `80-89`

Main conclusion:
- **Single WW scalar / geometry knobs were not enough.**

---

## Phase 10 — RT-distribution-shape loss branch

### 2026-04-06 to 2026-04-07 — shape-first objective family [dated by file timestamp]

Sources:
- `checkpoints_age_groups_rtshape_kl/20-29/smoke/comparison_WW_baseline_vs_WW_rt_dist_kl_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtshape_cdf/20-29/smoke/comparison_WW_baseline_vs_WW_rt_dist_cdf_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtshape_conditional/20-29/smoke/comparison_WW_baseline_vs_WW_rt_dist_conditional_behavior_balanced/summary_smoke.md`

What happened:
- Shifted from pointwise RT fitting to shape fitting.
- Tried three RT-distribution losses:
  1. `soft_hist_kl`
  2. `cdf_wasserstein`
  3. conditional congruency-CDF matching

What succeeded:
- All three changed the selected checkpoint.
- This proved that shape-aware losses have optimization signal.

What failed:
- None created a nonzero-error regime by themselves.
- None beat baseline behaviorally.

Main conclusion:
- **Shape-only losses can move the solution, but not far enough.**

---

## Phase 11 — Error-regime chain: shape + stronger noise + local corrections

### 2026-04-07 to 2026-04-08 — the full error-regime experiment chain [dated by file timestamp]

Primary source:
- `results/organized/handoff/error_regime_experiment_chain_memo.md`

Supporting run summaries:
- `checkpoints_age_groups_rtshape_noise_joint/20-29/smoke/comparison_WW_baseline_vs_WW_rt_dist_cdf_noise_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtshape_noise_joint_hi/20-29/smoke/comparison_WW_baseline_vs_WW_rt_dist_cdf_noise_hi_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtshape_noise_threshold_joint/20-29/smoke/comparison_WW_baseline_vs_WW_rt_dist_cdf_noise_thr_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtshape_noise_anchor_joint/20-29/smoke/comparison_WW_baseline_vs_WW_rt_dist_cdf_noise_anchor_behavior_balanced/summary_smoke.md`

What happened:

#### 11.1 `cdf_wasserstein + fixed_noise_ampa = 0.04`
- first joint shape+noise setup
- improved shape slightly
- still no model errors

#### 11.2 `cdf_wasserstein + fixed_noise_ampa = 0.06`
- first candidate with:
  - nonzero model errors
  - defined `error_minus_correct_rt`
  - acceptable response agreement
  - interpretable trajectories
- but it was still too fast and directionally wrong relative to current human error-conditioned RT behavior

#### 11.3 threshold correction on top of `noise=0.06`
- preserved error regime
- worsened RT scale further

#### 11.4 weak mean/median anchor on top of `noise=0.06`
- preserved error regime
- did not restore human-like directionality

Main conclusion:
- **WW can be pushed from a no-error regime into an informative error regime.**
- But the resulting regimes are still **not human-like**:
  - too fast
  - too heavy-tailed
  - error-vs-correct direction wrong

This is the key local answer of the whole project so far.

---

## 2026-04-08 — final high-level decision [dated by file timestamp + memo content]

Primary sources:
- `results/organized/handoff/final_post_prototype_decision_memo.md`
- `results/organized/handoff/error_regime_experiment_chain_memo.md`

Final high-level decision reached in the repo:

> **Pause local WW-side patching and consolidate findings.**

Reason:
- readout-only fixes failed
- selector-only fixes failed
- behavior-balanced eval made the selector informative but did not change the winner
- one lightweight objective tweak failed
- the first structured accumulator prototype was informative but not viable
- the best WW error-regime candidate is scientifically useful, but still not a human-like solution

---

## Condensed list of major attempts and how they failed

### A. VGG drift-rate notebooks / early VGG groundwork
- useful for establishing the visual front-end
- not yet a full behavioral model

### B. Baseline VGG+WW age-group fitting
- runs successfully
- too fast
- too accurate
- insufficiently human-like RT distributions

### C. Response-supervision correction
- improves scientific validity and congruency behavior
- still drifts toward idealized choice behavior

### D. Readout-only WW patches (`soft_hazard`, `urgency`)
- one collapses RT scale
- the other preserves scale better but worsens shape/error structure

### E. Selector / checkpoint redesign
- makes tradeoffs visible
- does not change the final WW winner

### F. One lightweight WW objective tweak (`error_ordering`)
- leaves selected solution unchanged

### G. Structured accumulator-RNN prototype
- trajectories interpretable
- response agreement and RT scale fail badly

### H. WW local scalar probes (noise / threshold / competition)
- move secondary metrics
- do not by themselves create a useful human-like error regime

### I. RT-distribution-shape losses
- move the selected checkpoint
- still cannot create errors on their own

### J. Joint shape + stronger noise
- finally creates an error regime
- but error regime is directionally wrong relative to current human data

### K. Post-error-regime local corrections
- preserve the gain
- do not fix scale/direction mismatch

---

## What we know now

1. **The project is no longer blocked on “how to get any errors.”**
   We now know at least one way to create model errors.

2. **The harder remaining problem is directional calibration.**
   The issue is not just to create an error regime, but to create a **human-like** error regime.

3. **Further tiny WW-side patches are likely low-value.**
   The evidence chain is already rich enough to justify consolidation rather than more ad hoc tuning.

---

## Recommended use of this log

For GitHub / supervisor reporting, this file should be read together with:

- `results/organized/handoff/error_regime_experiment_chain_memo.md`
- `results/organized/handoff/final_post_prototype_decision_memo.md`
- `results/organized/handoff/CURRENT_STATUS.md`
- `results/organized/proposal_aligned_human_behavior/integrated_current_results_analysis.md`

These provide, respectively:

1. the detailed WW error-regime chain,
2. the final decision point after the first prototype,
3. the current trusted branch structure,
4. the human-side behavioral anchor.

---

## One-paragraph executive summary

From 2026-03-26 onward, the project moved from initial VGG drift-rate prototyping into a formal VGG+Wong-Wang age-group modeling pipeline for LIM / Flanker behavior. The baseline model repeatedly showed the same pathology: high accuracy, conflict sensitivity, but RT distributions that were too fast and not human-like enough. Subsequent work systematically tested response-label supervision, subject-count-matched young controls, readout-only WW patches, selector and eval redesign, one lightweight WW objective tweak, a structured accumulator-RNN prototype, a series of RT-distribution-shape losses, and finally a full error-regime experiment chain combining shape supervision with stronger internal noise. The project’s strongest local finding is that WW can be pushed into an informative nonzero-error regime, but the resulting regime remains too fast and directionally wrong relative to human error-conditioned RT behavior, and later tiny corrections did not fix that mismatch. The evidence now supports consolidation rather than continued local WW patching.
