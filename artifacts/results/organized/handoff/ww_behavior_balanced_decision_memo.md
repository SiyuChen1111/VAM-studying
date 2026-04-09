# WW Behavior-Balanced Decision Memo

## Scope

This memo evaluates the latest saved behavior-balanced WW smoke outputs under:

- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_baseline_behavior_balanced/`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_checkpoint_tail_focus_behavior_balanced/`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`

It separates:

- **Observed evidence**: values quoted directly from saved outputs
- **Interpretation**: what those values imply about the current bottleneck
- **Recommendation**: exactly one next path

## File status check

Confirmed present:

- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_baseline_behavior_balanced/smoke_eval_subset_meta.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_baseline_behavior_balanced/checkpoint_ranking_summary.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_baseline_behavior_balanced/metrics_smoke.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_checkpoint_tail_focus_behavior_balanced/smoke_eval_subset_meta.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_checkpoint_tail_focus_behavior_balanced/checkpoint_ranking_summary.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_checkpoint_tail_focus_behavior_balanced/metrics_smoke.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`

Relevant prior-context files also confirmed present:

- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_soft_hazard/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_urgency/summary_smoke.md`
- `results/organized/handoff/rt_readout_smoke_decision_memo.md`
- `results/organized/handoff/NEXT_STEP_DECISION_MEMO.md`

## Observed evidence

### 1. The behavior-balanced selector is now informative

The saved subset is no longer degenerate.

From `WW_baseline_behavior_balanced/smoke_eval_subset_meta.json`:

- `selected_eval_trials = 4096`
- `actual_human_error_trials = 654`
- `congruent_trials = 1892`
- `incongruent_trials = 2204`
- `balance_constraints_satisfied = 1`

The cross-run summary states:

- `Baseline subset mode: behavior_balanced`
- `WW_checkpoint_tail_focus subset mode: behavior_balanced`
- `Ranking tradeoff visible (baseline): True`
- `Ranking tradeoff visible (WW_checkpoint_tail_focus): True`
- `Ranking tradeoff visible overall: True`

So the previous question of whether the smoke subset was too weak is answered: it is now informative enough to expose real checkpoint tradeoffs.

### 2. Final checkpoint identity is still unchanged

From `comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`:

- `Baseline checkpoint: scale=0.5000, epoch=5`
- `WW_checkpoint_tail_focus checkpoint: scale=0.5000, epoch=5`
- `Baseline and WW_checkpoint_tail_focus selected the same checkpoint: True`

From both ranking summaries:

- `selected_checkpoint = { "scale": 0.5, "epoch": 5 }`

So selector-only checkpoint scoring still does **not** change the winner.

### 3. The selected checkpoint is still behaviorally wrong in the same core ways

From `WW_baseline_behavior_balanced/metrics_smoke.json` and the cross-run summary:

- `pred_mean = 0.4496` vs `true_mean = 0.6137`
- `pred_median = 0.4000` vs `true_median = 0.6000`
- `pred_skewness = 0.5912` vs `true_skewness = 28.0716`
- `pred_q95 = 0.8200`, `pred_median = 0.4000`, so `q95-q50 ≈ 0.4200` vs human `0.1442`
- `model_congruency_rt_gap = 0.1138` vs human `0.0278`
- `response_agreement = 0.8403`

These are not subtle misses. The selected solution remains too fast on central tendency, far too weakly right-skewed, and too large in congruency-sensitive RT separation.

### 4. `error_minus_correct_rt` is still undefined at the selected checkpoint

This is explicit in all relevant artifacts.

From `WW_baseline_behavior_balanced/metrics_smoke.json`:

- `pred_error_rt = NaN`
- `pred_correct_rt = 0.4496`
- `error_minus_correct_rt = NaN`
- `human_error_minus_correct_rt = -0.0652`

The comparison summary also states:

- `Baseline error_minus_correct_rt defined: False`
- `WW_checkpoint_tail_focus error_minus_correct_rt defined: False`

So the selected winner still does not even instantiate the error-vs-correct RT signal we care about.

### 5. Nearby checkpoints expose real tradeoffs, but none is already a clearly better candidate

From `WW_baseline_behavior_balanced/checkpoint_ranking_summary.json`:

#### Selected winner: `scale=0.5, epoch=5`

- `behavior_optimal_score = 0.5794`
- `rt_shape_score = 0.3547`
- `response_agreement = 0.9673`
- `mean_median_score = 0.6979`
- `congruency_score = 0.1948`
- `error_minus_correct_rt = NaN`
- `pred_mean = 0.4499`
- `pred_median = 0.4000`

#### Candidate with defined error RT structure: `scale=0.1, epoch=5`

- `behavior_optimal_score = 0.5735`
- `congruency_score = 0.2515` (better)
- `error_minus_correct_rt = 0.1572` (defined, but wrong sign vs human `-0.0265`)
- `pred_mean = 0.7128`
- `pred_median = 0.7200`
- `rt_shape_score = 0.2931` (worse)
- `response_agreement = 0.9583` (worse)

#### Candidate with better mean/median fit: `scale=0.3, epoch=5`

- `behavior_optimal_score = 0.5579`
- `mean_median_score = 0.8265` (better)
- `pred_mean = 0.5159`
- `pred_median = 0.4900`
- `congruency_score = 0.0`
- `error_minus_correct_rt = NaN`

These candidates show that the frontier is real, but they do **not** show that a hidden already-good checkpoint is simply being suppressed by single-winner export.

### 6. Prior evidence has already ruled out several easier alternatives

From `comparison_baseline_vs_soft_hazard/summary_smoke.md`:

- `Worth scaling up: False`

From `comparison_baseline_vs_urgency/summary_smoke.md`:

- `Worth scaling up: False`
- `Conclusion: reject B`

From prior handoff memos:

- `simple RT readout changes are not fixing the qualitative structure of the RT distribution`
- `the model still does not naturally produce a strongly human-like right-skewed RT distribution`

So this is not the first failed attempt to move behavior through a small scoring or readout-side change.

## Interpretation

### 1. What is the bottleneck now?

The bottleneck is **not** that the eval subset is too weak, and **not** that the selector is fully blind.

The bottleneck is that the current WW objective/selection landscape still prefers a checkpoint that:

- preserves response agreement reasonably well,
- stays relatively favorable on aggregate RT-shape score,
- but still fails to produce usable error-vs-correct RT structure,
- and still misses the key human RT signatures.

In short: the current pipeline can now **see** tradeoffs, but it still does not **reward** the desired behavior strongly enough to make a different checkpoint win.

### 2. Why Path B is not the best next mainline move

Path B would be useful if the main problem were only that single-winner export hides a likely-good candidate.

That is not what the saved outputs show.

We already have the candidate frontier in `checkpoint_ranking_summary.json`, and the top alternatives are informative but not compelling:

- `scale=0.1` gives a defined error-minus-correct RT, but with the wrong sign and with much later RTs
- `scale=0.3` improves mean/median fit, but still has `error_minus_correct_rt = NaN` and loses congruency fit

So top-K export would improve observability, but the saved evidence already suggests it would **not** by itself produce a genuinely different and behaviorally meaningful WW candidate.

### 3. Why Path C is too early

Path C would be justified if the evidence clearly showed that WW has reached structural dead-end status.

The evidence is not that strong yet.

- WW still exposes meaningful internal tradeoffs.
- The selected and nearby checkpoints are behaviorally different.
- The failure pattern still looks tied to what the current objective/checkpoint policy rewards, not to the absolute impossibility of better WW behavior.

So leaving WW now would be premature given the requirement to avoid replacing WW unless the evidence clearly supports it.

## Recommendation

### Best-supported next path

**Path A — Lightweight objective redesign inside WW**

### Exact recommendation

Make **one tiny objective change only**: add a **weak error-ordering term** that rewards producing a defined, human-direction error-vs-correct RT relationship on the behavior-balanced smoke evaluation target.

### Why this is the smallest justified next experiment

1. The selector is now informative, so more pure selector work is no longer the most decisive lever.
2. The selected winner still has `error_minus_correct_rt = NaN`, which is a sharper failure than merely being numerically off.
3. A nearby candidate (`scale=0.1`) shows that the model can produce a defined error-minus-correct RT quantity, but current scoring still does not favor it enough because other terms dominate.
4. A single weak error-ordering term is smaller than a mechanism redesign and more action-driving than top-K export alone.

### Chosen micro-objective

Use **one weak error-ordering term** only.

Target behavior:

- discourage checkpoints that leave `error_minus_correct_rt` undefined
- weakly reward checkpoints whose error-vs-correct RT difference moves toward the human sign/direction

This should be implemented as a small auxiliary behavior term, not a full loss redesign and not a multi-term package.

## Ranked conclusion

### Best-supported next path

1. **Path A — lightweight objective redesign inside WW**
   - best matches the new evidence that the selector can see tradeoffs but still picks a winner with undefined error RT structure

### Rejected alternative paths

2. **Path B — export / compare top-K only**
   - rejected as the mainline move because the ranking summaries already reveal the nearby frontier, and those nearby checkpoints do not already look like hidden clearly-better WW candidates

3. **Path C — WW mainline has reached diminishing returns**
   - rejected for now because the evidence is not yet strong enough to justify leaving WW; the current failure still looks more like insufficient reward shaping than definitive backbone exhaustion

## Minimal execution plan for the chosen path

1. **Keep WW dynamics unchanged.**
   - no backbone change
   - no readout-family change
   - keep `response_labels` supervision

2. **Add one weak error-ordering objective term only.**
   - no extra tail-loss bundle
   - no congruency-plus-tail-plus-error package
   - no checkpoint-policy rewrite in the same step

3. **Run one WW behavior-balanced smoke experiment on `20-29` only.**
   - do not run `80-89`
   - do not scale up yet

4. **Gate success on these concrete outcomes.**
   - selected checkpoint differs from the current `scale=0.5, epoch=5`
   - `error_minus_correct_rt` becomes defined at the selected checkpoint
   - RT scale does not collapse
   - congruency gap remains sensible
   - response agreement remains acceptable

5. **Compare only against the current behavior-balanced baseline.**
   - current selected checkpoint remains the direct reference point

## Final one-paragraph decision

The new behavior-balanced smoke evidence shows that the selector is now informative, but the final winner is still unchanged and still has `error_minus_correct_rt = NaN`. Because the ranking summaries already expose the nearby frontier, the next best move is not top-K export alone; and because WW still shows meaningful internal tradeoffs, it is too early to leave the line entirely. The smallest next change most likely to produce a genuinely different WW candidate is therefore **Path A: one weak error-ordering objective term inside the existing WW pipeline**, with everything else kept fixed.
