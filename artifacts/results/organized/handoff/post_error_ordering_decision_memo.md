# Post-Error-Ordering Decision Memo

> **Historical status note (2026-05-02):** This memo captures the decision state immediately after the `error_ordering` and early prototype phase. Its recommendation to transition into successor planning has since been superseded by the completed bounded successor-screening bundle under `artifacts/results/rt_model_next_step/`. For current repo-wide status, use `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`.

## Scope

This memo synthesizes the saved WW smoke evidence through `WW_error_ordering_behavior_balanced` and chooses the single best next mainline path.

It separates:

- **Observed evidence**
- **Interpretation**
- **Recommendation**

## File status check

Confirmed present:

- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_soft_hazard/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_urgency/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_WW_error_ordering_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_baseline_behavior_balanced/smoke_eval_subset_meta.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_baseline_behavior_balanced/metrics_smoke.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_baseline_behavior_balanced/checkpoint_ranking_summary.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_error_ordering_behavior_balanced/metrics_smoke.json`
- `checkpoints_age_groups_rtreadout/20-29/smoke/WW_error_ordering_behavior_balanced/checkpoint_ranking_summary.json`
- `results/organized/handoff/rt_readout_smoke_decision_memo.md`
- `results/organized/handoff/ww_behavior_balanced_decision_memo.md`
- `results/organized/handoff/NEXT_STEP_DECISION_MEMO.md`
- `results/organized/handoff/rt_readout_redesign_options.md`

## Observed evidence

### 1. Readout-only fixes are already ruled out

From `comparison_baseline_vs_soft_hazard/summary_smoke.md`:

- `Predicted skewness: baseline=-0.1843, soft_hazard=-1.2633, human=75.5031`
- `Tail spread q95-q50: baseline=0.2800, soft_hazard=0.0036, human=0.1490`
- `Error minus correct RT: baseline=0.2801, soft_hazard=0.0009, human=-0.0651`
- `Worth scaling up: False`

From `comparison_baseline_vs_urgency/summary_smoke.md`:

- `Mean RT: baseline=0.7124, Urgency=0.7175, human=0.6256`
- `Predicted skewness: baseline=-0.1843, Urgency=-0.6799, human=75.5031`
- `Error minus correct RT: baseline=0.2801, Urgency=-0.4414, human=-0.0651`
- `Worth scaling up: False`
- `Conclusion: reject B`

So the repository has already ruled out the two simplest readout-side moves.

### 2. The behavior-balanced selector is now informative

From `WW_baseline_behavior_balanced/smoke_eval_subset_meta.json`:

- `selected_eval_trials = 4096`
- `actual_human_error_trials = 654`
- `congruent_trials = 1892`
- `incongruent_trials = 2204`
- `balance_constraints_satisfied = 1`

From `comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`:

- `Ranking tradeoff visible (baseline): True`
- `Ranking tradeoff visible (WW_checkpoint_tail_focus): True`
- `Ranking tradeoff visible overall: True`

So the remaining null results are not explained by a weak or degenerate smoke subset.

### 3. Selector-only changes did not move the selected checkpoint

From `comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`:

- `Baseline checkpoint: scale=0.5000, epoch=5`
- `WW_checkpoint_tail_focus checkpoint: scale=0.5000, epoch=5`
- `Baseline and WW_checkpoint_tail_focus selected the same checkpoint: True`
- `Baseline error_minus_correct_rt defined: False`
- `WW_checkpoint_tail_focus error_minus_correct_rt defined: False`
- `Worth scaling up: False`
- `Conclusion: reject and stay baseline`

So making the selector more behavior-aware did not change the winner.

### 4. The lightweight objective tweak also failed to move the selected solution

From `comparison_WW_baseline_vs_WW_error_ordering_behavior_balanced/summary_smoke.md`:

- `Baseline checkpoint: scale=0.5000, epoch=5`
- `WW_error_ordering checkpoint: scale=0.5000, epoch=5`
- `Mean RT: baseline=0.4496, WW_error_ordering=0.4496, human=0.6137`
- `Median RT: baseline=0.4000, WW_error_ordering=0.4000, human=0.6000`
- `Predicted skewness: baseline=0.5912, WW_error_ordering=0.5912, human=28.0716`
- `Tail spread q95-q50: baseline=0.4200, WW_error_ordering=0.4200, human=0.1442`
- `Error minus correct RT: baseline=nan, WW_error_ordering=nan, human=-0.0652`
- `Congruency gap: baseline=0.1138, WW_error_ordering=0.1138, human=0.0278`
- `Response agreement: baseline=0.8403, WW_error_ordering=0.8403`
- `WW_error_ordering error_minus_correct_rt defined: False`
- `Baseline and WW_error_ordering selected the same checkpoint: True`
- `Worth scaling up: False`
- `Conclusion: reject and stay baseline`

This is the strongest new fact: the first targeted WW-side objective tweak did not just fail to beat baseline; it left the selected checkpoint and selected summary behavior effectively unchanged.

### 5. The selected WW solution is still missing the same key behavioral signal

From `WW_baseline_behavior_balanced/metrics_smoke.json` and `WW_error_ordering_behavior_balanced/metrics_smoke.json`:

- `pred_mean = 0.44960448145866394` vs `true_mean = 0.6136533617973328`
- `pred_median = 0.3999999761581421` vs `true_median = 0.6000000238418579`
- `pred_skewness = 0.5911792516708374` vs `true_skewness = 28.07158660888672`
- `error_minus_correct_rt = NaN`
- `human_error_minus_correct_rt = -0.06515419483184814`
- `model_congruency_rt_gap = 0.11382618546485901` vs `human_congruency_rt_gap = 0.027750015258789062`
- `response_agreement = 0.84033203125`

So the selected WW solution is still too fast, too weakly right-skewed, too large in congruency gap, and still does not instantiate a usable error-vs-correct RT relationship.

### 6. The nearby frontier is visible, but still not compelling

From `WW_baseline_behavior_balanced/checkpoint_ranking_summary.json`:

- selected winner: `scale=0.5, epoch=5`, `behavior_optimal_score = 0.5794`, `error_minus_correct_rt = NaN`
- nearby alternative at `scale=0.1, epoch=5`: `behavior_optimal_score = 0.5735`, `error_minus_correct_rt = 0.1572`, `pred_mean = 0.7128`, `pred_median = 0.7200`

From `WW_error_ordering_behavior_balanced/checkpoint_ranking_summary.json`:

- selected winner still: `scale=0.5, epoch=5`, `behavior_optimal_score = 0.5794`, `error_minus_correct_rt = NaN`
- nearby alternative at `scale=0.1, epoch=5` shifts to `error_minus_correct_rt = 0.1152`, but also has `pred_mean = 0.8485`, `pred_median = 0.9800`, and remains non-winning

So the frontier is visible and real, but the nearby non-winning checkpoints still look like different tradeoffs, not hidden practically useful WW candidates.

### 7. Prior handoff reasoning already pointed toward a deeper bottleneck

From `rt_readout_smoke_decision_memo.md`:

- `simple RT readout changes are not fixing the qualitative structure of the RT distribution`
- `most likely a mismatch between accumulation geometry and RT decoding`

From `NEXT_STEP_DECISION_MEMO.md`:

- `pure RT-shape-only loss tuning` was listed under `Do not prioritize right now`
- `noise_ampa` was kept only as `mechanism probe, but not as current optimization path`
- accumulator-RNN v1/v2 were `abandon` / `pause`

From `rt_readout_redesign_options.md`:

- `the issue is not only the presence or absence of accumulation, but how RT is extracted from the accumulation trajectories`

## Interpretation

### 1. What has now been ruled out?

The accumulated saved evidence now rules out the following as **mainline** next moves:

- `soft_hazard`
- `urgency`
- selector-only retuning under behavior-balanced smoke
- the first lightweight WW objective tweak (`error_ordering`)

These are no longer speculative failures. They are all backed by saved smoke artifacts with `Worth scaling up: False` or equivalent unchanged-winner outcomes.

### 2. What is the most likely remaining bottleneck?

The remaining bottleneck now looks increasingly **structural**, not merely objective-level.

Earlier, it was reasonable to argue that the selector had been too weak or that the objective still was not rewarding the right behavior strongly enough. That case is much weaker now because:

- the behavior-balanced subset is informative,
- the ranking frontier is visible,
- the selector still does not move the winner,
- and even a direct WW-side `error_ordering` tweak leaves the selected checkpoint unchanged and leaves `error_minus_correct_rt` undefined at that selected checkpoint.

That pattern points away from “one more tiny scalar reward term should fix this” and toward a deeper mismatch between the current WW decision-time/readout geometry and the human RT structure we want.

### 3. Why Path A2 is no longer the best mainline move

Path A2 would require believing that one final tiny WW-side tweak is still likely to change the winner in a useful direction.

The saved outputs do not support that belief anymore.

- The selector can now see tradeoffs.
- `error_ordering` did change some lower-ranked frontier values.
- But the selected checkpoint still stayed at `scale=0.5, epoch=5` and remained behaviorally wrong in the same ways.

So another tiny tweak would now look more like continued local searching than the best evidence-based next direction.

### 4. Why Path B is not the best next mainline move

Path B would make sense if the current main problem were that useful candidates are hidden by single-winner export.

But the key frontier is already exposed in the ranking summaries, and the visible alternatives are still unattractive:

- defined error-ordering alternatives have the wrong sign and implausibly slow RTs,
- mean/median-improved alternatives still lack usable error-ordering structure,
- and none becomes the winner even after `error_ordering`.

So top-K export would mostly repackage evidence that is already decision-sufficient.

## Recommendation

### Best-supported next path

**Path C — Begin transition out of WW mainline**

This does **not** mean deleting WW or claiming it was useless. It means the current WW line should stop consuming the next mainline iteration through tiny loss or selector tweaks, because the saved evidence now supports a transition to a new prototype line more strongly than another local WW adjustment.

### Prototype direction

The best-supported prototype direction is:

**structured accumulator-RNN + differentiable decision readout (`DiffDecision`-style)**

### Why this prototype fits the evidence

It preserves the research principles that still matter:

- **N-choice generality**: maintain one accumulator channel per response option
- **explicit accumulation**: latent state evolves over time rather than collapsing to a static classifier head
- **explicit competition**: mutual inhibition or normalized competition across choice channels
- **explicit stochasticity**: noise remains part of the mechanism, not just post hoc output jitter
- **interpretable trajectories**: save and analyze per-choice trajectories directly
- **compatibility with human RT distribution analysis**: decision-time extraction can be made differentiable and distribution-aware rather than relying on the current hard/rigid winner geometry

This prototype is justified by the current evidence because the main unsolved issue is not just getting a slightly different checkpoint score. It is producing the right qualitative RT structure at all.

## Ranked conclusion

### Best-supported next path

1. **Path C — begin transition out of WW mainline**
   - because readout-only changes failed, selector-only changes failed, the eval subset is now informative, and one lightweight WW objective tweak also failed without moving the selected solution

### Rejected alternatives

2. **Path A2 — one final tiny WW objective tweak**
   - rejected because the evidence no longer supports another small WW-side term as the best use of the next iteration

3. **Path B — top-K WW checkpoint export and comparison**
   - rejected because the ranking frontier is already visible, and the nearby candidates look different rather than meaningfully better

## Minimal execution plan for Path C

1. **Freeze WW baseline as the comparison anchor.**
   - treat `WW_baseline_behavior_balanced` with selected `scale=0.5, epoch=5` as the final WW smoke reference

2. **Do not code a full replacement yet.**
   - next step is prototype planning only, not implementation

3. **Define a minimal prototype interface.**
   - input: existing matched cached logits
   - latent state: one accumulator per choice with explicit competition and stochasticity
   - decision head: differentiable first-passage / soft decision-time readout
   - outputs: choice, RT, and saved trajectories for analysis

4. **Keep evaluation exactly compatible with current smoke analysis.**
   - matched `20-29`
   - behavior-balanced smoke subset
   - same behavior metrics: mean RT, skew/tail spread, error-minus-correct RT, congruency gap, response agreement

5. **Use the first prototype only to answer one question.**
   - can a mechanism with explicit accumulation plus a more flexible differentiable decision readout generate a qualitatively more human-like RT distribution than the current WW line?

## Final one-paragraph decision

The accumulated saved evidence now supports **Path C**. Readout-only changes failed, selector-only changes failed, the behavior-balanced smoke subset is now informative, and the first lightweight WW objective tweak also failed to move the selected checkpoint or define `error_minus_correct_rt` at the selected solution. Because the nearby frontier is already visible and still not compelling, the remaining bottleneck now looks increasingly structural rather than just scoring-related. The best next mainline move is therefore to begin transition planning out of the current WW line and define a minimal structured accumulator prototype with a differentiable decision-time readout, while keeping the current WW baseline as the reference point.
