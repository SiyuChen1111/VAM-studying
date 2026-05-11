# Final Post-Prototype Decision Memo

> **Historical status note (2026-05-02):** This memo records the decision state after the first structured post-WW prototype stage. Its recommendation language is still useful as phase-local history, but it predates the later bounded successor-screening bundle under `artifacts/results/rt_model_next_step/`. For the current branch-level outcome, use `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`.

## Scope

This memo is the final post-prototype decision point after:

- WW readout-only patching,
- selector / behavior-balanced reruns,
- the lightweight WW `error_ordering` tweak,
- and the first structured accumulator-RNN smoke prototype.

It separates:

- **Observed evidence** — concrete values quoted from saved outputs
- **Interpretation** — what has now been ruled out and what bottleneck remains most plausible
- **Recommendation** — exactly one next path

## File status check

Confirmed present and used here:

- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_soft_hazard/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_baseline_vs_urgency/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_baseline_vs_WW_error_ordering_behavior_balanced/summary_smoke.md`
- `checkpoints_age_groups_rtreadout/20-29/smoke/comparison_WW_vs_accumulator_rnn_behavior_balanced/summary_smoke.md`
- `results/organized/handoff/rt_readout_smoke_decision_memo.md`
- `results/organized/handoff/ww_behavior_balanced_decision_memo.md`
- `results/organized/handoff/post_error_ordering_decision_memo.md`
- `results/organized/handoff/NEXT_STEP_DECISION_MEMO.md`
- `results/organized/handoff/rt_readout_redesign_options.md`

## Observed evidence

### 1. WW patching failed

#### soft_hazard

From `comparison_baseline_vs_soft_hazard/summary_smoke.md`:

- `Predicted skewness: baseline=-0.1843, soft_hazard=-1.2633, human=75.5031`
- `Tail spread q95-q50: baseline=0.2800, soft_hazard=0.0036, human=0.1490`
- `Error minus correct RT: baseline=0.2801, soft_hazard=0.0009, human=-0.0651`
- `Congruency gap: baseline=0.0944, soft_hazard=0.0009, human=0.0261`
- `Response agreement: baseline=0.9401, soft_hazard=0.9318`
- `Worth scaling up: False`

This was not a rescue; it collapsed RT structure.

#### urgency

From `comparison_baseline_vs_urgency/summary_smoke.md`:

- `Mean RT: baseline=0.7124, Urgency=0.7175, human=0.6256`
- `Median RT: baseline=0.7100, Urgency=0.7400, human=0.6010`
- `Predicted skewness: baseline=-0.1843, Urgency=-0.6799, human=75.5031`
- `Tail spread q95-q50: baseline=0.2800, Urgency=0.2200, human=0.1490`
- `Error minus correct RT: baseline=0.2801, Urgency=-0.4414, human=-0.0651`
- `Congruency gap: baseline=0.0944, Urgency=0.0592, human=0.0261`
- `Response agreement: baseline=0.9401, Urgency=0.9423`
- `RT-scale gate passed: True`
- `Worth scaling up: False`
- `Conclusion: reject B`

This preserved RT scale better than soft_hazard, but made skew and error-vs-correct behavior worse, so it also failed.

### 2. Selector became informative, but did not change the WW winner

From `ww_behavior_balanced_decision_memo.md` and `comparison_WW_baseline_vs_checkpoint_tail_focus_behavior_balanced/summary_smoke.md`:

- behavior-balanced subset diagnostics: `selected_eval_trials = 4096`, `actual_human_error_trials = 654`, `congruent_trials = 1892`, `incongruent_trials = 2204`, `balance_constraints_satisfied = 1`
- `Ranking tradeoff visible (baseline): True`
- `Ranking tradeoff visible (WW_checkpoint_tail_focus): True`
- `Ranking tradeoff visible overall: True`
- `Baseline checkpoint: scale=0.5000, epoch=5`
- `WW_checkpoint_tail_focus checkpoint: scale=0.5000, epoch=5`
- `Baseline and WW_checkpoint_tail_focus selected the same checkpoint: True`
- `Baseline error_minus_correct_rt defined: False`
- `WW_checkpoint_tail_focus error_minus_correct_rt defined: False`
- `Conclusion: reject and stay baseline`

So the selector problem was made visible, but selector improvement alone did not move the chosen WW solution.

### 3. `error_ordering` failed

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

This is the strongest WW-side null result: a direct objective tweak still left the selected solution unchanged.

### 4. The structured accumulator-RNN prototype had interpretable trajectories but failed RT scale and response agreement gates

From `comparison_WW_vs_accumulator_rnn_behavior_balanced/summary_smoke.md`:

- `Mean RT: baseline=0.4496, AccumulatorRNN=0.2763, human=0.6137`
- `Median RT: baseline=0.4000, AccumulatorRNN=0.2100, human=0.6000`
- `Predicted skewness: baseline=0.5912, AccumulatorRNN=1.4190, human=28.0716`
- `Tail spread q95-q50: baseline=0.4200, AccumulatorRNN=0.5600, human=0.1442`
- `Error minus correct RT: baseline=nan, AccumulatorRNN=-0.6524, human=-0.0652`
- `Congruency gap: baseline=0.1138, AccumulatorRNN=0.0309, human=0.0278`
- `Response agreement: baseline=0.8403, AccumulatorRNN=0.0476`
- `Mean RT stays near baseline scale: False`
- `Median RT stays near baseline scale: False`
- `Lower tail does not collapse earlier: False`
- `RT-scale gate passed: False`
- `Response agreement does not materially collapse: False`
- `AccumulatorRNN error_minus_correct_rt defined: True`
- `Trajectory shape: [4096, 100, 4]`
- `Trajectories interpretable: True`
- `Worth scaling up: False`
- `Conclusion: reject and stay baseline`

So the prototype did reveal useful mechanism-side signals — defined error RT structure, near-human congruency gap, interpretable trajectories — but still failed the decisive behavioral gates.

### 5. What has now been ruled out?

Grounded in the saved evidence chain, the following are now ruled out as the best immediate next move:

- WW readout-only patching (`soft_hazard`, `urgency`)
- selector-only rescue of WW
- one small WW objective tweak as likely rescue (`error_ordering`)
- immediate promotion of the first structured accumulator-RNN prototype to the new main line

## Interpretation

### 1. The key local questions have already been answered

The repository has now directly tested:

- whether WW readout is the main problem,
- whether checkpoint selection blindness is the main problem,
- whether the smoke eval subset was too weak,
- whether one small WW objective tweak could rescue the line,
- and whether the first next-line structured accumulator prototype is already viable.

All of those questions now have grounded answers from saved outputs, and none of them points to an obvious immediate implementation step.

### 2. Most likely remaining bottleneck

The strongest remaining bottleneck is a **structural mismatch in how these current setups couple latent evidence accumulation to observed response choice and RT decoding**.

More concretely:

- WW preserves response agreement far better than the accumulator prototype, but still fails to generate the desired RT structure.
- The accumulator prototype recovers some target statistics, but only by collapsing core behavioral fidelity.

That means improvements in one slice of behavior are currently being purchased by collapse in another. The issue is no longer well described as “one more tweakable scalar loss term” or “just pick a different saved checkpoint.”

### 3. Is the next step still a modeling step, or a framing/consolidation step?

The next step should be a **consolidation step**, not a new modeling step.

Why:

- Another WW tweak now looks like low-value churn because the selector is already informative and the first objective tweak did not change the winner.
- Immediate commitment to a second-generation structured model line is not justified because the first prototype failed on decisive criteria: RT scale and response agreement.
- Reframing may ultimately be needed, but the immediate high-value task is first to consolidate what has already been ruled out and what the bottleneck now looks like.

## Recommendation

### Best-supported next path

**Path E — Pause modeling expansion and consolidate findings.**

This is the single best-supported next direction after the full evidence chain.

### Why Path E is the best next direction

1. **WW patching failed.**
   - soft_hazard collapsed timing structure.
   - urgency preserved scale better but worsened skew and error-vs-correct behavior.

2. **Selector became informative but did not change the WW winner.**
   - The pipeline can now see tradeoffs, but that no longer appears to be the limiting problem.

3. **`error_ordering` failed.**
   - A direct WW objective nudge still left the selected solution unchanged.

4. **The accumulator-RNN prototype is not a viable immediate replacement.**
   - It produced interpretable trajectories and defined error RT structure, but RT scale and response agreement both failed hard.

5. **More coding now is more likely to create churn than knowledge.**
   - The evidence is already rich enough to justify a pause and consolidation memo.

## Rejected alternatives

### Path D — design a more constrained next prototype

Rejected for now because the first accumulator miss no longer looks like a merely incidental training miss. The collapse in response agreement (`0.0476`) and RT scale (`0.2763` mean, `0.2100` median) is too severe to justify immediate prototype iteration without first consolidating what was learned.

### Path F — reframe the problem before new modeling

Potentially relevant later, but not the best immediate next move. There may indeed be a framing mismatch underneath the modeling failures, but the highest-value action right now is to consolidate the evidence chain cleanly before re-opening the scientific target definition.

### Path G — commit to a second-generation structured model line

Rejected because the evidence does not support immediate commitment. The first structured prototype revealed a useful mechanism direction, but it failed too decisively on core behavioral criteria to justify instant escalation.

## Short ranked conclusion

1. **Best-supported next path: Path E**
   - consolidate negative findings and stop immediate modeling churn

2. **Rejected alternative: Path F**
   - plausible later, but secondary to consolidation right now

3. **Rejected alternative: Path D**
   - too early to iterate another prototype immediately after this failure mode

4. **Rejected alternative: Path G**
   - not enough support for immediate commitment to a new model family

## Minimal execution plan for Path E

1. **Freeze the current evidence chain as the local answer to this research phase.**
   - Treat `WW_baseline_behavior_balanced` as the final WW anchor for this phase.
   - Treat `accumulator_rnn_behavior_balanced` as the first next-line prototype result, not a promoted replacement.

2. **Consolidate what has been ruled out.**
   - WW readout-only patching failed.
   - Selector became informative but did not change the WW winner.
   - `error_ordering` failed.
   - The structured accumulator-RNN prototype had interpretable trajectories but failed RT scale and response-agreement gates.

3. **State the most likely bottleneck explicitly.**
   - Current model/readout families cannot yet preserve response fidelity and human-like RT structure at the same time.

4. **Prepare the next research question before more implementation.**
   - The next question should be scoped more carefully than “try another tweak.”
   - It should start from the consolidated evidence, not from another immediate code branch.

5. **Do not run more experiments until that next question is written down.**
   - This is the main anti-churn guardrail.

## Final one-paragraph decision

The best final next step is **Path E — pause modeling expansion and consolidate findings**. WW patching failed, the selector became informative but still did not change the WW winner, `error_ordering` failed, and the structured accumulator-RNN prototype produced interpretable trajectories but still failed the decisive RT-scale and response-agreement gates. At this point the key local question has already been answered: neither small WW-side rescue nor the first next-line prototype is the justified immediate path forward. The highest-value next move is therefore to consolidate the negative findings, state the most likely remaining bottleneck clearly, and define a better-scoped next research question before writing more code.
