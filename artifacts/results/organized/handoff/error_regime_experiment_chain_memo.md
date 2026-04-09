# Error-Regime Experiment Chain Memo

## Purpose

This memo summarizes the full experimental chain aimed at pushing the current WW pipeline into a **human-like error regime** while preserving useful RT-distribution structure.

The immediate scientific goal was not perfect human skew matching. The nearer-term goal was:

> make the model produce nontrivial errors, so that error-conditioned RT structure becomes measurable and comparable to human behavior.

---

## Baseline problem

The baseline behavior-balanced WW smoke run:

- already showed RT variability
- already showed mild right-skew
- already showed a congruent vs incongruent RT effect

but it still had a critical limitation:

- **model error count = 0**
- therefore `error_minus_correct_rt = NaN`

So the main bottleneck became:

> the model was not yet entering a useful error regime.

---

## Evaluation context

All experiments in this chain used the same controlled smoke setting unless otherwise noted:

- matched `20-29`
- cached logits
- behavior-balanced eval subset
- `response_labels` supervision maintained
- WW architecture and baseline readout preserved

This means the comparisons below are all within the same general experimental regime.

---

## Experiment chain

### 1. Shape-only RT distribution loss: `soft_hist_kl`

Run:

- `checkpoints_age_groups_rtshape_kl/20-29/smoke/WW_rt_dist_soft_hist_kl_behavior_balanced/`

What changed:

- added a soft-histogram KL divergence RT-distribution loss

What happened:

- selected checkpoint changed from baseline (`scale=0.5`) to `scale=0.4`
- but skew got worse relative to baseline
- congruency gap moved in the wrong direction
- `error_minus_correct_rt` remained undefined

Conclusion:

> shape supervision had signal, but this first formulation did not improve the candidate enough to beat baseline.

---

### 2. Shape-only RT distribution loss: `cdf_wasserstein`

Run:

- `checkpoints_age_groups_rtshape_cdf/20-29/smoke/WW_rt_dist_cdf_wasserstein_behavior_balanced/`

What changed:

- replaced KL-on-bins with a 1D empirical CDF / Wasserstein-style RT-distribution loss

What happened:

- selected checkpoint changed to `scale=0.3`
- distribution-shape supervision still clearly influenced optimization
- but skew still did not improve over baseline
- congruency gap still moved away from human
- `error_minus_correct_rt` remained undefined

Conclusion:

> a better shape loss changed the winner, but shape-only supervision was still insufficient to create a useful error regime.

---

### 3. Conditional shape loss: congruency-conditioned CDF matching

Run:

- `checkpoints_age_groups_rtshape_conditional/20-29/smoke/WW_rt_dist_conditional_behavior_balanced/`

What changed:

- matched RT distributions separately for congruent and incongruent subsets

What happened:

- selected checkpoint again changed to `scale=0.3`
- this was more behaviorally aligned than global shape-only fitting in design
- but practical outcome was still the same pattern:
  - no model errors
  - `error_minus_correct_rt = NaN`
  - no clear gain over baseline

Conclusion:

> condition-specific distribution fitting still could not unlock an error regime by itself.

---

### 4. Joint intervention: `cdf_wasserstein + fixed_noise_ampa = 0.04`

Run:

- `checkpoints_age_groups_rtshape_noise_joint/20-29/smoke/WW_rt_dist_cdf_wasserstein_noise_probe_behavior_balanced/`

What changed:

- retained CDF-shape loss
- added a small fixed WW internal noise probe (`fixed_noise_ampa = 0.04`)

What happened:

- shape improved modestly:
  - higher skew than baseline
  - slightly better congruency gap
- RT scale stayed plausible relative to baseline
- response agreement did not collapse
- but **model still had 0 errors**
- `error_minus_correct_rt` remained undefined

Conclusion:

> this was the first joint setup that looked directionally promising on shape, but it still did not create a usable error regime.

---

### 5. Joint intervention: `cdf_wasserstein + fixed_noise_ampa = 0.06`

Run:

- `checkpoints_age_groups_rtshape_noise_joint_hi/20-29/smoke/WW_rt_dist_cdf_wasserstein_noise_probe_behavior_balanced/`

What changed:

- same shape-loss setup as above
- slightly stronger fixed internal noise (`fixed_noise_ampa = 0.06`)

What happened:

- **this was the first run that produced nonzero model errors**
- `error_minus_correct_rt` became defined
- selected checkpoint remained at `epoch=5` but moved to `scale=0.3`
- skew improved strongly over baseline
- congruency gap moved somewhat toward human
- response agreement remained acceptable (`0.8303` vs baseline `0.8403`)

But key failures remained:

- RT mean was still too fast relative to human (`0.4171` vs human `0.6137`)
- tail spread overshot human strongly (`0.5625` vs human `0.1442`)
- error-vs-correct direction was opposite to the current human data:
  - human: negative
  - model: strongly positive

Conclusion:

> this was the **first informative error-regime candidate**, but it was still directionally wrong as a human-like solution.

---

### 6. Direction correction attempt: `cdf_wasserstein + noise=0.06 + threshold correction`

Run:

- `checkpoints_age_groups_rtshape_noise_threshold_joint/20-29/smoke/WW_rt_dist_cdf_wasserstein_noise_threshold_probe_behavior_balanced/`

What changed:

- kept the `noise=0.06` error-regime candidate
- added a small fixed threshold correction (`fixed_threshold = 0.45`)

What happened:

- error regime was preserved
- `error_minus_correct_rt` stayed defined
- congruency gap moved slightly closer to human

But:

- RT scale became even faster than before
- mean and median moved farther from human
- error-vs-correct direction remained wrong

Conclusion:

> threshold correction preserved the gain (nonzero errors) but pushed RT scale in the wrong direction.

---

### 7. Direction correction attempt: `cdf_wasserstein + noise=0.06 + weak mean/median anchor`

Run:

- `checkpoints_age_groups_rtshape_noise_anchor_joint/20-29/smoke/WW_rt_dist_cdf_wasserstein_noise_anchor_behavior_balanced/`

What changed:

- kept the `noise=0.06` error-regime candidate
- added a weak mean/median RT anchor

What happened:

- error regime was preserved
- `error_minus_correct_rt` stayed defined
- response agreement remained acceptable (`0.8354`)
- congruency gap was slightly better than the `noise=0.06` candidate

But:

- mean and median still moved farther from human rather than closer
- tail remained too heavy
- error-vs-correct direction remained wrong and even somewhat exaggerated

Conclusion:

> weak moment anchoring did not successfully correct the direction of the error-regime candidate.

---

## Best result in this chain

The most informative candidate from this entire chain is:

- `WW_rt_dist_cdf_wasserstein_noise_probe_behavior_balanced` with `fixed_noise_ampa = 0.06`

Why it matters:

- first candidate with nonzero model errors
- first candidate where `error_minus_correct_rt` became defined
- response agreement remained acceptable
- trajectory diagnostics remained interpretable

Why it still does **not** qualify as a final solution:

- RT scale remained too fast
- tail spread overshot human strongly
- error-vs-correct direction remained opposite to current human data

So this candidate should be treated as:

> **the first informative error-regime candidate, but not yet a human-like solution**.

---

## What has now been established

This chain shows a clear progression:

1. **Shape-only losses**
   - can change the selected checkpoint
   - but cannot create model errors by themselves

2. **Shape + modest noise**
   - can improve some shape metrics
   - but still may fail to create model errors

3. **Shape + stronger noise**
   - can create an error regime
   - but may create the wrong kind of error regime

4. **Small corrective tweaks after entering an error regime**
   - did not successfully restore human-like RT scale or error direction

---

## Final interpretation

The main local question has now been answered:

> Yes, the current WW pipeline can be pushed into an error regime.

But the harder question is now clearer too:

> The problem is not just “how to make the model commit errors.”
> The problem is “how to make the model commit errors in a human-like RT regime.”

That distinction matters.

This means the research bottleneck has shifted from:

- **error absence**

to:

- **error-regime directionality and calibration**

---

## Recommendation

At this point, further tiny WW-side corrections are likely to have diminishing returns.

The current evidence supports this summary:

- the error-regime question has been opened successfully
- but the resulting regimes are still not behaviorally aligned with human data
- the most useful candidate is scientifically informative, but not deployment-worthy as the “best model”

So the best next move is:

> consolidate this chain as a formal intermediate finding,
> and use it to motivate the next modeling or objective-design step rather than continuing many more local WW patches.

---

## One-sentence summary

We now know that WW can be pushed from a no-error regime into a measurable error regime using joint RT-distribution shaping plus stronger internal noise, but the resulting solutions remain too fast and directionally wrong relative to human error-conditioned RT behavior, and subsequent small corrections did not fix that mismatch.
