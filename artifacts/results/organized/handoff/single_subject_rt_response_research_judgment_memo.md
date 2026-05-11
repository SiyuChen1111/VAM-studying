# Single-Subject RT/Response Research Judgment Memo

> **Scope note (2026-05-02):** This memo is still current for the verified WW clean/noise and aligned WW-vs-AccumRNN single-subject evidence it summarizes. It is **not** the authoritative repo-wide successor-screening verdict. For the later branch-level outcome of the bounded `rt_model_next_step` program, use `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`.

## Purpose

This memo consolidates the current verified single-subject findings for the active VGG-based decision-dynamics workflow. It focuses on three linked result bundles:

- the **WW clean RT+response-only** bounded single-subject workflow
- the **WW fixed-noise probe** bounded single-subject workflow (`fixed_noise_ampa = 0.05`)
- the **aligned WW vs AccumRNN single-subject export comparison**

The goal is not to claim that any current model already matches human behavior well. The goal is to establish what the current verified results do and do not support, and to identify the most justified next experiments.

## Evidence base

This memo is grounded in the following verified artifacts:

- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only_noise05/`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_clean_vs_noise_comparison/20_29_clean_vs_noise05.csv`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_clean_vs_noise_comparison/20_29_clean_vs_noise05_interpretation.md`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_clean_vs_noise_comparison/80_89_clean_vs_noise05.csv`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_clean_vs_noise_comparison/80_89_clean_vs_noise05_interpretation.md`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_accumrnn_aligned/`
- `artifacts/results/repro_legacy_interim/single_subject_model_export_comparison_rt_response_only_aligned/single_subject_export_metrics_comparison.csv`
- `artifacts/results/repro_legacy_interim/single_subject_model_export_comparison_rt_response_only_aligned/single_subject_export_summary.md`

## Main finding

The strongest verified result is that **WW can now be pushed into an errorful single-subject regime** by increasing internal noise, but that errorful regime is still **not human-like**. At the same time, the aligned single-subject comparison does **not** yet support replacing WW with the current AccumRNN implementation. Instead, the current evidence supports a split judgment:

- **WW** is the stronger behaviorally coherent baseline in the current verified comparisons.
- **AccumRNN** appears to have more capacity for large positive skew / heavy-tail structure, but it is not yet sufficiently calibrated in RT scale or response fidelity.

## WW-specific interpretation

### Clean RT+response-only WW path

The verified clean WW bundle shows that a stripped objective can preserve very high response fidelity while keeping RTs in a more plausible range within the current verified single-subject workflow. However, that cleaner objective alone does **not** solve the main problem. In the clean root, the model often remains too accurate, too concentrated, or too light-tailed, and in several subjects it still fails to produce a usable error regime at all.

So the clean WW result should be read as:

> the objective is now simple enough to interpret, but the baseline dynamics still do not naturally generate the full human error-conditioned RT structure.

### Noise probe on the same WW path

The fixed-noise probe clarifies something important. Raising `noise_ampa` to `0.05` can push the model out of the degenerate no-error regime.

For the verified `20-29` panel:

- `noise05` induces model errors for all three subjects where the clean version produced none.
- Accuracy drops only slightly in those cases.
- RT-shape and total score improve modestly.

For the verified `80-89` panel:

- `noise05` changes the model more unevenly.
- It can induce or amplify model errors, but often with a larger cost in response fidelity.
- It does not produce a clean qualitative improvement across the panel.

The crucial limitation is that the induced model error regime still has the **wrong direction** relative to the human data. In both age panels, the model’s error-minus-correct RT tends to become **positive** (slower model errors), while the human error-minus-correct RT remains **negative** (faster human errors) in the subjects highlighted by the interpretation files.

So the supported statement is narrow but important:

> noise is a useful **mechanism probe** for WW regime sensitivity, but not yet a successful behavioral fix.

## AccumRNN-specific interpretation

The aligned WW vs AccumRNN export comparison suggests that AccumRNN is not simply worse in every respect. Instead, it fails in a different way.

In the aligned comparison bundle:

- AccumRNN often produces **much larger positive skewness** than WW.
- In a few subjects, that makes its RT-shape statistics look closer to the human subject than WW’s shape statistics do.
- But this often happens only by collapsing the RT center to implausibly small values or by degrading response fidelity substantially.

That means the current AccumRNN implementation looks less like a calibrated behavioral model and more like a mechanism with **tail-generating capacity but poor control of timing scale**.

The safest summary is:

> AccumRNN currently demonstrates potential for heavy-tailed RT structure, but not yet a convincing single-subject fit.

## Subject-by-subject pattern summary

The aligned export comparison suggests three recurring subject-level patterns.

### Pattern A: WW is clearly more trustworthy overall

Subjects like `20-29 / 3875`, `80-89 / 3403`, and `80-89 / 984` fall into this bucket.

- WW keeps RT center and response accuracy in a more plausible regime.
- AccumRNN may show larger skew, but only by producing unrealistically tiny medians or degraded accuracy.

### Pattern B: no clean winner, each model gets a different piece right

Subjects like `20-29 / 5675` fit this pattern.

- WW can keep the RT center near the human range but miss the skew badly.
- AccumRNN can preserve some behavior metrics while collapsing RT scale.

### Pattern C: AccumRNN reveals genuine shape potential

Subjects like `20-29 / 677` and `80-89 / 6609` are the strongest versions of this pattern.

- WW stays more stable on accuracy.
- AccumRNN can generate much stronger skew or a broader tail.
- But the cost in RT scale or response fidelity still prevents it from being a superior fit overall.

So the most defensible subject-level interpretation is not that one model wins everywhere. It is that:

> WW is the stronger current baseline, while AccumRNN remains an unstable mechanism candidate with some apparent tail-generating capacity.

## Research judgment

Taken together, the current verified artifacts support the following judgment.

1. **Keep WW as the main interpretable baseline.** It is currently the only line that reliably preserves response behavior and a roughly plausible RT scale across subjects.
2. **Do not claim that WW now explains human error-conditioned RT structure.** The new noise probe only shows that the WW system can be pushed into an errorful regime, not that it gets the human regime right.
3. **Do not replace WW with the current AccumRNN implementation.** The aligned export comparison does not support that move yet.
4. **Do keep AccumRNN alive as a structural candidate.** Its skew/tail behavior suggests it may be expressing a mechanism that is not yet being translated into a good overall behavioral fit.

## Recommended next steps

### For WW

- Continue to treat `noise_ampa` as a controlled probe, not a final answer.
- Test whether the errorful WW regime can be preserved while correcting the error RT direction.
- Prefer **small constrained probes** (noise plus one additional limited mechanism change) over broad retuning.

### For AccumRNN

- Prioritize **stabilization of RT scale** before chasing additional tail structure.
- Ask whether the mechanism can retain its stronger positive skew while moving RT mean and median into the human range.
- Only after that should model-vs-model judgment focus on shape or error structure.

### For comparisons

- Keep all future WW vs AccumRNN work aligned to the same subject panel and the same persisted fit/test subsets.
- Continue using the aligned single-subject export bundle as the main comparison surface, because it cleanly exposes RT center, skew, and response accuracy together.

## Final conclusion

The verified result is not that a model has solved the single-subject problem. The verified result is that:

> **WW noise can create an errorful regime, but that regime is still not human-like, and the aligned single-subject comparison does not yet show a stronger replacement model.**

That is enough to guide the next research step. It is not enough to claim success.
