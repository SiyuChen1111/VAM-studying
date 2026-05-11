# Handoff document index

**Updated:** 2026-05-07

This index explains how to read the `handoff/` folder without confusing:

- **current repo-wide status**,
- **authoritative branch-screening verdicts**, and
- **older phase-local recommendation memos**.

## Reading order

If you want the fastest reliable picture of the repository's current state, use this order:

1. `CURRENT_STATUS.md`
   - best repo-wide restart note inside `handoff/`
2. `../../rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md`
   - authoritative HSFA repair-audit outcome
3. `../../rt_model_next_step/06_synthesis/final_successor_branch_memo.md`
   - authoritative successor-screening outcome
4. `single_subject_rt_response_research_judgment_memo.md`
   - current single-subject WW clean/noise and aligned WW-vs-AccumRNN judgment
5. `../../rt_model_semisup_spea_v1_1_calibration/00_protocol/spea_v1_1_calibration_protocol.md`
   - active bounded SPEA calibration follow-up protocol; use when the task is about the live SPEA v1.1 branch
6. `../../rt_model_dmc_var_ww/summary_smoke.md`
   - retained Phase 18 branch-local summary for the first negative-ΔRT DMC+Var→WW result; points to the kept figure `rt_model_breakdown.png`
7. `supervisor_update_2026-04-08.md`
   - dated historical supervisor summary before the later successor-screening bundle
8. `error_regime_experiment_chain_memo.md`
   - detailed technical record of the WW error-regime branch

## Document roles

### Current repo-wide status

- `CURRENT_STATUS.md`
  - the main operational restart note for the current repo state
  - should be treated as the top `handoff/` entry point

### Authoritative branch-level verdict

- `../../rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md`
  - final outcome of the bounded HSFA repair program
  - authoritative for the question: *did fixing the known HSFA Stage-2 issues rescue that branch enough to justify promotion?*
  - current answer: no; final synthesis token is `HSFA_V3_1_KILL_AND_START_STAGE1_UNCERTAINTY_PLAN`

- `../../rt_model_next_step/06_synthesis/final_successor_branch_memo.md`
  - final outcome of the bounded successor-screening program
  - authoritative for the question: *which tested next-line branch, if any, was promoted?*
  - current answer: `NO_SUCCESSOR_BRANCH_CLEARED_GATES`

### Current topical judgment docs

- `single_subject_rt_response_research_judgment_memo.md`
  - still current for the verified WW clean/noise and aligned WW-vs-AccumRNN single-subject evidence
  - not the authoritative repo-wide successor-screening verdict

- `../../rt_model_dmc_var_ww/summary_smoke.md`
  - current retained Phase 18 branch-local summary
  - use when the task is specifically about the DMC+Var→WW negative-ΔRT breakthrough or the kept figure `rt_model_breakdown.png`
  - not a repo-wide promotion verdict by itself

### Current active protocol docs

- `../../rt_model_semisup_spea_v1_1_calibration/00_protocol/spea_v1_1_calibration_protocol.md`
  - current protocol lock for the bounded SPEA v1.1 calibration follow-up
  - authoritative for the question: *what is the live scoped goal of the current SPEA calibration branch?*
  - not yet a scientific result memo by itself

### Historical but still useful summaries

- `supervisor_update_2026-04-08.md`
- `NEXT_STEP_DECISION_MEMO.md`
- `post_error_ordering_decision_memo.md`
- `final_post_prototype_decision_memo.md`
- `ww_behavior_balanced_decision_memo.md`
- `rt_readout_smoke_decision_memo.md`
- `rt_readout_redesign_options.md`

These files remain useful as dated summaries of specific phases, but their “next step” or “recommendation” language should be read as **historical context**, not as the current repo-wide decision.

### Historical technical evidence memos

- `error_regime_experiment_chain_memo.md`
- `logs.md`

These are important for reconstructing what was tried and why it failed or partially succeeded. They are evidence-rich and still valuable, but they are not by themselves the top current-state entry point.

### Operational or maintenance notes

- `archive_cleanup_plan.md`
- `OFFLINE_AUTORUN_NOTE.md`
- `agent_file_structure_cleanup_prompt.md`
- `ai_prompt_error_regime_experiment.md`

These are helper or maintenance notes, not scientific current-state summaries.

## Practical rule

When two files disagree about “what to do next,” use this precedence:

1. `CURRENT_STATUS.md`
2. `../../rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md`
3. `../../rt_model_next_step/06_synthesis/final_successor_branch_memo.md`
4. current topical judgment memos
5. older dated recommendation memos

In practice, this means that older recommendations to keep probing WW, repair HSFA locally, or transition into a successor line are superseded when they conflict with the later saved verdicts `HSFA_V3_1_KILL_AND_START_STAGE1_UNCERTAINTY_PLAN` and `NO_SUCCESSOR_BRANCH_CLEARED_GATES`.
