# Research Plan: Age-Related Differences in Noise-Driven Decision Dynamics

## 1. Positioning and Scope

This document is the **execution plan** for the age-related extension of the broader noise proposal.

The goal is not only to fit response times, but to test whether a Wong-Wang-style noise-driven decision model can:

1. reproduce core **human signatures** of decision behavior in the LIM/Flanker task,
2. mimic the behavioral differences between **young (20-29)** and **old (80-89)** groups, and
3. support a constrained, mechanism-level interpretation of those age differences.

This plan is intentionally narrower than `research_proposal_v4.md`:

- **Task focus**: LIM / Flanker only
- **Age comparison**: `20-29` vs `80-89` only
- **Analysis level**: group-level only in the current phase
- **Main model path**: shared Stage 1, age-specific Stage 2
- **Main explanatory target**: internal noise / internal fluctuation in decision dynamics

## 2. Research Question

**Can a Wong-Wang-style decision model reproduce human decision signatures and explain age-related differences in LIM behavior through differences in internal decision dynamics, especially noise-related dynamics?**

More specifically, this project asks:

1. Can the model reproduce human signatures at the group level for each age group?
2. Do young and old groups require different fitted Stage-2 parameters to match behavior?
3. Are age differences better explained by changes in internal fluctuation (`noise_ampa`) than by threshold-related variability or other effective parameters?
4. Do the fitted parameters induce different trajectory geometries under congruent and incongruent conditions when evaluated on matched stimulus trials?

## 3. Methodological Stance

The project adopts the following stance:

- **Human signatures are necessary but not sufficient.**
  Reproducing signatures such as right-skewed RT distributions, congruency effects, and error-related RT asymmetries is a baseline requirement, not by itself a mechanism claim.
- **Age claims are constrained.**
  Age effects are interpreted as differences in fitted **effective decision-dynamics parameters**, not as direct measurements of neural physiology.
- **Noise remains the primary mechanism of interest.**
  `noise_ampa` is the primary age-linked candidate parameter. Threshold-related variability is treated as a secondary, competing explanation.
- **Mechanism claims require trajectory analysis.**
  A mechanism claim must be supported by the geometry and variability of Stage-2 trajectories under matched stimulus conditions, not only by parameter differences.

## 4. Hypotheses

### 4.1 Behavioral Signatures

- **H1a**: Older adults have longer RTs than younger adults.
- **H1b**: Older adults show higher RT variability.
- **H1c**: Both groups exhibit human-like RT signatures, including right-skewed RT distributions.
- **H1d**: Both groups exhibit a congruency effect, with incongruent trials producing slower RTs and/or worse performance than congruent trials.
- **H1e**: If the model is adequate, it should reproduce these signatures for each age group.

### 4.2 Parameter-Level Hypotheses

- **H2a (primary)**: Older adults will require higher fitted `noise_ampa`, consistent with greater internal fluctuation requirements in the decision module.
- **H2b (secondary)**: Threshold-related variability will differ across age groups, consistent with changes in decision caution / criterion setting.
- **H2c (effective coupling)**: Other effective parameters such as `J_ext`, `I_0`, and `J_matrix` may also differ between groups, but these are interpreted as effective decision-dynamics parameters rather than direct biological measurements.

### 4.3 Mechanism-Level Hypotheses

- **H3a**: Age differences in behavior will be reflected in different Stage-2 trajectory geometries.
- **H3b**: Under matched stimulus conditions, the old group will show altered trajectory spread, state-space occupancy, or threshold-crossing dynamics relative to the young group.
- **H3c**: Congruent and incongruent trials will separate differently across age groups in Stage-2 state space.

## 5. Main Modeling Strategy

### 5.1 Main Path

The main path for this project is:

1. **Shared Stage 1**: Use one common Stage-1 visual model to generate logits.
2. **Age-Specific Stage 2**: Fit a separate group-level Stage-2 Wong-Wang module for `20-29` and `80-89`.
3. **Behavior -> Parameters -> Mechanism**: Evaluate the model in that order.

This design isolates age differences primarily at the decision-dynamics level.

### 5.2 Control Branches (Not Main Path)

These are not part of the default execution path, but can be opened later if needed:

- **Control Branch 1**: Compare Stage-1 logit statistics across age groups.
- **Control Branch 2**: Train age-specific Stage-1 models.

These controls are future branches, not default requirements for the current phase.

## 6. Analysis Flow

The required analysis order is:

1. **Behavior / human signatures**
2. **Parameter comparison**
3. **Mechanism analysis**

This ordering is mandatory because behavior-fitting adequacy must be established before parameter or mechanism claims are made.

## 7. Data and Analysis Units

### 7.1 Age Groups

- Young group: `20-29`
- Old group: `80-89`

### 7.2 Analysis Level

- Current phase: **group-level only**
- Future extension: subject-level modeling if the group-level pipeline is successful

### 7.3 Stimulus Structure for Mechanism Analysis

Mechanism analysis will use **matched stimulus trials** organized by congruency condition.

The comparison will preserve trial/stimulus correspondence as much as possible, relying on the limited number of canonical LIM stimulus categories already present in the dataset. The key comparison is:

- young vs old,
- under **congruent** and **incongruent** conditions,
- on matched stimulus identities or matched stimulus families.

## 8. Stage Definitions

### 8.1 Stage 1: Shared Visual Evidence Extraction

Stage 1 produces the visual evidence representation for both age groups.

Main-path assumptions:

- the same Stage-1 model is used for both age groups,
- logits are extracted separately for each group’s trials,
- Stage 1 is treated as a shared visual front-end,
- age-specific Stage-1 learning is not part of the default pipeline.

Stage-1 outputs to preserve:

- train logits
- test logits
- trial metadata needed for congruent/incongruent matching
- basic logit summary statistics

### 8.2 Stage 2: Joint Decision-Dynamics Modeling

Stage 2 is **not RT-only** in its scientific role.

The Stage-2 Wong-Wang model must be evaluated against:

- **RT behavior**
- **choice / accuracy behavior**
- **congruency effects**

Even if the optimization target remains RT-centric in implementation, the scientific acceptance criteria are joint.

## 9. Core Human Signatures to Reproduce

The model must reproduce, at minimum, the following group-level signatures:

1. **Right-skewed RT distributions**
2. **Longer RTs for older adults**
3. **Higher RT variability in older adults**
4. **Congruency effect**
5. **Choice / accuracy patterns consistent with human data**
6. **Preferably**: error-related RT asymmetry if the current implementation supports it adequately

These signatures are baseline requirements.

## 10. Main Mechanism Analysis

### 10.1 Main Mechanism Question

Given shared Stage-1 evidence, do the fitted Stage-2 models for the young and old groups produce different decision geometries that explain the behavioral age differences?

### 10.2 Primary Mechanism Analysis

The primary mechanism analysis uses:

- shared Stage-1 logits,
- matched stimulus trials,
- age-specific Stage-2 parameters,
- separate evaluation of **congruent** and **incongruent** conditions.

### 10.3 Primary Outputs

Primary geometry outputs should include:

- trajectory/state-space visualization (APA 7 figure style)
- young vs old comparisons in shared state space
- congruent vs incongruent comparisons within each age group
- trajectory spread / variability summaries
- threshold-crossing summaries

### 10.4 Main Figure Logic

The central mechanism figure should compare:

- **young vs old trajectory geometry**, with
- **congruent vs incongruent** conditions shown explicitly.

Correct/error splits may be included as a secondary exploratory analysis, not as the main result.

## 11. Evaluation Criteria

### 11.1 Quantitative Fit Criteria

The model should be evaluated on both quantitative and qualitative criteria.

Quantitative criteria should include, where appropriate:

- mean RT difference from human data
- median RT difference from human data
- skewness comparison
- RT distribution overlap or distance metric
- congruent vs incongruent RT gap comparison
- choice / accuracy comparison

### 11.2 Ordered Acceptance Criteria

Acceptance should be assessed in the following order:

1. **RT distribution shape**
2. **choice / accuracy adequacy**
3. **congruency effect**
4. **mechanism-level geometry**

### 11.3 APA 7 Reporting Standard

All planned figures and summaries should be prepared using APA 7 style conventions where applicable:

- `Figure 1`, `Figure 2`, `Table 1`, etc.
- report `M`, `SD`, and confidence intervals where relevant
- use consistent condition labeling and axis naming
- avoid overloaded decorative plotting

## 12. Execution Gates

### Gate 1: Stage-1 Artifact Readiness

Required outputs:

- group-specific logits generated successfully
- metadata preserved for congruency-based matching
- no missing required Stage-1 artifacts

### Gate 2: Human Signature Gate

Required condition:

- the Stage-2 model must reproduce the baseline human signatures well enough to justify downstream interpretation

If this gate fails, mechanism claims must stop.

### Gate 3: Joint Behavior Gate

Required condition:

- the model must be acceptable not just for RT summary statistics, but also for choice/accuracy and congruency patterns

### Gate 4: Mechanism Gate

Required condition:

- the geometry analysis must support or appropriately revise the hypothesized explanation of age differences

## 13. Failure Escalation Rules

### Escalation 1: If Human Signatures Fail

If the model fails to reproduce the required behavioral signatures:

1. retune fitting-related parameters,
2. expand the tuning range for runtime / optimization parameters,
3. adjust search ranges for scale and temporal simulation settings,
4. only after that, consider deeper model changes.

### Escalation 2: If Joint Behavior Fit Fails

If RT can be fit but choice/accuracy/congruency cannot:

1. downgrade mechanism claims,
2. document which signatures fail,
3. re-evaluate whether the current Stage-2 parameterization is sufficient.

### Escalation 3: If Shared Stage 1 Appears Inadequate

If evidence strongly suggests that the shared Stage-1 front-end is masking age effects:

1. preserve the current main-path results,
2. open the age-specific Stage-1 branch as a future control analysis,
3. do not overwrite the interpretation of the main path retroactively.

### Escalation 4: If Noise Is Not the Best Explanation

If age differences are better captured by threshold-related variability or another effective parameter:

1. report that result directly,
2. revise the mechanism conclusion,
3. do not force a noise-dominant interpretation.

## 14. Minimal Viable Execution Flow

This is the default runnable order.

### Step 1: Prepare age-group data

- generate or verify age-group CSV splits
- verify RT summary statistics for `20-29` and `80-89`

### Step 2: Generate shared Stage-1 logits

- extract logits for both age groups using the same Stage-1 model
- save logits and metadata artifacts

### Step 3: Fit age-specific Stage-2 models

- fit one group-level Stage-2 model for `20-29`
- fit one group-level Stage-2 model for `80-89`
- preserve fitted parameters and evaluation outputs

### Step 4: Check baseline signatures

- RT distribution shape
- choice/accuracy
- congruency effect

### Step 5: Compare parameters

- `noise_ampa`
- threshold-related terms
- effective parameters such as `J_ext`, `I_0`, and summary properties of `J_matrix`

### Step 6: Run mechanism analysis

- use matched stimulus trials
- compare young vs old trajectory geometry
- compare congruent vs incongruent conditions

### Step 7: Write results in behavior -> parameter -> mechanism order

## 15. Agent Execution Protocol

Another agent should execute the project in this order:

1. verify prerequisite files and directories
2. run shared Stage-1 artifact generation
3. confirm Gate 1
4. run Stage-2 fitting for both age groups
5. confirm Gate 2 and Gate 3
6. run matched-trial geometry analysis
7. confirm Gate 4
8. write outputs in APA-style figure/table-ready form

## 16. Offline and Background Execution Constraints

The pipeline should be runnable **without internet access** once local data, checkpoints, and dependencies are present.

However, the following constraint must be stated explicitly:

- **Offline / disconnected operation is feasible.**
- **Closing a laptop lid is not guaranteed to keep local training running**, because the machine may sleep.

Therefore, the practical default is:

- run locally without internet dependence,
- use background/session tools for resumable execution,
- keep the machine awake during long jobs if they must continue locally.

## 17. Current Implementation Status

Completed or already available:

- age-group split logic
- stimulus image generation / mapping workflow
- shared Stage-1 checkpoint path
- scripts for Stage-1 logit extraction and Stage-2 fitting

Not yet completed under the revised plan:

- clean signature-gated evaluation of both age groups
- aligned parameter comparison under the revised interpretation
- matched-trial trajectory/state-space mechanism analysis
- final results write-up in behavior -> parameter -> mechanism order

## 18. Deliverables

The current phase should produce:

1. age-group logits artifacts
2. fitted Stage-2 parameter artifacts
3. human-signature comparison outputs
4. age-group parameter comparison summary
5. trajectory/state-space mechanism figures
6. a results narrative ordered as:
   - behavior,
   - parameters,
   - mechanism
