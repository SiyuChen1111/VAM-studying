# Research Proposal v4: Noise Effects on Reaction Time Distributions

## 1. Research Questions

### 1.1 Core Question

**Does noise affect human perceptual decision-making, and if so, how does it shape reaction time (RT) characteristics?**

### 1.2 Specific Questions

1. **Where does noise affect decision-making?** We examine three processing stages: sensory encoding, evidence accumulation, and decision threshold.
2. **How does noise at each stage affect RT characteristics?** We test whether different noise types produce distinguishable effects on RT distributions.
3. **Do noise effects generalize across tasks?** We validate noise effects across MNIST, RDM, and Flanker tasks.
4. **Do noise effects differ across age groups?** We examine whether age-related RT differences in the Flanker/LIM task can be explained by changes in effective internal decision-dynamics parameters, with internal noise as the primary candidate mechanism.

***

## 2. Research Hypothesis

**H**: Noise is a fundamental mechanism shaping RT characteristics in perceptual decision-making. Specifically:

1. **Encoding noise** reduces stimulus discriminability, leading to lower accuracy and longer RTs, with stronger effects under higher task difficulty.
2. **Sampling noise** increases evidence accumulation variability, producing higher RT variance, right-skewed RT distributions, and the error-slower effect.
3. **Threshold fluctuations** modulate speed-accuracy trade-offs, with lower thresholds yielding faster RTs but lower accuracy.
4. **Cross-task generalization**: These noise effects generalize across MNIST, RDM, and Flanker tasks.
5. **Age differences**: Age-related RT differences (slower RTs, higher RT variability in older adults) are hypothesized to be explained primarily by higher fitted sampling-noise parameters (`noise_ampa`) in the WongWang decision module, with threshold-related variability treated as a secondary competing explanation.

**Validation Summary**:

| Noise Type       | Key Predictions                          | Validation Method                          |
| ---------------- | ---------------------------------------- | ------------------------------------------ |
| Encoding noise   | Lower accuracy, longer RT under noise    | RSA on CNN activations                     |
| Sampling noise   | Higher RT variance, right-skewed RT, error-slower | Accumulation trajectory visualization |
| Threshold        | SAT modulation                           | Compare speed vs accuracy conditions       |
| Cross-task       | Similar effects across tasks             | Compare noise effects on MNIST/RDM/Flanker |
| Age differences  | Higher fitted internal fluctuation in older adults | Fit model to age groups, compare parameters and trajectory geometry |

***

## 3. Validation Methods

### 3.1 Behavioral Validation

Human decision signatures are treated as **necessary but not sufficient** for mechanism claims. Reproducing these signatures establishes model adequacy; mechanism claims require additional parameter and trajectory analyses.

| Prediction                        | Validation Method                    | Success Criterion                                |
| --------------------------------- | ------------------------------------ | ------------------------------------------------ |
| Stochastic responses              | RT variance across trials            | RT variance > 0                                  |
| SAT modulation                    | Compare speed vs accuracy conditions | Accuracy difference > 10%, RT difference > 100ms |
| Difficulty effects                | Compare easy vs difficult conditions | Accuracy difference > 20%, RT difference > 100ms |
| Right-skewed RT                   | Compute skewness                     | Skewness > 0.5                                   |
| Error-slower                      | Compare RT_error vs RT_correct       | RT_error - RT_correct > 50ms                     |

### 3.2 Representational Analysis

**Encoding Noise**:

- Method: Representational Similarity Analysis (RSA) on CNN activations
- Prediction: Higher encoding noise → lower stimulus discriminability in RSM

**Sampling Noise**:

- Method: Evidence accumulation trajectory visualization
- Prediction: Higher sampling noise → more variable trajectories
- Age extension (Flanker/LIM): compare trajectory/state-space geometry between `20-29` and `80-89` groups using a shared Stage-1 visual front-end and matched congruent/incongruent trials

### 3.3 Cross-Task Validation

- Test same noise manipulations on MNIST, RDM, and Flanker tasks
- Compare noise sensitivity across tasks
- Examine noise × congruency interaction in Flanker task

### 3.4 Age-Related Validation

- Fit the Flanker/LIM model to `20-29` and `80-89` age groups at the group level
- Use a **shared Stage-1** model as the main analysis path; treat age-specific Stage-1 training as a future control branch
- Compare fitted `noise_ampa` and threshold-related parameters across age groups
- Evaluate RT, choice/accuracy, and congruency jointly rather than RT alone
- Test whether age-related differences in trajectory/state-space geometry under congruent and incongruent conditions support the parameter-level interpretation

***

## 4. Model Implementation

### 4.1 Architecture

```
Input Image (task-specific size)
    ↓
VGG16 (pretrained feature extraction)
    ↓
FC Layer → logits (task-specific classes)
    ↓
Linear Transform → input signals for decision module
    ↓
WongWang Decision Module (with sampling noise via AMPA receptors)
    ↓
Decision Times → RT prediction
```

**Architecture Rationale**:

1. **VGG16 Feature Extraction**: Uses pretrained VGG16 for robust visual feature extraction, suitable for complex stimuli across different tasks

2. **FC Layer (Task-Specific)**:
   - MNIST: 10 classes (digits 0-9)
   - RDM: 2 classes (up/down motion direction)
   - Flanker: 4 classes (L/R/U/D directions)

3. **WongWang Decision Module**: Implements biophysically realistic evidence accumulation with:
   - Competing neural populations (n_classes populations for n_choices)
   - NMDA receptor-mediated slow recurrent excitation (τ_s = 100ms)
   - AMPA receptor-mediated noise (Ornstein-Uhlenbeck process)
   - Winner-take-all competition mechanism
   - **Trainable parameters**: J_matrix, J_ext, I_0, noise_ampa, threshold
   - **Key extensions from Wong & Wang (2006)**:
     - Differentiability via custom autograd function (DiffDecisionMultiClass)
     - Multi-class support (2 to n_classes competing populations)
     - CNN integration (accepts CNN features as input signals)
     - Batch processing (parallel simulation for efficient training)

4. **Behavioral Prediction**: Decision dynamics are evaluated against RT, choice/accuracy, and congruency signatures
    - Uses differentiable decision time computation from RTify (Cheng et al., 2024)
    - Enables end-to-end training with RT loss
    - Mechanism claims are based on both fitted parameters and trajectory geometry

### 4.2 Noise Implementation

**Encoding Noise**:

- **Implementation**: Stimulus-level noise added to input images (Gaussian noise)
- **Analysis**: RSA on CNN activations to examine stimulus discriminability
- **Hypothesis**: Encoding noise reduces stimulus discriminability, leading to lower accuracy and longer RTs

**Sampling Noise**:

- **Implementation**: Based on Wong-Wang (2006) model
  - Gaussian noise injected through AMPA receptor-mediated input currents
  - Noise evolves according to Ornstein-Uhlenbeck process
  - Parameters:
    - noise_ampa: AMPA noise intensity
    - tau_ampa: AMPA receptor time constant
    - tau_s: NMDA receptor time constant
    - J_matrix: Recurrent connection matrix
    - J_ext: External input strength
    - I_0: background input
    - threshold: Decision threshold
  - Code implementation:
    ```python
    # Noise evolution at each time step (Ornstein-Uhlenbeck process)
    I_noise = I_noise * torch.exp(-dt / tau_ampa) + \
        noise_ampa * torch.sqrt((1 - torch.exp(-2 * dt / tau_ampa)) / 2.0) * \
        torch.randn(batch_size, n_classes)
    ```

**Threshold Fluctuations**:

- **Implementation**: Manipulation of decision thresholds under different SAT conditions
- **Analysis**: Compare SAT curves across threshold conditions; test which implementation better matches human data
- **Hypothesis**: Threshold-related variability is a secondary candidate mechanism that may contribute to age and SAT-related differences even when internal sampling noise remains the primary hypothesis

***

## 5. Tasks and Data

| Task    | Data Source                        | Key Manipulation            |
| ------- | ---------------------------------- | --------------------------- |
| MNIST   | `/data/raw/rtnet/`                 | Encoding noise (2.1 vs 2.9) |
| RDM     | `/data/raw/rdm/`                   | Motion coherence (5-80%)    |
| Flanker | `/data/raw/vam/` (75 participants) | Congruency, age groups      |

***

## 6. Expected Outcomes

### 6.1 Core Predictions

- ✓ Encoding noise effects on accuracy and RT
- ✓ Sampling noise effects on RT variance and distribution shape
- ✓ Threshold effects on SAT modulation
- ✓ Cross-task generalization of noise effects

### 6.2 Extension (Key Contribution if Achieved)

- ✓ Demonstrate age differences in fitted internal fluctuation parameters in the Flanker/LIM task
- ✓ Show that age-related RT differences can be interpreted through a combination of parameter comparison and trajectory/state-space analysis

***
