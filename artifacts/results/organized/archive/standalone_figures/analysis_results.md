# Human Data Analysis Report

## 1. Overview

This report analyzes human behavioral data from the Lost in Migration (LIM) task to understand key signatures of perceptual decision-making.

**Total samples**: 3,229,416
**Age groups**: 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89

## 2. RT Distribution Analysis

### 2.1 Overall RT Distribution

| Metric | Value |
|--------|-------|
| Mean | 0.697s |
| Median | 0.666s |
| Std | 0.216s |
| Min | 0.006s |
| Max | 43.282s |
| **Skewness** | **26.084** |
| Kurtosis | 3148.212 |

**Interpretation**: 
- Skewness > 0.5 indicates right-skewed distribution ✓
- Current skewness = 26.084 → Right-skewed ✓

### 2.2 Percentiles

| Percentile | RT (s) |
|------------|--------|
| 5% | 0.483 |
| 25% | 0.576 |
| 50% (Median) | 0.666 |
| 75% | 0.777 |
| 95% | 1.009 |
| 99% | 1.294 |

## 3. Congruency Effect Analysis

### 3.1 Congruency Definition

In LIM task, we define congruency based on **flanker_direction vs response_direction**:
- **Congruent**: flanker_direction == response_direction
- **Incongruent**: flanker_direction != response_direction

### 3.2 Overall Congruency Effect

| Condition | Mean RT | Median RT | Std | N |
|-----------|---------|-----------|-----|---|
| Congruent | 0.665s | 0.637s | 0.208s | 1,617,633 |
| Incongruent | 0.730s | 0.700s | 0.219s | 1,611,783 |
| **Difference** | **0.065s** | | | |

**Statistical Test**: t = -273.605, p = 0.00e+00

**Interpretation**: 
- Congruency effect = 0.065s (incongruent slower)
- This is consistent with the typical congruency effect in Flanker tasks

### 3.3 Congruency Effect by Age Group

| Age Group | Congruent Mean | Incongruent Mean | Difference |
|-----------|----------------|------------------|------------|
| 20-29 | 0.583s | 0.627s | 0.044s |
| 30-39 | 0.618s | 0.675s | 0.057s |
| 40-49 | 0.595s | 0.662s | 0.067s |
| 50-59 | 0.634s | 0.698s | 0.065s |
| 60-69 | 0.682s | 0.756s | 0.073s |
| 70-79 | 0.746s | 0.822s | 0.076s |
| 80-89 | 0.897s | 0.982s | 0.085s |

## 4. Error-Slower Effect Analysis

### 4.1 Overall Error-Slower Effect

| Condition | Mean RT | Median RT | Std | N |
|-----------|---------|-----------|-----|---|
| Correct | nans | nans | nans | 0 |
| Error | 0.697s | 0.666s | 0.216s | 3,229,416 |
| **Difference** | **nans** | | | |

**Statistical Test**: t = nan, p = nan

**Interpretation**: 
- Error-slower effect = nans (error faster)
- Typical error-slower effect: 50-100ms
- Current effect is different from typical findings

### 4.2 Error-Slower Effect by Age Group

| Age Group | Correct Mean | Error Mean | Difference |
|-----------|--------------|------------|------------|
| 20-29 | nans | 0.605s | nans |
| 30-39 | nans | 0.647s | nans |
| 40-49 | nans | 0.629s | nans |
| 50-59 | nans | 0.666s | nans |
| 60-69 | nans | 0.719s | nans |
| 70-79 | nans | 0.784s | nans |
| 80-89 | nans | 0.939s | nans |

## 5. Age Group Analysis

### 5.1 RT by Age Group

| Age Group | N | Mean RT | Median RT | Std | Skewness |
|-----------|---|---------|-----------|-----|----------|
| 20-29 | 728,809 | 0.605s | 0.580s | 0.188s | 40.219 |
| 30-39 | 287,531 | 0.647s | 0.619s | 0.181s | 38.587 |
| 40-49 | 122,926 | 0.629s | 0.608s | 0.172s | 80.156 |
| 50-59 | 447,311 | 0.666s | 0.648s | 0.171s | 90.798 |
| 60-69 | 882,896 | 0.719s | 0.695s | 0.192s | 7.230 |
| 70-79 | 633,329 | 0.784s | 0.759s | 0.208s | 19.512 |
| 80-89 | 126,614 | 0.939s | 0.879s | 0.377s | 23.946 |

## 6. Summary of Human Signatures

| Signature | Criterion | Human Data | Status |
|-----------|-----------|------------|--------|
| Right-skewed RT | Skewness > 0.5 | 26.084 | ✓ PASS |
| Congruency effect | Incongruent > Congruent | 0.065s | ✓ PASS |
| Error-slower effect | Error > Correct (50-100ms) | nans | ⚠ CHECK |

## 7. Implications for Model Training

Based on the analysis:

1. **RT Distribution**: Model should produce right-skewed RT distribution (skewness > 0.5)

2. **Congruency Effect**: Model should show slower RT for incongruent trials

3. **Error-Slower Effect**: Model may not show typical error-slower effect

4. **Age Differences**: 
   - Young group (20-29): Mean RT = 0.605s
   - Old group (80-89): Mean RT = 0.939s
   - Age difference = 0.335s (55.3%)

## 8. Generated Figures

- `rt_distribution_by_age.png`: RT distributions for each age group
- `congruency_effect.png`: Congruency effect overall and by age
- `error_slower_effect.png`: Error-slower effect overall and by age
