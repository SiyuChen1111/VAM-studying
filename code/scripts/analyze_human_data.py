"""
Human Data Analysis for Lost in Migration (LIM) Task

This script analyzes human behavioral data to understand:
1. Congruency effect: RT difference between congruent and incongruent conditions
2. Error-slower effect: RT difference between correct and error trials
3. RT distribution shape: Skewness and right-tail characteristics
4. Age group differences: RT characteristics across different age groups

Output: analysis_results.md with detailed findings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

def load_data():
    """Load all user data and merge with metadata."""
    metadata = pd.read_csv('vam_data/metadata.csv')
    data_dir = 'vam_data'
    all_data = []
    
    for f in os.listdir(data_dir):
        if f.startswith('user') and f.endswith('df.csv'):
            user_id = int(f.replace('user', '').replace('df.csv', ''))
            df = pd.read_csv(os.path.join(data_dir, f))
            df['user_id'] = user_id
            all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.merge(metadata[['user_id', 'binned_age']], on='user_id', how='left')
    df['rt_s'] = df['response_time'] / 1000
    
    return df

def analyze_congruency(df):
    """
    Analyze congruency effect.
    
    In LIM task:
    - target_direction: target direction (L/R/U/D)
    - flanker_direction: flanker direction (L/R/U/D)
    - response_direction: direction of response (L/R/U/D)
    
    Congruency definition:
    - Congruent: flanker_direction == target_direction (flanker与target方向相同)
    - Incongruent: flanker_direction != target_direction (flanker与target方向不同)
    """
    results = {}
    
    # All columns are strings (L/R/U/D)
    # Congruency based on flanker vs target
    df['is_congruent'] = (df['flanker_direction'] == df['target_direction']).astype(int)
    
    # Analyze RT by congruency
    congruent_rt = df[df['is_congruent'] == 1]['rt_s']
    incongruent_rt = df[df['is_congruent'] == 0]['rt_s']
    
    results['method1'] = {
        'congruent_mean': congruent_rt.mean(),
        'congruent_median': congruent_rt.median(),
        'congruent_std': congruent_rt.std(),
        'congruent_n': len(congruent_rt),
        'incongruent_mean': incongruent_rt.mean(),
        'incongruent_median': incongruent_rt.median(),
        'incongruent_std': incongruent_rt.std(),
        'incongruent_n': len(incongruent_rt),
        'difference': incongruent_rt.mean() - congruent_rt.mean(),
        't_stat': stats.ttest_ind(congruent_rt, incongruent_rt)[0],
        'p_value': stats.ttest_ind(congruent_rt, incongruent_rt)[1]
    }
    
    return results

def analyze_error_slower(df):
    """
    Analyze error-slower effect.
    
    Error-slower effect: RT on error trials is slower than RT on correct trials.
    
    Correct response: response_direction == target_direction
    """
    results = {}
    
    # Map target_direction to numeric
    target_to_resp = {'L': 0, 'R': 1, 'U': 2, 'D': 3}
    df['target_resp'] = df['target_direction'].map(target_to_resp)
    
    # Define correct trials
    df['is_correct'] = (df['target_resp'] == df['response_direction']).astype(int)
    
    correct_rt = df[df['is_correct'] == 1]['rt_s']
    error_rt = df[df['is_correct'] == 0]['rt_s']
    
    results = {
        'correct_mean': correct_rt.mean(),
        'correct_median': correct_rt.median(),
        'correct_std': correct_rt.std(),
        'correct_n': len(correct_rt),
        'error_mean': error_rt.mean(),
        'error_median': error_rt.median(),
        'error_std': error_rt.std(),
        'error_n': len(error_rt),
        'difference': error_rt.mean() - correct_rt.mean(),
        't_stat': stats.ttest_ind(correct_rt, error_rt)[0],
        'p_value': stats.ttest_ind(correct_rt, error_rt)[1]
    }
    
    return results

def analyze_rt_distribution(df):
    """
    Analyze RT distribution shape.
    
    Key metrics:
    - Skewness: positive indicates right-skewed
    - Kurtosis: measures tail heaviness
    - Percentiles: distribution spread
    """
    results = {}
    
    rt = df['rt_s']
    
    results['skewness'] = stats.skew(rt)
    results['kurtosis'] = stats.kurtosis(rt)
    results['mean'] = rt.mean()
    results['median'] = rt.median()
    results['std'] = rt.std()
    results['min'] = rt.min()
    results['max'] = rt.max()
    
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        results[f'percentile_{p}'] = np.percentile(rt, p)
    
    return results

def analyze_by_age_group(df):
    """Analyze RT characteristics by age group."""
    results = {}
    
    for age in sorted(df['binned_age'].unique()):
        group_df = df[df['binned_age'] == age]
        group_rt = group_df['rt_s']
        
        results[age] = {
            'n': len(group_df),
            'mean': group_rt.mean(),
            'median': group_rt.median(),
            'std': group_rt.std(),
            'skewness': stats.skew(group_rt),
            'min': group_rt.min(),
            'max': group_rt.max()
        }
    
    return results

def analyze_congruency_by_age(df):
    """Analyze congruency effect by age group."""
    results = {}
    
    for age in sorted(df['binned_age'].unique()):
        group_df = df[df['binned_age'] == age]
        
        congruent = group_df[group_df['is_congruent'] == 1]['rt_s']
        incongruent = group_df[group_df['is_congruent'] == 0]['rt_s']
        
        results[age] = {
            'congruent_mean': congruent.mean(),
            'incongruent_mean': incongruent.mean(),
            'difference': incongruent.mean() - congruent.mean(),
            'congruent_n': len(congruent),
            'incongruent_n': len(incongruent)
        }
    
    return results

def analyze_error_slower_by_age(df):
    """Analyze error-slower effect by age group."""
    results = {}
    
    for age in sorted(df['binned_age'].unique()):
        group_df = df[df['binned_age'] == age]
        
        correct = group_df[group_df['is_correct'] == 1]['rt_s']
        error = group_df[group_df['is_correct'] == 0]['rt_s']
        
        results[age] = {
            'correct_mean': correct.mean(),
            'error_mean': error.mean(),
            'difference': error.mean() - correct.mean(),
            'correct_n': len(correct),
            'error_n': len(error)
        }
    
    return results

def plot_rt_distribution(df, save_dir='results'):
    """Plot RT distribution overall and by age group."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Overall distribution
    ax = axes[0, 0]
    rt = df['rt_s']
    ax.hist(rt, bins=100, density=True, alpha=0.7, color='blue')
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Density')
    ax.set_title(f'Overall RT Distribution\nSkewness={stats.skew(rt):.3f}')
    ax.axvline(rt.mean(), color='red', linestyle='--', label=f'Mean={rt.mean():.3f}s')
    ax.axvline(rt.median(), color='green', linestyle='--', label=f'Median={rt.median():.3f}s')
    ax.legend()
    
    # By age group
    ages = sorted(df['binned_age'].unique())
    for i, age in enumerate(ages[:7]):
        ax = axes.flatten()[i + 1]
        group_rt = df[df['binned_age'] == age]['rt_s']
        ax.hist(group_rt, bins=50, density=True, alpha=0.7, color='blue')
        ax.set_xlabel('RT (s)')
        ax.set_ylabel('Density')
        ax.set_title(f'{age}\nSkew={stats.skew(group_rt):.3f}, Mean={group_rt.mean():.3f}s')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rt_distribution_by_age.png'), dpi=150)
    plt.close()
    
    # Congruency effect plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall congruency
    ax = axes[0]
    congruent = df[df['is_congruent'] == 1]['rt_s']
    incongruent = df[df['is_congruent'] == 0]['rt_s']
    ax.hist(congruent, bins=50, density=True, alpha=0.5, label=f'Congruent (n={len(congruent)})')
    ax.hist(incongruent, bins=50, density=True, alpha=0.5, label=f'Incongruent (n={len(incongruent)})')
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Density')
    ax.set_title(f'Congruency Effect\nDiff={incongruent.mean()-congruent.mean():.3f}s')
    ax.legend()
    
    # Congruency by age
    ax = axes[1]
    age_list = sorted(df['binned_age'].unique())
    x = np.arange(len(age_list))
    width = 0.35
    
    congruent_means = [df[(df['binned_age'] == age) & (df['is_congruent'] == 1)]['rt_s'].mean() for age in age_list]
    incongruent_means = [df[(df['binned_age'] == age) & (df['is_congruent'] == 0)]['rt_s'].mean() for age in age_list]
    
    ax.bar(x - width/2, congruent_means, width, label='Congruent')
    ax.bar(x + width/2, incongruent_means, width, label='Incongruent')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Mean RT (s)')
    ax.set_title('Congruency Effect by Age Group')
    ax.set_xticks(x)
    ax.set_xticklabels(age_list, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'congruency_effect.png'), dpi=150)
    plt.close()
    
    # Error-slower effect plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall error-slower
    ax = axes[0]
    correct = df[df['is_correct'] == 1]['rt_s']
    error = df[df['is_correct'] == 0]['rt_s']
    ax.hist(correct, bins=50, density=True, alpha=0.5, label=f'Correct (n={len(correct)})')
    ax.hist(error, bins=50, density=True, alpha=0.5, label=f'Error (n={len(error)})')
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Density')
    ax.set_title(f'Error-Slower Effect\nDiff={error.mean()-correct.mean():.3f}s')
    ax.legend()
    
    # Error-slower by age
    ax = axes[1]
    correct_means = [df[(df['binned_age'] == age) & (df['is_correct'] == 1)]['rt_s'].mean() for age in age_list]
    error_means = [df[(df['binned_age'] == age) & (df['is_correct'] == 0)]['rt_s'].mean() for age in age_list]
    
    ax.bar(x - width/2, correct_means, width, label='Correct')
    ax.bar(x + width/2, error_means, width, label='Error')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Mean RT (s)')
    ax.set_title('Error-Slower Effect by Age Group')
    ax.set_xticks(x)
    ax.set_xticklabels(age_list, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_slower_effect.png'), dpi=150)
    plt.close()

def generate_report(df, congruency_results, error_slower_results, rt_dist_results, 
                   age_results, congruency_age_results, error_slower_age_results, save_dir='results'):
    """Generate markdown report with all findings."""
    
    report = """# Human Data Analysis Report

## 1. Overview

This report analyzes human behavioral data from the Lost in Migration (LIM) task to understand key signatures of perceptual decision-making.

**Total samples**: {total_n:,}
**Age groups**: {age_groups}

## 2. RT Distribution Analysis

### 2.1 Overall RT Distribution

| Metric | Value |
|--------|-------|
| Mean | {rt_mean:.3f}s |
| Median | {rt_median:.3f}s |
| Std | {rt_std:.3f}s |
| Min | {rt_min:.3f}s |
| Max | {rt_max:.3f}s |
| **Skewness** | **{skewness:.3f}** |
| Kurtosis | {kurtosis:.3f} |

**Interpretation**: 
- Skewness > 0.5 indicates right-skewed distribution ✓
- Current skewness = {skewness:.3f} → {skewness_interpretation}

### 2.2 Percentiles

| Percentile | RT (s) |
|------------|--------|
| 5% | {p5:.3f} |
| 25% | {p25:.3f} |
| 50% (Median) | {p50:.3f} |
| 75% | {p75:.3f} |
| 95% | {p95:.3f} |
| 99% | {p99:.3f} |

## 3. Congruency Effect Analysis

### 3.1 Congruency Definition

In LIM task, we define congruency based on **flanker_direction vs response_direction**:
- **Congruent**: flanker_direction == response_direction
- **Incongruent**: flanker_direction != response_direction

### 3.2 Overall Congruency Effect

| Condition | Mean RT | Median RT | Std | N |
|-----------|---------|-----------|-----|---|
| Congruent | {cong_mean:.3f}s | {cong_med:.3f}s | {cong_std:.3f}s | {cong_n:,} |
| Incongruent | {incong_mean:.3f}s | {incong_med:.3f}s | {incong_std:.3f}s | {incong_n:,} |
| **Difference** | **{cong_diff:.3f}s** | | | |

**Statistical Test**: t = {cong_t:.3f}, p = {cong_p:.2e}

**Interpretation**: 
- Congruency effect = {cong_diff:.3f}s ({cong_interpretation})
- This is {cong_expected} the typical congruency effect in Flanker tasks

### 3.3 Congruency Effect by Age Group

| Age Group | Congruent Mean | Incongruent Mean | Difference |
|-----------|----------------|------------------|------------|
""".format(
        total_n=len(df),
        age_groups=', '.join(sorted(df['binned_age'].unique())),
        rt_mean=rt_dist_results['mean'],
        rt_median=rt_dist_results['median'],
        rt_std=rt_dist_results['std'],
        rt_min=rt_dist_results['min'],
        rt_max=rt_dist_results['max'],
        skewness=rt_dist_results['skewness'],
        kurtosis=rt_dist_results['kurtosis'],
        skewness_interpretation="Right-skewed ✓" if rt_dist_results['skewness'] > 0.5 else "Not sufficiently right-skewed",
        p5=rt_dist_results['percentile_5'],
        p25=rt_dist_results['percentile_25'],
        p50=rt_dist_results['percentile_50'],
        p75=rt_dist_results['percentile_75'],
        p95=rt_dist_results['percentile_95'],
        p99=rt_dist_results['percentile_99'],
        cong_mean=congruency_results['method1']['congruent_mean'],
        cong_med=congruency_results['method1']['congruent_median'],
        cong_std=congruency_results['method1']['congruent_std'],
        cong_n=congruency_results['method1']['congruent_n'],
        incong_mean=congruency_results['method1']['incongruent_mean'],
        incong_med=congruency_results['method1']['incongruent_median'],
        incong_std=congruency_results['method1']['incongruent_std'],
        incong_n=congruency_results['method1']['incongruent_n'],
        cong_diff=congruency_results['method1']['difference'],
        cong_t=congruency_results['method1']['t_stat'],
        cong_p=congruency_results['method1']['p_value'],
        cong_interpretation="incongruent slower" if congruency_results['method1']['difference'] > 0 else "incongruent faster",
        cong_expected="consistent with" if congruency_results['method1']['difference'] > 0 else "opposite to"
    )
    
    for age in sorted(congruency_age_results.keys()):
        r = congruency_age_results[age]
        report += f"| {age} | {r['congruent_mean']:.3f}s | {r['incongruent_mean']:.3f}s | {r['difference']:.3f}s |\n"
    
    report += """
## 4. Error-Slower Effect Analysis

### 4.1 Overall Error-Slower Effect

| Condition | Mean RT | Median RT | Std | N |
|-----------|---------|-----------|-----|---|
| Correct | {corr_mean:.3f}s | {corr_med:.3f}s | {corr_std:.3f}s | {corr_n:,} |
| Error | {err_mean:.3f}s | {err_med:.3f}s | {err_std:.3f}s | {err_n:,} |
| **Difference** | **{err_diff:.3f}s** | | | |

**Statistical Test**: t = {err_t:.3f}, p = {err_p:.2e}

**Interpretation**: 
- Error-slower effect = {err_diff:.3f}s ({err_interpretation})
- Typical error-slower effect: 50-100ms
- Current effect is {err_expected} typical findings

### 4.2 Error-Slower Effect by Age Group

| Age Group | Correct Mean | Error Mean | Difference |
|-----------|--------------|------------|------------|
""".format(
        corr_mean=error_slower_results['correct_mean'],
        corr_med=error_slower_results['correct_median'],
        corr_std=error_slower_results['correct_std'],
        corr_n=error_slower_results['correct_n'],
        err_mean=error_slower_results['error_mean'],
        err_med=error_slower_results['error_median'],
        err_std=error_slower_results['error_std'],
        err_n=error_slower_results['error_n'],
        err_diff=error_slower_results['difference'],
        err_t=error_slower_results['t_stat'],
        err_p=error_slower_results['p_value'],
        err_interpretation="error slower" if error_slower_results['difference'] > 0 else "error faster",
        err_expected="consistent with" if 0.05 < error_slower_results['difference'] < 0.15 else "different from"
    )
    
    for age in sorted(error_slower_age_results.keys()):
        r = error_slower_age_results[age]
        report += f"| {age} | {r['correct_mean']:.3f}s | {r['error_mean']:.3f}s | {r['difference']:.3f}s |\n"
    
    report += """
## 5. Age Group Analysis

### 5.1 RT by Age Group

| Age Group | N | Mean RT | Median RT | Std | Skewness |
|-----------|---|---------|-----------|-----|----------|
"""
    
    for age in sorted(age_results.keys()):
        r = age_results[age]
        report += f"| {age} | {r['n']:,} | {r['mean']:.3f}s | {r['median']:.3f}s | {r['std']:.3f}s | {r['skewness']:.3f} |\n"
    
    report += """
## 6. Summary of Human Signatures

| Signature | Criterion | Human Data | Status |
|-----------|-----------|------------|--------|
| Right-skewed RT | Skewness > 0.5 | {skewness:.3f} | {skew_status} |
| Congruency effect | Incongruent > Congruent | {cong_diff:.3f}s | {cong_status} |
| Error-slower effect | Error > Correct (50-100ms) | {err_diff:.3f}s | {err_status} |

## 7. Implications for Model Training

Based on the analysis:

1. **RT Distribution**: Model should produce right-skewed RT distribution (skewness > 0.5)

2. **Congruency Effect**: Model should show slower RT for incongruent trials

3. **Error-Slower Effect**: {err_implication}

4. **Age Differences**: 
   - Young group (20-29): Mean RT = {young_rt:.3f}s
   - Old group (80-89): Mean RT = {old_rt:.3f}s
   - Age difference = {age_diff:.3f}s ({age_diff_pct:.1f}%)

## 8. Generated Figures

- `rt_distribution_by_age.png`: RT distributions for each age group
- `congruency_effect.png`: Congruency effect overall and by age
- `error_slower_effect.png`: Error-slower effect overall and by age
""".format(
        skewness=rt_dist_results['skewness'],
        skew_status="✓ PASS" if rt_dist_results['skewness'] > 0.5 else "✗ FAIL",
        cong_diff=congruency_results['method1']['difference'],
        cong_status="✓ PASS" if congruency_results['method1']['difference'] > 0 else "✗ FAIL",
        err_diff=error_slower_results['difference'],
        err_status="✓ PASS" if 0.05 < error_slower_results['difference'] < 0.15 else "⚠ CHECK",
        err_implication="Model should show slower RT on error trials" if error_slower_results['difference'] > 0 else "Model may not show typical error-slower effect",
        young_rt=age_results['20-29']['mean'],
        old_rt=age_results['80-89']['mean'],
        age_diff=age_results['80-89']['mean'] - age_results['20-29']['mean'],
        age_diff_pct=(age_results['80-89']['mean'] - age_results['20-29']['mean']) / age_results['20-29']['mean'] * 100
    )
    
    with open(os.path.join(save_dir, 'analysis_results.md'), 'w') as f:
        f.write(report)
    
    print(f"Report saved to {os.path.join(save_dir, 'analysis_results.md')}")
    return report

def main():
    print("Loading data...")
    df = load_data()
    
    print("Analyzing congruency effect...")
    congruency_results = analyze_congruency(df)
    
    print("Analyzing error-slower effect...")
    error_slower_results = analyze_error_slower(df)
    
    print("Analyzing RT distribution...")
    rt_dist_results = analyze_rt_distribution(df)
    
    print("Analyzing by age group...")
    age_results = analyze_by_age_group(df)
    congruency_age_results = analyze_congruency_by_age(df)
    error_slower_age_results = analyze_error_slower_by_age(df)
    
    print("Generating plots...")
    plot_rt_distribution(df)
    
    print("Generating report...")
    report = generate_report(df, congruency_results, error_slower_results, rt_dist_results,
                            age_results, congruency_age_results, error_slower_age_results)
    
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    print(f"RT Skewness: {rt_dist_results['skewness']:.3f} (> 0.5 = right-skewed)")
    print(f"Congruency Effect: {congruency_results['method1']['difference']:.3f}s")
    print(f"Error-Slower Effect: {error_slower_results['difference']:.3f}s")
    print(f"Age RT Difference (80-89 vs 20-29): {age_results['80-89']['mean'] - age_results['20-29']['mean']:.3f}s")

if __name__ == "__main__":
    main()
