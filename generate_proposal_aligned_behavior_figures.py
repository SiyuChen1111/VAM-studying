from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


OUT_DIR = Path('results/proposal_aligned_behavior')
BLUE_FILL = '#B9CCE8'
BLUE_POINT = '#6F8FC1'
ORANGE_FILL = '#F3C7A2'
ORANGE_POINT = '#D9915B'
RED_FILL = '#EBC1C1'
RED_LINE = '#C97B7B'
GREEN_FILL = '#D0E8D8'
GREEN_LINE = '#7FB092'


def ensure_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_human_df(path, age_label):
    df = pd.read_csv(path).copy()
    df['age_group'] = age_label
    df['human_rt'] = df['response_time'] / 1000.0
    df['correct'] = (df['response_direction'] == df['target_direction']).astype(int)
    df['congruency'] = (df['target_direction'] != df['flanker_direction']).astype(int)
    df['condition'] = df['congruency'].map(lambda x: 'Congruent' if x == 0 else 'Incongruent')
    return df


def make_human_rt_distribution_figure(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    bins = np.linspace(0, 2, 81)
    xs = np.linspace(0, 2, 400)
    fill_colors = {'Congruent': BLUE_FILL, 'Incongruent': ORANGE_FILL}
    line_colors = {'Congruent': BLUE_POINT, 'Incongruent': ORANGE_POINT}
    for row, age in enumerate(['20-29 matched', '80-89']):
        age_df = df[df['age_group'] == age]
        for col, correctness in enumerate([1, 0]):
            ax = axes[row, col]
            subset = age_df[age_df['correct'] == correctness]
            for cond in ['Congruent', 'Incongruent']:
                series = subset.loc[subset['condition'] == cond, 'human_rt'].to_numpy()
                ax.hist(series, bins=bins, density=True, alpha=0.22, color=fill_colors[cond])
                if len(series) > 1:
                    ax.plot(xs, gaussian_kde(series)(xs), color=line_colors[cond], linewidth=2.2, label=cond)
            corr_label = 'Correct trials' if correctness == 1 else 'Error trials'
            ax.set_title(f'{age} — {corr_label}')
            ax.set_xlim(0, 2)
            ax.set_xlabel('RT (s)')
            ax.set_ylabel('Density')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row == 0 and col == 0:
                ax.legend(frameon=False)
    fig.suptitle('Figure P1. Human RT distributions by age, congruency, and correctness')
    fig.savefig(OUT_DIR / 'figureP1_human_rt_distributions.png', bbox_inches='tight')
    plt.close(fig)


def make_human_signature_figure(df):
    rows = []
    for age, grp in df.groupby('age_group'):
        correct_rt = grp.loc[grp['correct'] == 1, 'human_rt']
        error_rt = grp.loc[grp['correct'] == 0, 'human_rt']
        rows.append({
            'age_group': age,
            'mean_rt': grp['human_rt'].mean(),
            'skewness': grp['human_rt'].skew(),
            'accuracy': grp['correct'].mean(),
            'error_slower': error_rt.mean() - correct_rt.mean(),
            'cong_gap': grp.loc[grp['congruency'] == 1, 'human_rt'].mean() - grp.loc[grp['congruency'] == 0, 'human_rt'].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / 'human_behavior_signature_summary.csv', index=False)

    metrics = [('mean_rt', 'Mean RT (s)'), ('skewness', 'RT skewness'), ('accuracy', 'Accuracy'), ('error_slower', 'Error slower (s)'), ('cong_gap', 'Congruency RT gap (s)')]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(len(summary))
    for ax, (metric, title) in zip(axes, metrics):
        vals = summary[metric].to_numpy()
        ax.bar(x, vals, color=[BLUE_FILL, ORANGE_FILL], edgecolor=[BLUE_POINT, ORANGE_POINT], linewidth=1.2, alpha=0.9)
        for i, age in enumerate(summary['age_group']):
            age_df = df[df['age_group'] == age]
            if metric == 'mean_rt':
                pts = age_df.groupby('stimulus_layout')['human_rt'].mean().to_numpy()
            elif metric == 'skewness':
                pts = age_df.groupby('stimulus_layout')['human_rt'].apply(pd.Series.skew).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            elif metric == 'accuracy':
                pts = age_df.groupby('stimulus_layout')['correct'].mean().to_numpy()
            elif metric == 'error_slower':
                vals_by_layout = []
                for _, grp in age_df.groupby('stimulus_layout'):
                    c = grp.loc[grp['correct'] == 1, 'human_rt']
                    e = grp.loc[grp['correct'] == 0, 'human_rt']
                    if len(c) and len(e):
                        vals_by_layout.append(e.mean() - c.mean())
                pts = np.array(vals_by_layout)
            else:
                vals_by_layout = []
                for _, grp in age_df.groupby('stimulus_layout'):
                    if (grp['congruency'] == 0).any() and (grp['congruency'] == 1).any():
                        vals_by_layout.append(grp.loc[grp['congruency'] == 1, 'human_rt'].mean() - grp.loc[grp['congruency'] == 0, 'human_rt'].mean())
                pts = np.array(vals_by_layout)
            if len(pts):
                jitter = np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else np.array([0.0])
                ax.scatter(np.full(len(pts), i) + jitter, pts, color=[BLUE_POINT, ORANGE_POINT][i], s=18, alpha=0.5, edgecolors='white', linewidths=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(summary['age_group'], rotation=15)
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Figure P2. Human behavioral signatures by age group')
    fig.savefig(OUT_DIR / 'figureP2_human_signature_summary.png', bbox_inches='tight')
    plt.close(fig)


def make_model_summary_figure():
    df = pd.read_csv('results/age_groups_response_supervision_frozen/frozen_current_best_comparison.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(df))
    for ax, metric, title in zip(axes, ['score', 'model_accuracy', 'model_congruency_rt_gap'], ['Total score', 'Model accuracy', 'Model congruency gap (s)']):
        ax.bar(x, df[metric], color=[BLUE_FILL, ORANGE_FILL], edgecolor=[BLUE_POINT, ORANGE_POINT], linewidth=1.2, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(df['age_group'], rotation=15)
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Figure P3. Frozen current-best model summary by age group')
    fig.savefig(OUT_DIR / 'figureP3_model_summary.png', bbox_inches='tight')
    plt.close(fig)


def make_human_multipanel_figure(df):
    fig, axes = plt.subplots(5, 2, figsize=(11, 15), constrained_layout=True)
    fig.suptitle('Figure B2. Human behavioral profile for 20-29 matched and 80-89', fontsize=13)
    age_order = ['20-29 matched', '80-89']
    fill_colors = {'20-29 matched': BLUE_FILL, '80-89': ORANGE_FILL}
    point_colors = {'20-29 matched': BLUE_POINT, '80-89': ORANGE_POINT}
    cond_fill = {'Congruent': BLUE_FILL, 'Incongruent': ORANGE_FILL}
    cond_line = {'Congruent': BLUE_POINT, 'Incongruent': ORANGE_POINT}

    def add_row_label(ax, label, title):
        ax.text(-0.34, 1.12, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
        ax.text(-0.18, 1.12, title, transform=ax.transAxes, fontsize=12, va='top')

    def bar_scatter(ax, values, grouped, ylabel):
        x = np.arange(len(age_order))
        vals = [values[a] for a in age_order]
        ax.bar(x, vals, color=[fill_colors[a] for a in age_order], edgecolor=[point_colors[a] for a in age_order], linewidth=1.2, alpha=0.9)
        for i, age in enumerate(age_order):
            pts = grouped.get(age, np.array([]))
            if len(pts):
                jitter = np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else np.array([0.0])
                ax.scatter(np.full(len(pts), i) + jitter, pts, color=point_colors[age], s=18, alpha=0.5, edgecolors='white', linewidths=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(age_order, rotation=15)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    add_row_label(axes[0, 0], 'a', 'Accuracy')
    acc_values = {age: grp['correct'].mean() * 100 for age, grp in df.groupby('age_group')}
    acc_grouped = {age: grp.groupby('stimulus_layout')['correct'].mean().mul(100).to_numpy() for age, grp in df.groupby('age_group')}
    bar_scatter(axes[0, 0], acc_values, acc_grouped, 'Percentage correct (%)')
    axes[0, 1].axis('off')

    add_row_label(axes[1, 0], 'b', 'RT')
    rt_values = {age: grp['human_rt'].mean() for age, grp in df.groupby('age_group')}
    rt_grouped = {age: grp.groupby('stimulus_layout')['human_rt'].mean().to_numpy() for age, grp in df.groupby('age_group')}
    bar_scatter(axes[1, 0], rt_values, rt_grouped, 'Average RT (s)')
    axes[1, 1].axis('off')

    add_row_label(axes[2, 0], 'c', 'RT distributions')
    bins = np.linspace(0, 2, 81)
    xs = np.linspace(0, 2, 400)
    for col, age in enumerate(age_order):
        ax = axes[2, col]
        grp = df[df['age_group'] == age]
        for cond in ['Congruent', 'Incongruent']:
            series = grp.loc[grp['condition'] == cond, 'human_rt'].to_numpy()
            ax.hist(series, bins=bins, density=True, alpha=0.22, color=cond_fill[cond])
            if len(series) > 1:
                ax.plot(xs, gaussian_kde(series)(xs), color=cond_line[cond], linewidth=2.2, label=cond)
        ax.set_title(age)
        ax.set_xlim(0, 2)
        ax.set_xlabel('RT (s)')
        ax.set_ylabel('Density')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[2, 0].legend(frameon=False)

    add_row_label(axes[3, 0], 'd', 'Skew of RT distributions')
    skew_values = {age: grp['human_rt'].skew() for age, grp in df.groupby('age_group')}
    skew_grouped = {age: grp.groupby('stimulus_layout')['human_rt'].apply(pd.Series.skew).replace([np.inf, -np.inf], np.nan).dropna().to_numpy() for age, grp in df.groupby('age_group')}
    bar_scatter(axes[3, 0], skew_values, skew_grouped, 'Skewness')
    axes[3, 1].axis('off')

    add_row_label(axes[4, 0], 'e', 'RT for error and correct trials')
    outcome_fill = {'Error': RED_FILL, 'Correct': GREEN_FILL}
    outcome_line = {'Error': RED_LINE, 'Correct': GREEN_LINE}
    for col, age in enumerate(age_order):
        ax = axes[4, col]
        grp = df[df['age_group'] == age]
        values = []
        grouped = {}
        for label, corr in [('Error', 0), ('Correct', 1)]:
            rt = grp.loc[grp['correct'] == corr, 'human_rt']
            values.append(rt.mean())
            per_layout = []
            for _, lgrp in grp.groupby('stimulus_layout'):
                subset = lgrp.loc[lgrp['correct'] == corr, 'human_rt']
                if len(subset):
                    per_layout.append(subset.mean())
            grouped[label] = np.array(per_layout)
        x = np.arange(2)
        ax.bar(x, values, color=[outcome_fill['Error'], outcome_fill['Correct']], edgecolor=[outcome_line['Error'], outcome_line['Correct']], linewidth=1.2, alpha=0.9)
        for i, label in enumerate(['Error', 'Correct']):
            pts = grouped[label]
            if len(pts):
                jitter = np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else np.array([0.0])
                ax.scatter(np.full(len(pts), i) + jitter, pts, color=[outcome_line['Error'], outcome_line['Correct']][i], s=18, alpha=0.5, edgecolors='white', linewidths=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(['Error', 'Correct'])
        ax.set_title(age)
        ax.set_ylabel('Average RT (s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / 'figureB2_human_multipanel_20_29_vs_80_89.png', bbox_inches='tight')
    plt.close(fig)


def write_note():
    note = """# Proposal-aligned behavior figure note

Generated figures:
- Figure P1. Human RT distributions by age, congruency, and correctness
- Figure P2. Human behavioral signatures by age group
- Figure P3. Frozen current-best model summary by age group

Why these figures are available now:
- Human-side RT distributions, skewness, congruency effects, and error-slower summaries can be computed directly from the age-group test CSVs.
- Frozen current-best model summaries can be computed from the saved comparison CSV extracted from response-supervision logs.

What is not yet available:
- True model RT distribution plots under response-label supervision for both age groups
- Updated response-supervision trajectory geometry analogous to Figure 4

These model-level distribution and geometry plots require saved best-so-far parameter files or trial-level model predictions, which were not written before the runs were stopped.
"""
    (OUT_DIR / 'proposal_aligned_figure_note.md').write_text(note)


def write_analysis():
    spread = pd.read_csv('results/age_groups_interim/figureA4_interim_trajectory_spread.csv')
    human = pd.read_csv(OUT_DIR / 'human_behavior_signature_summary.csv')
    model = pd.read_csv('results/age_groups_response_supervision_frozen/frozen_current_best_comparison.csv')
    note = f"""# Integrated current-results analysis

## Behavioral patterns from the human data
The human behavior figures show the classic signatures emphasized in `research_proposal_v4.md`: age-related slowing, skewed RT distributions, congruency effects, and an error-slower component. The `80-89` group is slower on average than the matched `20-29` group, and the RT-distribution panels make the age difference visually obvious rather than reducing it to a single mean.

## Frozen current-best model comparison
Under the current frozen best-so-far response-supervision checkpoints, both model branches still drift back toward ceiling-level accuracy. However, the congruency RT gap is much closer to human behavior than it was under target-label supervision. This suggests that switching to response supervision corrected part of the regime problem, but did not eliminate the model's tendency toward overly idealized choice behavior.

## How to read Figure A4 right now
`figureA4_interim_trajectory_geometry.png` is still useful, but it should be interpreted as a geometry preview from the earlier supervision path rather than the final response-supervision mechanism figure. The spread summary shows:

{spread.to_markdown(index=False)}

This older geometry preview suggests that the `80-89` branch occupies a much broader state-space regime than `20-29`, regardless of congruency condition. That pattern is directionally compatible with the broader research hypothesis that older adults may require a noisier or more variable decision-dynamics regime. But because the response-supervision runs did not write best-so-far parameter snapshots before we stopped them, this geometry result cannot yet be treated as the final mechanism analysis for the corrected supervision branch.

## What is solid now
1. Human-side behavior plots can already support age-related slowing, skewness, congruency, and error-slower claims.
2. Frozen response-supervision model summaries can support a cautious behavioral comparison.
3. The old geometry figure can be used as a provisional mechanism preview, not as the final mechanism result.

## What remains blocked
True response-supervision model RT distributions and updated trajectory geometry still require saved best-so-far parameter files or full trial-level predictions from the corrected branch.
"""
    (OUT_DIR / 'integrated_current_results_analysis.md').write_text(note)


def main():
    ensure_dir()
    young = load_human_df('data_age_groups_matched/20-29/test_data.csv', '20-29 matched')
    old = load_human_df('data_age_groups/80-89/test_data.csv', '80-89')
    df = pd.concat([young, old], ignore_index=True)
    make_human_rt_distribution_figure(df)
    make_human_signature_figure(df)
    make_human_multipanel_figure(df)
    make_model_summary_figure()
    write_note()
    write_analysis()
    print(f'Saved proposal-aligned behavior figures to {OUT_DIR}')


if __name__ == '__main__':
    main()
