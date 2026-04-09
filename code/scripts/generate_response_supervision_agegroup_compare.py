from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


OUT_DIR = Path('results/age_groups_response_supervision_interim')
BLUE_FILL = '#B9CCE8'
BLUE_POINT = '#6F8FC1'
ORANGE_FILL = '#F3C7A2'
ORANGE_POINT = '#D9915B'


def ensure_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def format_p_value(p):
    if p < 0.001:
        return 'p < .001'
    return f'p = {p:.3f}'.replace('0.', '.')


def add_sig(ax, p, y):
    if not np.isfinite(p) or p >= 0.05:
        return
    x1, x2 = 0, 1
    h = max(y * 0.05, 0.01)
    y0 = y * 1.03 if y > 0 else 0.02
    ax.plot([x1, x1, x2, x2], [y0, y0 + h, y0 + h, y0], color='#555555', linewidth=1.0)
    ax.text((x1 + x2) / 2, y0 + h * 1.15, format_p_value(p), ha='center', va='bottom', fontsize=10)


def human_metric_distributions():
    young = pd.read_csv('data_age_groups_matched/20-29/test_data.csv').copy()
    old = pd.read_csv('data_age_groups/80-89/test_data.csv').copy()

    def enrich(df):
        d = df.copy()
        d['correct'] = (d['target_direction'] == d['response_direction']).astype(float)
        d['congruency'] = (d['target_direction'] != d['flanker_direction']).astype(int)
        d['human_rt'] = d['response_time'] / 1000.0
        return d

    young = enrich(young)
    old = enrich(old)

    def layout_accuracy(df):
        return df.groupby('stimulus_layout')['correct'].mean().mul(100)

    def layout_cong_gap(df):
        rows = []
        for layout, grp in df.groupby('stimulus_layout'):
            if (grp['congruency'] == 0).any() and (grp['congruency'] == 1).any():
                gap = grp.loc[grp['congruency'] == 1, 'human_rt'].mean() - grp.loc[grp['congruency'] == 0, 'human_rt'].mean()
                rows.append((layout, gap))
        return pd.Series({layout: gap for layout, gap in rows})

    return {
        'accuracy': {'20-29': layout_accuracy(young), '80-89': layout_accuracy(old)},
        'cong_gap': {'20-29': layout_cong_gap(young), '80-89': layout_cong_gap(old)},
        'means': {
            'accuracy': {'20-29': young['correct'].mean() * 100, '80-89': old['correct'].mean() * 100},
            'cong_gap': {
                '20-29': young.loc[young['congruency'] == 1, 'human_rt'].mean() - young.loc[young['congruency'] == 0, 'human_rt'].mean(),
                '80-89': old.loc[old['congruency'] == 1, 'human_rt'].mean() - old.loc[old['congruency'] == 0, 'human_rt'].mean(),
            },
        }
    }


def model_metric_summary():
    df = pd.read_csv('results/age_groups_response_supervision_interim/response_supervision_current_comparison.csv')
    return {
        'accuracy': {'20-29': float(df.loc[df['age_group'] == '20-29', 'model_accuracy'].iloc[0] * 100), '80-89': float(df.loc[df['age_group'] == '80-89', 'model_accuracy'].iloc[0] * 100)},
        'cong_gap': {'20-29': float(df.loc[df['age_group'] == '20-29', 'model_congruency_rt_gap'].iloc[0]), '80-89': float(df.loc[df['age_group'] == '80-89', 'model_congruency_rt_gap'].iloc[0])},
    }


def draw_panel(ax, means, distributions=None, ylabel='', title='', annotate_p=False):
    ages = ['20-29', '80-89']
    fill_colors = [BLUE_FILL, ORANGE_FILL]
    point_colors = [BLUE_POINT, ORANGE_POINT]
    vals = [means[a] for a in ages]
    x = np.arange(2)
    ax.bar(x, vals, color=fill_colors, edgecolor=point_colors, linewidth=1.2, alpha=0.9, width=0.65)
    if distributions is not None:
        for i, age in enumerate(ages):
            pts = distributions[age].dropna().to_numpy()
            if len(pts):
                jitter = np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else np.array([0.0])
                ax.scatter(np.full(len(pts), i) + jitter, pts, color=point_colors[i], s=20, alpha=0.55, edgecolors='white', linewidths=0.35)
        if annotate_p:
            p = stats.mannwhitneyu(distributions['20-29'].dropna(), distributions['80-89'].dropna(), alternative='two-sided').pvalue
            add_sig(ax, p, max(vals))
    ax.set_xticks(x)
    ax.set_xticklabels(ages)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def make_metric_figure(metric_key, ylabel, panel_label, output_name, human, model):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)
    fig.suptitle(f'Figure {panel_label}. Age-group comparison under response supervision', fontsize=13)
    axes[0].set_title('Human', fontsize=12)
    axes[1].set_title('Model', fontsize=12)
    draw_panel(axes[0], human['means'][metric_key], human[metric_key], ylabel=ylabel, title=f'a  {ylabel}', annotate_p=True)
    draw_panel(axes[1], model[metric_key], None, ylabel=ylabel, title=f'b  {ylabel}', annotate_p=False)
    fig.savefig(OUT_DIR / output_name, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_dir()
    human = human_metric_distributions()
    model = model_metric_summary()
    make_metric_figure('accuracy', 'Accuracy (%)', 'RS3A', 'figureRS3A_agegroup_accuracy_human_vs_model.png', human, model)
    make_metric_figure('cong_gap', 'Congruency RT gap (s)', 'RS3B', 'figureRS3B_agegroup_congruency_gap_human_vs_model.png', human, model)
    print(f'Saved age-group comparison figure to {OUT_DIR}')


if __name__ == '__main__':
    main()
