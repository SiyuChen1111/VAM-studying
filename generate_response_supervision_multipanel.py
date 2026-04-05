import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path('results/age_groups_response_supervision_interim')


def ensure_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_epoch_dynamics(log_path: Path, age_group: str):
    text = log_path.read_text(errors='replace')
    lines = []
    for line in text.splitlines():
        if f'[{age_group} scale 1/5]' in line and 'avg_loss=' in line:
            lines.append(line)
        if len(lines) >= 5:
            break
    epochs, losses, durations = [], [], []
    for line in lines:
        epoch = int(line.split('Epoch ')[1].split('/20')[0])
        loss = float(line.split('avg_loss=')[1].split()[0])
        duration = float(line.split('duration=')[1].split('s')[0])
        epochs.append(epoch)
        losses.append(loss)
        durations.append(duration)
    return {'epochs': epochs, 'losses': losses, 'durations': durations}


def make_figure(df: pd.DataFrame, dynamics: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    x = np.arange(len(df))
    width = 0.35

    axes[0, 0].bar(x - width / 2, df['human_accuracy'], width, label='Human', color='#A0A0A0')
    axes[0, 0].bar(x + width / 2, df['model_accuracy'], width, label='Model', color='#7B9BD1')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(df['age_group'])
    axes[0, 0].set_title('a  Accuracy (current partial-best)')
    axes[0, 0].legend(frameon=False)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    axes[0, 1].bar(x - width / 2, df['human_congruency_rt_gap'], width, label='Human', color='#A0A0A0')
    axes[0, 1].bar(x + width / 2, df['model_congruency_rt_gap'], width, label='Model', color='#E8A87C')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df['age_group'])
    axes[0, 1].set_title('b  Congruency RT gap (current partial-best)')
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    axes[1, 0].bar(x - width / 2, df['score'], width, label='Total score', color='#72B7B2')
    axes[1, 0].bar(x + width / 2, df['rt_score'], width, label='RT score', color='#F3C7A2')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(df['age_group'])
    axes[1, 0].set_title('c  Score summary (current partial-best)')
    axes[1, 0].legend(frameon=False)
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    ax_loss = axes[1, 1]
    ax_dur = ax_loss.twinx()
    colors = {'20-29': '#7B9BD1', '80-89': '#E8A87C'}
    for age, info in dynamics.items():
        epochs = np.array(info['epochs'])
        ax_loss.plot(epochs, info['losses'], marker='o', color=colors[age], label=f'{age} loss')
        ax_dur.plot(epochs, info['durations'], marker='s', linestyle='--', color=colors[age], alpha=0.6, label=f'{age} epoch sec')
    ax_loss.set_title('d  Training dynamics through first 5 epochs')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_dur.set_ylabel('Epoch duration (s)')
    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_dur.get_legend_handles_labels()
    ax_loss.legend(h1 + h2, l1 + l2, frameon=False, loc='upper right')
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)

    fig.suptitle('Figure RS2. Response-supervision current partial-best comparison')
    fig.savefig(OUT_DIR / 'figureRS2_response_supervision_multipanel.png', bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_dir()
    df = pd.read_csv(OUT_DIR / 'response_supervision_current_comparison.csv')
    dynamics = {
        '20-29': parse_epoch_dynamics(Path('logs/train_20_29_matched_response_supervision_safe2.log'), '20-29'),
        '80-89': parse_epoch_dynamics(Path('logs/train_80_89_response_supervision_safe2.log'), '80-89'),
    }
    df.to_csv(OUT_DIR / 'response_supervision_multipanel_summary.csv', index=False)
    make_figure(df, dynamics)
    print(f'Saved multipanel response-supervision outputs to {OUT_DIR}')


if __name__ == '__main__':
    main()
