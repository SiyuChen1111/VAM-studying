from pathlib import Path

import numpy as np
import pandas as pd


MATCHED_DIR = Path('data_age_groups_matched/20-29')
SOURCE_DIR = Path('data_age_groups/20-29')
OUT_DIR = Path('checkpoints_age_groups_matched/20-29/stage2')


def subset_split(split: str):
    full_df = pd.read_csv(SOURCE_DIR / f'{split}_data.csv')
    matched_df = pd.read_csv(MATCHED_DIR / f'{split}_data.csv')
    full_npz = np.load(Path('checkpoints_age_groups/20-29/stage2') / f'{split}_logits.npz')

    key_cols = [
        'user_id', 'anon_id', 'nth_play', 'trial', 'xpos', 'ypos',
        'flanker_direction', 'response_direction', 'response_time',
        'stimulus_layout', 'target_direction', 'stimulus_image_idx'
    ]

    full_keys = full_df[key_cols].astype(str).agg('||'.join, axis=1)
    matched_keys = matched_df[key_cols].astype(str).agg('||'.join, axis=1)
    full_index = {key: idx for idx, key in enumerate(full_keys)}
    indices = [full_index[key] for key in matched_keys]

    subset = {key: full_npz[key][indices] for key in full_npz.files}
    return subset


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'test']:
        subset = subset_split(split)
        np.savez_compressed(OUT_DIR / f'{split}_logits.npz', **subset)
        print(split, {k: v.shape for k, v in subset.items()})


if __name__ == '__main__':
    main()
