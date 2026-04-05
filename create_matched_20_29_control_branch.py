import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_USERS = [182, 899, 1478]
TEST_USERS = [677]
SOURCE_DIR = Path('data_age_groups/20-29')
TARGET_DIR = Path('data_age_groups_matched/20-29')


def compute_rt_stats(df: pd.DataFrame):
    rt_s = df['response_time'] / 1000.0
    return {
        'mean': float(rt_s.mean()),
        'median': float(rt_s.median()),
        'std': float(rt_s.std()),
        'min': float(rt_s.min()),
        'max': float(rt_s.max()),
        'skewness': float(pd.Series(rt_s).skew()),
        'percentile_95': float(np.percentile(rt_s, 95)),
        'percentile_99': float(np.percentile(rt_s, 99)),
    }


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(SOURCE_DIR / 'train_data.csv')
    test_df = pd.read_csv(SOURCE_DIR / 'test_data.csv')

    matched_train = train_df[train_df['user_id'].isin(TRAIN_USERS)].copy()
    matched_test = test_df[test_df['user_id'].isin(TEST_USERS)].copy()

    matched_train.to_csv(TARGET_DIR / 'train_data.csv', index=False)
    matched_test.to_csv(TARGET_DIR / 'test_data.csv', index=False)

    rt_stats = compute_rt_stats(pd.concat([matched_train, matched_test], ignore_index=True))
    with open(TARGET_DIR / 'rt_stats.json', 'w') as f:
        json.dump(rt_stats, f, indent=2)

    metadata = {
        'source_age_group': '20-29',
        'branch_type': 'subject-count-matched-control',
        'matched_against': '80-89',
        'train_users': TRAIN_USERS,
        'test_users': TEST_USERS,
        'train_rows': int(len(matched_train)),
        'test_rows': int(len(matched_test)),
        'notes': [
            'Selected to approximate 80-89 subject-count structure (3 train users, 1 test user).',
            'Selected to minimize row-count difference relative to current 80-89 train/test splits.',
            'This branch is a control analysis and does not replace the full-data 20-29 main path.',
        ],
    }
    with open(TARGET_DIR / 'matched_branch_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print('Created matched 20-29 control branch:')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
