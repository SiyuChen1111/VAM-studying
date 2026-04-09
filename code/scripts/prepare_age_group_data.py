"""
Prepare age-group specific data for model training.

This script:
1. Loads all user data
2. Filters by age group
3. Splits into train/test sets
4. Saves to separate directories
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from project_paths import DATA_AGE_GROUPS_ROOT, VAM_DATA_ROOT

def prepare_age_group_data(age_group: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare data for a specific age group.
    
    Args:
        age_group: Age group string (e.g., '20-29', '80-89')
        output_dir: Output directory for the prepared data
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    print(f"Preparing data for age group: {age_group}")
    
    # Load metadata
    metadata = pd.read_csv(VAM_DATA_ROOT / 'metadata.csv')
    
    # Get users in this age group
    age_users = metadata[metadata['binned_age'] == age_group]['user_id'].tolist()
    print(f"  Users in age group: {len(age_users)}")
    
    # Load data for these users
    data_dir = str(VAM_DATA_ROOT)
    all_data = []
    
    for user_id in age_users:
        user_file = os.path.join(data_dir, f'user{user_id}df.csv')
        if os.path.exists(user_file):
            df = pd.read_csv(user_file)
            df['user_id'] = user_id
            all_data.append(df)
    
    if not all_data:
        print(f"  No data found for age group {age_group}")
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"  Total trials: {len(df)}")
    
    # Calculate RT statistics for this age group
    rt_s = df['response_time'] / 1000
    rt_stats = {
        'mean': rt_s.mean(),
        'median': rt_s.median(),
        'std': rt_s.std(),
        'min': rt_s.min(),
        'max': rt_s.max(),
        'skewness': float(pd.Series(rt_s).skew()),
        'percentile_95': np.percentile(rt_s, 95),
        'percentile_99': np.percentile(rt_s, 99)
    }
    
    print(f"  RT statistics:")
    print(f"    Mean: {rt_stats['mean']:.3f}s")
    print(f"    Median: {rt_stats['median']:.3f}s")
    print(f"    Skewness: {rt_stats['skewness']:.3f}")
    print(f"    95th percentile: {rt_stats['percentile_95']:.3f}s")
    
    # Split users into train/test
    train_users, test_users = train_test_split(
        age_users, test_size=test_size, random_state=random_state
    )
    
    train_df = df[df['user_id'].isin(train_users)]
    test_df = df[df['user_id'].isin(test_users)]
    
    print(f"  Train trials: {len(train_df)}")
    print(f"  Test trials: {len(test_df)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    
    # Save statistics
    import json
    with open(os.path.join(output_dir, 'rt_stats.json'), 'w') as f:
        json.dump(rt_stats, f, indent=2)
    
    print(f"  Data saved to {output_dir}")
    
    return rt_stats

def main():
    # Prepare data for 20-29 age group
    print("="*50)
    print("Preparing data for 20-29 age group")
    print("="*50)
    young_stats = prepare_age_group_data('20-29', str(DATA_AGE_GROUPS_ROOT / '20-29'))
    
    # Prepare data for 80-89 age group
    print("\n" + "="*50)
    print("Preparing data for 80-89 age group")
    print("="*50)
    old_stats = prepare_age_group_data('80-89', str(DATA_AGE_GROUPS_ROOT / '80-89'))
    
    # Compare
    if young_stats and old_stats:
        print("\n" + "="*50)
        print("Comparison")
        print("="*50)
        print(f"20-29 Mean RT: {young_stats['mean']:.3f}s")
        print(f"80-89 Mean RT: {old_stats['mean']:.3f}s")
        print(f"Difference: {old_stats['mean'] - young_stats['mean']:.3f}s ({(old_stats['mean'] - young_stats['mean'])/young_stats['mean']*100:.1f}%)")
        
        # Suggest time_steps
        print("\nSuggested time_steps (dt=10ms):")
        for age, stats in [('20-29', young_stats), ('80-89', old_stats)]:
            suggested = int(np.ceil(stats['percentile_99'] * 100))
            print(f"  {age}: {suggested} (max {suggested*10/1000:.2f}s)")

if __name__ == "__main__":
    main()
