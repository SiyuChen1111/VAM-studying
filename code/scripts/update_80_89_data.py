"""
Update 80-89 age group to use the same 28 stimulus images as 20-29 group.
"""

import os
import pandas as pd


def update_80_89_data():
    """Update 80-89 age group data to use shared stimulus images."""
    
    data_dir = 'data_age_groups/80-89'
    stim_images_dir = 'data_age_groups/20-29/stimulus_images'
    
    print("Updating 80-89 age group data...")
    
    # Load train data
    train_csv = os.path.join(data_dir, 'train_data.csv')
    train_df = pd.read_csv(train_csv)
    print(f"Train data: {len(train_df)} samples")
    
    # Create mapping from stimulus_layout and flanker_direction to image index
    dir_map = {'L': 0, 'R': 1, 'U': 2, 'D': 3}
    
    # Calculate stimulus image index
    train_df['stimulus_image_idx'] = train_df.apply(
        lambda row: row['stimulus_layout'] * 4 + dir_map[row['flanker_direction']], 
        axis=1
    )
    train_df['stimulus_image_path'] = train_df['stimulus_image_idx'].apply(
        lambda idx: os.path.join(stim_images_dir, f'stimulus_{idx:02d}.png')
    )
    
    train_df.to_csv(train_csv, index=False)
    print(f"Updated {train_csv}")
    
    # Load test data
    test_csv = os.path.join(data_dir, 'test_data.csv')
    test_df = pd.read_csv(test_csv)
    print(f"Test data: {len(test_df)} samples")
    
    # Calculate stimulus image index
    test_df['stimulus_image_idx'] = test_df.apply(
        lambda row: row['stimulus_layout'] * 4 + dir_map[row['flanker_direction']], 
        axis=1
    )
    test_df['stimulus_image_path'] = test_df['stimulus_image_idx'].apply(
        lambda idx: os.path.join(stim_images_dir, f'stimulus_{idx:02d}.png')
    )
    
    test_df.to_csv(test_csv, index=False)
    print(f"Updated {test_csv}")
    
    # Remove empty stimulus_images directory
    empty_dir = os.path.join(data_dir, 'stimulus_images')
    if os.path.exists(empty_dir) and not os.listdir(empty_dir):
        os.rmdir(empty_dir)
        print(f"Removed empty directory: {empty_dir}")
    
    print("\nDone! 80-89 age group now uses shared stimulus images.")


if __name__ == "__main__":
    update_80_89_data()
