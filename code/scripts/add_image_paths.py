"""
Add image paths to age group data CSVs.

This script adds image_path column to train_data.csv and test_data.csv
based on the images already generated in the images/ directory.
"""

import os
import pandas as pd
from tqdm import tqdm


def add_image_paths(age_group: str, data_dir: str):
    """
    Add image paths to CSV files.
    
    Args:
        age_group: Age group string (e.g., '20-29')
        data_dir: Path to age group data directory
    """
    print(f"\nAdding image paths for age group: {age_group}")
    
    images_dir = os.path.join(data_dir, 'images')
    
    if not os.path.exists(images_dir):
        print(f"  Images directory not found: {images_dir}")
        return
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    print(f"  Found {len(image_files)} images")
    
    # Process train data
    train_csv = os.path.join(data_dir, 'train_data.csv')
    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)
        print(f"  Train data: {len(train_df)} samples")
        
        # Add image paths
        train_df['image_path'] = [os.path.join(images_dir, f'image_{i:06d}.png') for i in range(len(train_df))]
        
        # Save updated CSV
        train_df.to_csv(train_csv, index=False)
        print(f"  Updated {train_csv}")
    
    # Process test data
    test_csv = os.path.join(data_dir, 'test_data.csv')
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        print(f"  Test data: {len(test_df)} samples")
        
        # Add image paths (continuing from train images)
        start_idx = len(image_files) - len(test_df)
        test_df['image_path'] = [os.path.join(images_dir, f'image_{i:06d}.png') for i in range(start_idx, start_idx + len(test_df))]
        
        # Save updated CSV
        test_df.to_csv(test_csv, index=False)
        print(f"  Updated {test_csv}")


def main():
    age_groups = ['20-29', '80-89']
    
    for age_group in age_groups:
        data_dir = f'data_age_groups/{age_group}'
        
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            continue
        
        add_image_paths(age_group, data_dir)


if __name__ == "__main__":
    main()
