"""
Create proper stimulus mapping based on stimulus_layout, target_direction, and flanker_direction.

This script:
1. Creates unique stimulus images for each (layout, target, flanker) combination
2. Maps each trial to the correct stimulus image based on its stimulus_layout, target_direction, and flanker_direction
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import shutil

from vgg_wongwang_lim_data import (
    DIRECTION_MAP,
    LAYOUT_SPACERS,
    CANVAS_SIZE,
    get_distractor_positions,
    create_stimulus_image,
)


def load_graphics(graphics_dir: str = 'vam'):
    background = Image.open(os.path.join(graphics_dir, 'bkgrnd.png')).convert('RGB')
    birds = {}
    for direction, idx in DIRECTION_MAP.items():
        birds[direction] = Image.open(os.path.join(graphics_dir, f'bird{idx}.png')).convert('RGBA')
    return background, birds


def generate_from_row(row: pd.Series, background: Image.Image, birds: dict):
    spacer = [float(v) for v in LAYOUT_SPACERS[int(row['stimulus_layout'])]]
    target_pos = (
        float(row['xpos']) - CANVAS_SIZE[0] / 2,
        -float(row['ypos']) + CANVAS_SIZE[1] / 2,
    )
    distractor_positions = get_distractor_positions(target_pos, int(row['stimulus_layout']), spacer)
    target_bird = birds[row['target_direction']]
    distractor_bird = birds[row['flanker_direction']]
    return create_stimulus_image(background, target_bird, distractor_bird, target_pos, distractor_positions)


def create_stimulus_mapping(age_group: str, data_dir: str):
    """
    Create stimulus mapping for an age group.
    
    Args:
        age_group: Age group string (e.g., '20-29')
        data_dir: Path to age group data directory
    """
    print(f"\n{'='*60}")
    print(f"Creating stimulus mapping for age group: {age_group}")
    print(f"{'='*60}")
    
    # Load train data
    train_csv = os.path.join(data_dir, 'train_data.csv')
    if not os.path.exists(train_csv):
        print(f"Train data not found: {train_csv}")
        return
    
    train_df = pd.read_csv(train_csv)
    print(f"Train data: {len(train_df)} samples")
    
    # Get unique combinations
    combinations = train_df.groupby(['stimulus_layout', 'target_direction', 'flanker_direction']).size().reset_index()
    print(f"Unique combinations: {len(combinations)}")
    
    mapping = {}
    for idx, row in combinations.iterrows():
        key = (row['stimulus_layout'], row['target_direction'], row['flanker_direction'])
        mapping[key] = idx
    
    # Save mapping
    mapping_df = pd.DataFrame([
        {'stimulus_layout': k[0], 'target_direction': k[1], 'flanker_direction': k[2], 'image_index': v}
        for k, v in mapping.items()
    ])
    mapping_df.to_csv(os.path.join(data_dir, 'stimulus_mapping.csv'), index=False)
    print(f"Saved mapping to {os.path.join(data_dir, 'stimulus_mapping.csv')}")
    
    # Create stimulus_images directory
    stim_images_dir = os.path.join(data_dir, 'stimulus_images')
    os.makedirs(stim_images_dir, exist_ok=True)
    background, birds = load_graphics()
    
    existing_images_dir = os.path.join(data_dir, 'images')

    if os.path.exists(existing_images_dir):
        print(f"Copying stimulus images from {existing_images_dir}")
        use_image_path_column = False
        use_shared_stimulus_dir = False
    elif 'image_path' in train_df.columns:
        print("Existing images directory not found; using image_path column as source")
        use_image_path_column = True
        use_shared_stimulus_dir = False
    elif age_group != '20-29' and os.path.exists('data_age_groups/20-29/stimulus_images'):
        print("Existing images directory not found; using shared 20-29 stimulus_images directory")
        use_image_path_column = False
        use_shared_stimulus_dir = True
    else:
        print(f"Existing images directory not found: {existing_images_dir}")
        print("No image_path column available; please run image generation first")
        return

    for key, image_idx in mapping.items():
        mask = (
            (train_df['stimulus_layout'] == key[0])
            & (train_df['target_direction'] == key[1])
            & (train_df['flanker_direction'] == key[2])
        )
        first_trial = train_df[mask].iloc[0]

        if use_image_path_column:
            src_path = first_trial['image_path']
        elif use_shared_stimulus_dir:
            src_path = os.path.join('data_age_groups/20-29/stimulus_images', f'stimulus_{image_idx:03d}.png')
        else:
            first_trial_idx = train_df[mask].index[0]
            src_path = os.path.join(existing_images_dir, f'image_{first_trial_idx:06d}.png')

        dst_path = os.path.join(stim_images_dir, f'stimulus_{image_idx:03d}.png')

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            generated = generate_from_row(first_trial, background, birds)
            generated.save(dst_path)

    print(f"Saved {len(mapping)} stimulus images to {stim_images_dir}")
    
    # Update train data with stimulus image index
    train_df['stimulus_image_idx'] = train_df.apply(
        lambda row: mapping[(row['stimulus_layout'], row['target_direction'], row['flanker_direction'])], axis=1
    )
    train_df['stimulus_image_path'] = train_df['stimulus_image_idx'].apply(
        lambda idx: os.path.join(stim_images_dir, f'stimulus_{idx:03d}.png')
    )
    
    train_df.to_csv(train_csv, index=False)
    print(f"Updated {train_csv}")
    
    # Update test data
    test_csv = os.path.join(data_dir, 'test_data.csv')
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        print(f"Test data: {len(test_df)} samples")
        
        test_df['stimulus_image_idx'] = test_df.apply(
            lambda row: mapping[(row['stimulus_layout'], row['target_direction'], row['flanker_direction'])], axis=1
        )
        test_df['stimulus_image_path'] = test_df['stimulus_image_idx'].apply(
            lambda idx: os.path.join(stim_images_dir, f'stimulus_{idx:03d}.png')
        )
        
        test_df.to_csv(test_csv, index=False)
        print(f"Updated {test_csv}")
    
    return mapping


def main():
    age_groups = ['20-29', '80-89']
    
    for age_group in age_groups:
        data_dir = f'data_age_groups/{age_group}'
        
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            continue
        
        create_stimulus_mapping(age_group, data_dir)


if __name__ == "__main__":
    main()
