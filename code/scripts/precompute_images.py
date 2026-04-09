"""
Precompute and save images for age groups.

This script:
1. Loads age group data
2. Generates all images and saves them to disk
3. Saves image paths to CSV
4. This avoids on-the-fly image generation during logits extraction
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from vgg_wongwang_lim_data import LIMDataset


def generate_images_for_age_group(age_group: str, data_dir: str, output_dir: str, image_size: int = 128):
    """
    Generate images for an age group and save them.
    
    Args:
        age_group: Age group string (e.g., '20-29')
        data_dir: Path to vam_data directory
        output_dir: Directory to save images
        image_size: Size for generated images
    """
    print(f"\n{'='*60}")
    print(f"Generating images for age group: {age_group}")
    print(f"{'='*60}")
    
    # Load metadata
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    age_users = metadata[metadata['binned_age'] == age_group]['user_id'].tolist()
    print(f"Users in {age_group}: {len(age_users)}")
    
    # Create dataset to generate images
    dataset = LIMDataset(
        data_dir=data_dir,
        graphics_dir='vam',
        users=age_users,
        split='train',
        image_size=image_size,
        train_ratio=0.8,
        precompute_images=True
    )
    
    # Create output directory
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Generate images and save paths
    print(f"Generating {len(dataset)} images...")
    image_paths = []
    
    for idx in tqdm(range(len(dataset)), desc="Generating images"):
        sample = dataset[idx]
        
        # Save image
        image_filename = f"image_{idx:06d}.png"
        image_path = os.path.join(images_output_dir, image_filename)
        
        # Convert tensor to PIL Image
        image = sample['image']
        image_np = image.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_pil.save(image_path)
        
        image_paths.append(image_path)
    
    # Update CSV with image paths
    df = pd.read_csv(os.path.join(data_dir, f'{age_group}/train_data.csv'))
    df['image_path'] = image_paths[:len(df)]
    df.to_csv(os.path.join(data_dir, f'{age_group}/train_data.csv'), index=False)
    
    # Do the same for test data
    print(f"\nGenerating test images...")
    test_dataset = LIMDataset(
        data_dir=data_dir,
        graphics_dir='vam',
        users=age_users,
        split='test',
        image_size=image_size,
        train_ratio=0.8,
        precompute_images=True
    )
    
    test_image_paths = []
    for idx in tqdm(range(len(test_dataset)), desc="Generating test images"):
        image, rt, rt_norm, label, response, congruent = test_dataset[idx]
        
        image_filename = f"image_{idx:06d}.png"
        image_path = os.path.join(images_output_dir, image_filename)
        
        image_np = image.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_pil.save(image_path)
        
        test_image_paths.append(image_path)
    
    # Update test CSV
    test_df = pd.read_csv(os.path.join(data_dir, f'{age_group}/test_data.csv'))
    test_df['image_path'] = test_image_paths[:len(test_df)]
    test_df.to_csv(os.path.join(data_dir, f'{age_group}/test_data.csv'), index=False)
    
    print(f"\nCompleted!")
    print(f"Train images: {len(image_paths)}")
    print(f"Test images: {len(test_image_paths)}")
    print(f"Images saved to {images_output_dir}")
    
    return len(image_paths), len(test_image_paths)


def main():
    age_groups = ['20-29', '80-89']
    
    for age_group in age_groups:
        data_dir = f'data_age_groups/{age_group}'
        output_dir = f'data_age_groups/{age_group}'
        
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            continue
        
        generate_images_for_age_group(age_group, 'vam_data', output_dir)


if __name__ == "__main__":
    main()
