"""
Extract logits for age groups using existing Stage 1 model.

This script:
1. Uses LIMDataset to load age group data with image generation
2. Extracts logits using Stage 1 model
3. Saves logits for Stage 2 training
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vgg_wongwang_lim import VGGFeatureExtractor
from vgg_wongwang_lim_data import LIMDataset


def extract_logits_for_age_group(
    age_group: str,
    stage1_model_path: str,
    data_dir: str,
    graphics_dir: str,
    output_dir: str,
    batch_size: int = 64,
    device: str = 'cpu'
):
    """
    Extract logits for an age group using Stage 1 model.
    
    Args:
        age_group: Age group string (e.g., '20-29')
        stage1_model_path: Path to Stage 1 model checkpoint
        data_dir: Path to vam_data directory
        graphics_dir: Path to graphics directory
        output_dir: Output directory for logits
        batch_size: Batch size for inference
        device: Device to use
    """
    print(f"\n{'='*60}")
    print(f"Extracting logits for age group: {age_group}")
    print(f"{'='*60}")
    
    # Load Stage 1 model
    print("Loading Stage 1 model...")
    model = VGGFeatureExtractor(pretrained=False, n_classes=4)
    
    checkpoint = torch.load(stage1_model_path, map_location=device, weights_only=False)
    
    # The checkpoint keys are already in the correct format (features.0.weight, etc.)
    # Just load directly
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully. Loaded {len(checkpoint['model_state_dict'])} parameters.")
    
    # Load age group user IDs
    import pandas as pd
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    age_users = metadata[metadata['binned_age'] == age_group]['user_id'].tolist()
    print(f"Users in {age_group}: {len(age_users)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = LIMDataset(
        data_dir=data_dir,
        graphics_dir=graphics_dir,
        users=age_users,
        split='train',
        image_size=128,
        train_ratio=0.8,
        precompute_images=False
    )
    
    test_dataset = LIMDataset(
        data_dir=data_dir,
        graphics_dir=graphics_dir,
        users=age_users,
        split='test',
        image_size=128,
        train_ratio=0.8,
        precompute_images=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract logits
    def extract_logits(dataloader, desc):
        all_logits = []
        all_rts = []
        all_rts_norm = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                images = batch['image'].to(device)
                logits = model(images)
                
                all_logits.append(logits.cpu().numpy())
                all_rts.extend(batch['rt_original'].numpy())
                all_rts_norm.extend(batch['rt_normalized'].numpy())
        
        return (
            np.concatenate(all_logits, axis=0),
            np.array(all_rts),
            np.array(all_rts_norm)
        )
    
    print("\nExtracting train logits...")
    train_logits, train_rts, train_rts_norm = extract_logits(train_loader, "Train")
    
    print("Extracting test logits...")
    test_logits, test_rts, test_rts_norm = extract_logits(test_loader, "Test")
    
    # Save logits
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez(
        os.path.join(output_dir, 'train_logits.npz'),
        logits=train_logits,
        rts=train_rts,
        rts_normalized=train_rts_norm
    )
    
    np.savez(
        os.path.join(output_dir, 'test_logits.npz'),
        logits=test_logits,
        rts=test_rts,
        rts_normalized=test_rts_norm
    )
    
    # Save normalization params
    np.savez(
        os.path.join(output_dir, 'rt_normalization_params.npz'),
        log_rt_min=train_dataset.log_rt_min,
        log_rt_max=train_dataset.log_rt_max,
        log_rt_range=train_dataset.log_rt_range,
        rt_min=np.exp(train_dataset.log_rt_min),
        rt_max=np.exp(train_dataset.log_rt_max)
    )
    
    # Print statistics
    print(f"\nLogits statistics:")
    print(f"  Train: {train_logits.shape}, RT mean={train_rts.mean():.3f}s")
    print(f"  Test: {test_logits.shape}, RT mean={test_rts.mean():.3f}s")
    print(f"  Logits mean: {train_logits.mean():.4f}, std: {train_logits.std():.4f}")
    
    print(f"\nSaved to {output_dir}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    stage1_model_path = 'checkpoints_test/stage1/best_model.pth'
    if not os.path.exists(stage1_model_path):
        print(f"Stage 1 model not found at {stage1_model_path}")
        return
    
    age_groups = ['20-29', '80-89']
    
    for age_group in age_groups:
        output_dir = f'checkpoints_age_groups/{age_group}/stage1'
        
        extract_logits_for_age_group(
            age_group=age_group,
            stage1_model_path=stage1_model_path,
            data_dir='vam_data',
            graphics_dir='vam',
            output_dir=output_dir,
            batch_size=64,
            device=device
        )


if __name__ == "__main__":
    main()
