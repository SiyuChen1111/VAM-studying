"""
Train Wong-Wang model for age groups with automatic scale search.

This script:
1. Loads age-group specific data
2. Trains Wong-Wang model with different scale values
3. Evaluates RT distribution and congruency effect
4. Finds optimal scale that matches human signatures
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from vgg_wongwang_lim import VGGWongWangLIM


class AgeGroupDataset(Dataset):
    """Dataset for age group data."""
    
    def __init__(self, data_path: str, stage1_logits_dir: str, split: str = 'train'):
        """
        Args:
            data_path: Path to age group data directory
            stage1_logits_dir: Path to Stage 1 logits directory
            split: 'train' or 'test'
        """
        self.data = pd.read_csv(os.path.join(data_path, f'{split}_data.csv'))
        
        # Load Stage 1 logits
        logits_data = np.load(os.path.join(stage1_logits_dir, f'{split}_logits.npz'))
        self.logits = logits_data['logits']
        self.rts = logits_data['rts']
        self.rts_normalized = logits_data['rts_normalized']
        
        # Load normalization params
        norm_params = np.load(os.path.join(stage1_logits_dir, 'rt_normalization_params.npz'))
        self.log_rt_min = float(norm_params['log_rt_min'])
        self.log_rt_max = float(norm_params['log_rt_max'])
        self.log_rt_range = float(norm_params['log_rt_range'])
        
        print(f"Loaded {len(self.logits)} samples for {split}")
    
    def __len__(self):
        return len(self.logits)
    
    def __getitem__(self, idx):
        return {
            'logits': torch.tensor(self.logits[idx], dtype=torch.float32),
            'rt': torch.tensor(self.rts[idx], dtype=torch.float32),
            'rt_normalized': torch.tensor(self.rts_normalized[idx], dtype=torch.float32)
        }


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> Dict:
    """Evaluate model and return RT predictions and statistics."""
    model.eval()
    
    all_pred_rt = []
    all_true_rt = []
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            logits = batch['logits'].to(device)
            rt = batch['rt'].to(device)
            
            # Forward pass
            _, _, pred_rt = model(logits)
            
            all_pred_rt.extend(pred_rt.cpu().numpy())
            all_true_rt.extend(rt.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    all_pred_rt = np.array(all_pred_rt)
    all_true_rt = np.array(all_true_rt)
    
    # Calculate statistics
    results = {
        'pred_rt': all_pred_rt,
        'true_rt': all_true_rt,
        'pred_mean': all_pred_rt.mean(),
        'pred_median': np.median(all_pred_rt),
        'pred_std': all_pred_rt.std(),
        'pred_skewness': stats.skew(all_pred_rt),
        'true_mean': all_true_rt.mean(),
        'true_median': np.median(all_true_rt),
        'true_std': all_true_rt.std(),
        'true_skewness': stats.skew(all_true_rt),
        'correlation': np.corrcoef(all_pred_rt, all_true_rt)[0, 1]
    }
    
    return results


def calculate_signature_score(results: Dict, human_stats: Dict) -> float:
    """
    Calculate a score based on how well the model matches human signatures.
    
    Criteria:
    1. RT distribution skewness > 0.5 (right-skewed)
    2. Predicted RT peak close to human RT peak
    3. RT distribution width similar to human
    
    Returns:
        score: Higher is better (0-1 range)
    """
    score = 0.0
    
    # 1. Skewness score (right-skewed is good)
    skewness = results['pred_skewness']
    if skewness > 0.5:
        skewness_score = min(skewness / human_stats['skewness'], 1.0)
    else:
        skewness_score = 0.0
    score += 0.3 * skewness_score
    
    # 2. Peak position score (close to human peak)
    pred_peak = results['pred_median']
    human_peak = human_stats['median']
    peak_diff = abs(pred_peak - human_peak) / human_peak
    peak_score = max(0, 1 - peak_diff)
    score += 0.4 * peak_score
    
    # 3. Mean RT score (close to human mean)
    pred_mean = results['pred_mean']
    human_mean = human_stats['mean']
    mean_diff = abs(pred_mean - human_mean) / human_mean
    mean_score = max(0, 1 - mean_diff)
    score += 0.3 * mean_score
    
    return score


def train_with_scale(
    age_group: str,
    scale: float,
    time_steps: int,
    data_dir: str,
    stage1_logits_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-4,
    dt: int = 10,
    device: str = 'cpu'
) -> Tuple[Dict, float]:
    """
    Train model with a specific scale value.
    
    Returns:
        results: Evaluation results
        signature_score: Score based on human signature matching
    """
    print(f"\nTraining with scale={scale}, time_steps={time_steps}")
    
    # Load human stats
    with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
        human_stats = json.load(f)
    
    # Create model
    model = VGGWongWangLIM(n_classes=4, dt=dt, time_steps=time_steps)
    
    # Set scale
    model.ww_wrapper.scale = torch.tensor(scale)
    
    model = model.to(device)
    
    # Create dataloaders
    train_dataset = AgeGroupDataset(data_dir, stage1_logits_dir, 'train')
    test_dataset = AgeGroupDataset(data_dir, stage1_logits_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    best_score = 0.0
    best_results = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            logits = batch['logits'].to(device)
            rt = batch['rt_normalized'].to(device)
            
            optimizer.zero_grad()
            _, _, pred_rt = model(logits)
            
            # Loss: MSE on normalized RT
            loss = criterion(pred_rt, rt)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            results = evaluate_model(model, test_loader, device)
            score = calculate_signature_score(results, human_stats)
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Score={score:.4f}, Corr={results['correlation']:.4f}, "
                  f"PredMean={results['pred_mean']:.3f}s, HumanMean={human_stats['mean']:.3f}s")
            
            if score > best_score:
                best_score = score
                best_results = results.copy()
                
                # Save best model
                os.makedirs(output_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'scale': scale,
                    'time_steps': time_steps,
                    'score': score,
                    'results': {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
                }, os.path.join(output_dir, 'best_model.pth'))
    
    return best_results, best_score


def search_optimal_scale(
    age_group: str,
    data_dir: str,
    stage1_logits_dir: str,
    output_dir: str,
    scale_range: Tuple[float, float] = (0.1, 0.5),
    n_scales: int = 5,
    epochs: int = 20,
    device: str = 'cpu'
) -> Dict:
    """
    Search for optimal scale value.
    
    Args:
        age_group: Age group string
        data_dir: Path to age group data
        stage1_logits_dir: Path to Stage 1 logits
        output_dir: Output directory for models
        scale_range: (min_scale, max_scale)
        n_scales: Number of scale values to try
        epochs: Training epochs per scale
        device: Device to use
    
    Returns:
        best_config: Best configuration found
    """
    # Load human stats
    with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
        human_stats = json.load(f)
    
    # Determine time_steps based on human RT
    time_steps = int(np.ceil(human_stats['percentile_99'] * 100))
    print(f"Human RT stats for {age_group}:")
    print(f"  Mean: {human_stats['mean']:.3f}s")
    print(f"  Median: {human_stats['median']:.3f}s")
    print(f"  Skewness: {human_stats['skewness']:.3f}")
    print(f"  99th percentile: {human_stats['percentile_99']:.3f}s")
    print(f"  Using time_steps={time_steps} (max {time_steps*10/1000:.2f}s)")
    
    # Search scales
    scales = np.linspace(scale_range[0], scale_range[1], n_scales)
    results_list = []
    
    for scale in scales:
        scale_output_dir = os.path.join(output_dir, f'scale_{scale:.3f}')
        results, score = train_with_scale(
            age_group=age_group,
            scale=scale,
            time_steps=time_steps,
            data_dir=data_dir,
            stage1_logits_dir=stage1_logits_dir,
            output_dir=scale_output_dir,
            epochs=epochs,
            device=device
        )
        
        results_list.append({
            'scale': scale,
            'score': score,
            'results': results
        })
    
    # Find best scale
    best_idx = np.argmax([r['score'] for r in results_list])
    best_config = results_list[best_idx]
    
    print(f"\n{'='*50}")
    print(f"Best scale for {age_group}: {best_config['scale']:.3f}")
    print(f"Score: {best_config['score']:.4f}")
    print(f"Pred Mean RT: {best_config['results']['pred_mean']:.3f}s")
    print(f"Human Mean RT: {human_stats['mean']:.3f}s")
    print(f"{'='*50}")
    
    # Save results
    with open(os.path.join(output_dir, 'scale_search_results.json'), 'w') as f:
        json.dump({
            'age_group': age_group,
            'time_steps': time_steps,
            'human_stats': human_stats,
            'best_scale': best_config['scale'],
            'best_score': best_config['score'],
            'all_results': [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in results_list]
        }, f, indent=2)
    
    return best_config


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Age groups to train
    age_groups = ['20-29', '80-89']
    
    for age_group in age_groups:
        print(f"\n{'='*60}")
        print(f"Training for age group: {age_group}")
        print(f"{'='*60}")
        
        data_dir = f'data_age_groups/{age_group}'
        stage1_logits_dir = f'checkpoints_age_groups/{age_group}/stage1'
        output_dir = f'checkpoints_age_groups/{age_group}/stage2'
        
        # Check if Stage 1 logits exist
        if not os.path.exists(stage1_logits_dir):
            print(f"Stage 1 logits not found for {age_group}. Please run Stage 1 first.")
            continue
        
        # Search for optimal scale
        best_config = search_optimal_scale(
            age_group=age_group,
            data_dir=data_dir,
            stage1_logits_dir=stage1_logits_dir,
            output_dir=output_dir,
            scale_range=(0.1, 0.5),
            n_scales=5,
            epochs=20,
            device=device
        )


if __name__ == "__main__":
    main()
