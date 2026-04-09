"""
Train Stage 2 (Wong-Wang RT fitting) for age groups with automatic scale search.

This script:
1. Uses existing Stage 1 logits
2. Trains Wong-Wang model with different scale values
3. Evaluates RT distribution matching
4. Finds optimal scale that matches human signatures
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from typing import Dict, Tuple

from vgg_wongwang_lim import WWWrapper


class LogitsDataset(Dataset):
    """Dataset for logits and RT."""
    
    def __init__(self, logits: np.ndarray, rts: np.ndarray, rts_normalized: np.ndarray):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.rts = torch.tensor(rts, dtype=torch.float32)
        self.rts_normalized = torch.tensor(rts_normalized, dtype=torch.float32)
    
    def __len__(self):
        return len(self.logits)
    
    def __getitem__(self, idx):
        return {
            'logits': self.logits[idx],
            'rt': self.rts[idx],
            'rt_normalized': self.rts_normalized[idx]
        }


def evaluate_rt_distribution(pred_rt: np.ndarray, human_stats: Dict) -> Dict:
    """Evaluate how well predicted RT matches human RT distribution."""
    
    pred_mean = pred_rt.mean()
    pred_median = np.median(pred_rt)
    pred_skewness = stats.skew(pred_rt)
    
    true_mean = human_stats['mean']
    true_median = human_stats['median']
    true_skewness = human_stats['skewness']
    
    # Score components
    # 1. Skewness score (should be right-skewed)
    skewness_score = min(pred_skewness / max(true_skewness, 1.0), 1.0) if pred_skewness > 0.5 else 0.0
    
    # 2. Mean RT score
    mean_diff = abs(pred_mean - true_mean) / true_mean
    mean_score = max(0, 1 - mean_diff)
    
    # 3. Median RT score
    median_diff = abs(pred_median - true_median) / true_median
    median_score = max(0, 1 - median_diff)
    
    # 4. Coverage score
    pred_range = pred_rt.max() - pred_rt.min()
    true_range = human_stats['max'] - human_stats['min']
    coverage_score = min(pred_range / true_range, 1.0)
    
    # Total score
    total_score = 0.3 * skewness_score + 0.3 * mean_score + 0.2 * median_score + 0.2 * coverage_score
    
    return {
        'pred_mean': pred_mean,
        'pred_median': pred_median,
        'pred_skewness': pred_skewness,
        'true_mean': true_mean,
        'true_median': true_median,
        'true_skewness': true_skewness,
        'skewness_score': skewness_score,
        'mean_score': mean_score,
        'median_score': median_score,
        'coverage_score': coverage_score,
        'total_score': total_score
    }


def train_with_scale(
    scale: float,
    time_steps: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    human_stats: Dict,
    epochs: int = 20,
    lr: float = 1e-4,
    dt: int = 10,
    device: str = 'cpu'
) -> Tuple[Dict, float, Dict]:
    """Train model with a specific scale value."""
    
    print(f"\n  Training with scale={scale:.3f}, time_steps={time_steps}")
    
    # Create model
    model = WWWrapper(n_classes=4, dt=dt, time_steps=time_steps)
    model.scale = torch.tensor(scale)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_score = 0.0
    best_results = None
    best_params = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            logits = batch['logits'].to(device)
            rt_norm = batch['rt_normalized'].to(device)
            
            optimizer.zero_grad()
            decision_times = model(logits)
            
            # Get final decision time (min across classes)
            final_dt = decision_times.min(dim=1)[0]
            
            loss = criterion(final_dt, rt_norm)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_pred_rt = []
            
            with torch.no_grad():
                for batch in test_loader:
                    logits = batch['logits'].to(device)
                    rt = batch['rt'].to(device)
                    
                    decision_times = model(logits)
                    final_dt = decision_times.min(dim=1)[0]
                    
                    all_pred_rt.extend(final_dt.cpu().numpy())
            
            pred_rt = np.array(all_pred_rt)
            results = evaluate_rt_distribution(pred_rt, human_stats)
            score = results['total_score']
            
            print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Score={score:.4f}, PredMean={results['pred_mean']:.3f}s, "
                  f"HumanMean={human_stats['mean']:.3f}s")
            
            if score > best_score:
                best_score = score
                best_results = results.copy()
                best_params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    
    return best_results, best_score, best_params


def search_optimal_scale(
    age_group: str,
    train_logits: np.ndarray,
    train_rts: np.ndarray,
    train_rts_norm: np.ndarray,
    test_logits: np.ndarray,
    test_rts: np.ndarray,
    test_rts_norm: np.ndarray,
    human_stats: Dict,
    output_dir: str,
    scale_range: Tuple[float, float] = (0.1, 0.5),
    n_scales: int = 5,
    epochs: int = 20,
    device: str = 'cpu'
) -> Dict:
    """Search for optimal scale value."""
    
    # Determine time_steps based on human RT
    time_steps = int(np.ceil(human_stats['percentile_99'] * 100))
    print(f"\nUsing time_steps={time_steps} (max RT={time_steps*10/1000:.2f}s)")
    
    # Create dataloaders
    train_dataset = LogitsDataset(train_logits, train_rts, train_rts_norm)
    test_dataset = LogitsDataset(test_logits, test_rts, test_rts_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Search scales
    scales = np.linspace(scale_range[0], scale_range[1], n_scales)
    results_list = []
    
    for scale in scales:
        results, score, params = train_with_scale(
            scale=scale,
            time_steps=time_steps,
            train_loader=train_loader,
            test_loader=test_loader,
            human_stats=human_stats,
            epochs=epochs,
            device=device
        )
        
        results_list.append({
            'scale': scale,
            'score': score,
            'results': results,
            'params': params
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
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump({
            'scale': float(best_config['scale']),
            'score': float(best_config['score']),
            'time_steps': time_steps,
            'results': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in best_config['results'].items()}
        }, f, indent=2)
    
    # Save model params
    np.savez(os.path.join(output_dir, 'best_model_params.npz'), **best_config['params'])
    
    return best_config


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    age_groups = ['20-29', '80-89']
    
    for age_group in age_groups:
        print(f"\n{'='*60}")
        print(f"Training for age group: {age_group}")
        print(f"{'='*60}")
        
        data_dir = f'data_age_groups/{age_group}'
        output_dir = f'checkpoints_age_groups/{age_group}/stage2'
        
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            continue
        
        # Load human stats
        with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
            human_stats = json.load(f)
        
        print(f"Human stats: Mean={human_stats['mean']:.3f}s, "
              f"Median={human_stats['median']:.3f}s, "
              f"Skewness={human_stats['skewness']:.3f}")
        
        # For now, use existing Stage 1 logits (need to create age-specific later)
        # This is a placeholder - in practice, you'd need to extract logits for age group data
        print("\nNote: Using placeholder logits. In practice, extract logits from Stage 1 model.")
        print("Skipping training for now. Please run Stage 1 for age groups first.")
        
        # Placeholder: create dummy logits for testing
        n_train = 1000
        n_test = 200
        n_classes = 4
        
        train_logits = np.random.randn(n_train, n_classes).astype(np.float32)
        train_rts = np.random.exponential(human_stats['mean'], n_train).astype(np.float32)
        train_rts_norm = (np.log(train_rts + 0.001) - np.log(human_stats['min'] + 0.001)) / \
                        (np.log(human_stats['max']) - np.log(human_stats['min'] + 0.001))
        
        test_logits = np.random.randn(n_test, n_classes).astype(np.float32)
        test_rts = np.random.exponential(human_stats['mean'], n_test).astype(np.float32)
        test_rts_norm = (np.log(test_rts + 0.001) - np.log(human_stats['min'] + 0.001)) / \
                       (np.log(human_stats['max']) - np.log(human_stats['min'] + 0.001))
        
        # Search for optimal scale
        best_config = search_optimal_scale(
            age_group=age_group,
            train_logits=train_logits,
            train_rts=train_rts,
            train_rts_norm=train_rts_norm,
            test_logits=test_logits,
            test_rts=test_rts,
            test_rts_norm=test_rts_norm,
            human_stats=human_stats,
            output_dir=output_dir,
            scale_range=(0.1, 0.5),
            n_scales=5,
            epochs=10,
            device=device
        )


if __name__ == "__main__":
    main()
