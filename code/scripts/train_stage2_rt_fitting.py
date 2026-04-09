"""
Stage 2 Training: Wong-Wang RT Fitting

Train Wong-Wang decision module to fit human reaction times.
Uses precomputed logits from Stage 1.

Usage:
    python train_stage2_rt_fitting.py --logits_dir checkpoints/stage1 --epochs 10000
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Dict

from vgg_wongwang_lim import (
    WWWrapper,
    WongWangMultiClassDecision,
    NegativePearsonCorrelationLoss
)
from vgg_wongwang_lim_data import LogitsDataset


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_pred_rt = []
    all_true_rt = []
    
    for batch in train_loader:
        logits = batch['logits'].to(device)
        rt = batch['rt_normalized'].to(device)
        
        optimizer.zero_grad()
        
        decision_times = model(logits)
        pred_rt = decision_times.min(dim=1).values
        
        loss = criterion(pred_rt, rt)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.0001)
        optimizer.step()
        
        total_loss += loss.item() * logits.size(0)
        total_samples += logits.size(0)
        
        all_pred_rt.extend(pred_rt.detach().cpu().numpy())
        all_true_rt.extend(rt.cpu().numpy())
    
    correlation = np.corrcoef(all_pred_rt, all_true_rt)[0, 1]
    
    return {
        'loss': total_loss / total_samples,
        'correlation': correlation
    }


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_pred_rt = []
    all_true_rt = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            logits = batch['logits'].to(device)
            rt = batch['rt_normalized'].to(device)
            labels = batch['label']
            
            decision_times = model(logits)
            pred_rt = decision_times.min(dim=1).values
            
            loss = criterion(pred_rt, rt)
            
            total_loss += loss.item() * logits.size(0)
            total_samples += logits.size(0)
            
            all_pred_rt.extend(pred_rt.cpu().numpy())
            all_true_rt.extend(rt.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            pred_direction = decision_times.argmin(dim=1).cpu().numpy()
            all_preds.extend(pred_direction)
    
    correlation, p_value = stats.pearsonr(all_pred_rt, all_true_rt)
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    slope, intercept, r_value, _, _ = stats.linregress(all_pred_rt, all_true_rt)
    
    return {
        'loss': total_loss / total_samples,
        'correlation': correlation,
        'p_value': p_value,
        'accuracy': accuracy,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'pred_rt': np.array(all_pred_rt),
        'true_rt': np.array(all_true_rt)
    }


def plot_results(
    results: Dict,
    save_dir: str,
    epoch: int,
    rt_norm_params: dict = None
):
    """Plot RT fitting results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    pred_rt = results['pred_rt']
    true_rt = results['true_rt']
    
    if rt_norm_params is not None:
        log_rt_min = float(rt_norm_params['log_rt_min'])
        log_rt_range = float(rt_norm_params['log_rt_range'])
        
        # Only denormalize true_rt (which is normalized)
        # pred_rt is already in seconds from Wong-Wang model
        true_rt_denorm = np.exp(true_rt * log_rt_range + log_rt_min)
        true_rt = true_rt_denorm
    
    axes[0].scatter(pred_rt, true_rt, alpha=0.3, s=1)
    
    slope, intercept, r_value, _, _ = stats.linregress(pred_rt, true_rt)
    x_line = np.array([pred_rt.min(), pred_rt.max()])
    y_line = slope * x_line + intercept
    axes[0].plot(x_line, y_line, 'r-', label=f'y={slope:.2f}x+{intercept:.2f}')
    
    axes[0].set_xlabel('Predicted RT (s)')
    axes[0].set_ylabel('True RT (s)')
    axes[0].set_title(f'RT Correlation (r={results["correlation"]:.3f}, p={results["p_value"]:.2e})')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].hist(pred_rt, bins=50, alpha=0.5, label='Predicted RT', density=True)
    axes[1].hist(true_rt, bins=50, alpha=0.5, label='True RT', density=True)
    axes[1].set_xlabel('RT (s)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('RT Distributions')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rt_fitting_epoch_{epoch}.png'), dpi=150)
    plt.close()


def plot_training_curves(history: Dict[str, list], save_path: str):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['test_loss'], label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_corr'], label='Train Correlation')
    axes[1].plot(history['test_corr'], label='Test Correlation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Correlation Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Wong-Wang RT Fitting')
    
    parser.add_argument('--logits_dir', type=str, default='checkpoints/stage1',
                        help='Directory containing logits from Stage 1')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage2',
                        help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--dt', type=int, default=10,
                        help='Time step for Wong-Wang dynamics (ms)')
    parser.add_argument('--time_steps', type=int, default=200,
                        help='Total time steps for Wong-Wang dynamics')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--loss_type', type=str, default='pearson',
                        choices=['pearson', 'mse'],
                        help='Loss function type')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save model every N epochs')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
    
    print("\n" + "="*60)
    print("Stage 2: Wong-Wang RT Fitting")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Logits Directory: {args.logits_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  DT: {args.dt} ms")
    print(f"  Time Steps: {args.time_steps}")
    print(f"  Loss Type: {args.loss_type}")
    print(f"  Device: {device}")
    
    print("\nLoading logits...")
    train_data = np.load(os.path.join(args.logits_dir, 'train_logits.npz'))
    test_data = np.load(os.path.join(args.logits_dir, 'test_logits.npz'))
    
    rt_norm_params = None
    rt_norm_path = os.path.join(args.logits_dir, 'rt_normalization_params.npz')
    if os.path.exists(rt_norm_path):
        rt_norm_params = np.load(rt_norm_path)
        print(f"  Loaded RT normalization params")
    
    train_dataset = LogitsDataset(
        train_data['logits'],
        train_data['labels'],
        train_data['responses'],
        train_data['rts'],
        train_data['rt_normalized']
    )
    
    test_dataset = LogitsDataset(
        test_data['logits'],
        test_data['labels'],
        test_data['responses'],
        test_data['rts'],
        test_data['rt_normalized']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    print("\nCreating Wong-Wang model...")
    model = WWWrapper(
        n_classes=args.n_classes,
        dt=args.dt,
        time_steps=args.time_steps
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")
    
    if args.loss_type == 'pearson':
        criterion = NegativePearsonCorrelationLoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    history = {
        'train_loss': [],
        'train_corr': [],
        'test_loss': [],
        'test_corr': []
    }
    
    best_corr = -1.0
    best_epoch = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_corr'].append(train_metrics['correlation'])
        
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            test_metrics = evaluate(model, test_loader, criterion, device)
            
            history['test_loss'].append(test_metrics['loss'])
            history['test_corr'].append(test_metrics['correlation'])
            
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Corr: {train_metrics['correlation']:.4f}")
            print(f"  Test Loss: {test_metrics['loss']:.4f}, Test Corr: {test_metrics['correlation']:.4f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']*100:.1f}%")
            
            if test_metrics['correlation'] > best_corr:
                best_corr = test_metrics['correlation']
                best_epoch = epoch + 1
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'correlation': test_metrics['correlation'],
                    'config': vars(args)
                }, os.path.join(args.output_dir, 'best_model.pth'))
                print(f"  Saved best model (correlation: {best_corr:.4f})")
                
                plot_results(test_metrics, args.output_dir, epoch+1, rt_norm_params)
        
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Test Correlation: {best_corr:.4f} (epoch {best_epoch})")
    
    plot_training_curves(
        history,
        os.path.join(args.output_dir, 'training_curves.png')
    )
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'best_correlation': best_corr,
        'best_epoch': best_epoch
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    print(f"\nFinal model saved to {args.output_dir}")
    print("\nStage 2 complete!")


if __name__ == '__main__':
    main()
