"""
Stage 1 Training: VGG Classification Training

Train VGG16 feature extractor on Lost in Migration classification task.
This stage learns to classify the target bird direction (L/R/U/D).

Usage:
    python train_stage1_classification.py --data_dir vam_data --graphics_dir vam --epochs 30
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from vgg_wongwang_lim_data import create_lim_datasets, LIMDataset
from vgg_wongwang_lim import VGGFeatureExtractor, create_model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    learn_correct_label: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        responses = batch['response'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        
        target = labels if learn_correct_label else responses
        loss = criterion(logits, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        preds = logits.argmax(dim=1)
        correct_vs_label = (preds == labels).sum().item()
        correct_vs_response = (preds == responses).sum().item()
        
        total_correct += correct_vs_label
        total_samples += images.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc_label': f'{correct_vs_label / images.size(0):.3f}',
            'acc_resp': f'{correct_vs_response / images.size(0):.3f}'
        })
    
    return {
        'loss': total_loss / total_samples,
        'accuracy_label': total_correct / total_samples
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
    total_correct_label = 0
    total_correct_response = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    all_responses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            responses = batch['response'].to(device)
            
            logits = model(images)
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            
            preds = logits.argmax(dim=1)
            total_correct_label += (preds == labels).sum().item()
            total_correct_response += (preds == responses).sum().item()
            total_samples += images.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_responses.extend(responses.cpu().numpy())
    
    return {
        'loss': total_loss / total_samples,
        'accuracy_label': total_correct_label / total_samples,
        'accuracy_response': total_correct_response / total_samples,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'responses': np.array(all_responses)
    }


def save_logits(
    model: nn.Module,
    dataset: LIMDataset,
    device: torch.device,
    batch_size: int = 64
) -> Dict[str, np.ndarray]:
    """Save logits and labels for stage 2 training."""
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_logits = []
    all_labels = []
    all_responses = []
    all_rts = []
    all_rt_normalized = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Saving logits'):
            images = batch['image'].to(device)
            logits = model(images)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch['label'].numpy())
            all_responses.append(batch['response'].numpy())
            all_rts.append(batch['rt_original'].numpy())
            all_rt_normalized.append(batch['rt_normalized'].numpy())
    
    return {
        'logits': np.concatenate(all_logits, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
        'responses': np.concatenate(all_responses, axis=0),
        'rts': np.concatenate(all_rts, axis=0),
        'rt_normalized': np.concatenate(all_rt_normalized, axis=0)
    }


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
    
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['test_acc'], label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 1: VGG Classification Training')
    
    parser.add_argument('--data_dir', type=str, default='vam_data',
                        help='Path to vam_data directory')
    parser.add_argument('--graphics_dir', type=str, default='vam',
                        help='Path to graphics directory (bird images)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage1',
                        help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--freeze_features', action='store_true',
                        help='Freeze VGG features')
    parser.add_argument('--learn_correct_label', action='store_true',
                        help='Learn correct label instead of human response (default: learn human response)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_trials_per_user', type=int, default=25000,
                        help='Maximum trials per user')
    
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
    print("Stage 1: VGG Classification Training")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Graphics Directory: {args.graphics_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Freeze Features: {args.freeze_features}")
    print(f"  Learn Correct Label: {args.learn_correct_label}")
    print(f"  Device: {device}")
    
    print("\nLoading datasets...")
    train_dataset, test_dataset = create_lim_datasets(
        args.data_dir,
        args.graphics_dir,
        image_size=args.image_size,
        max_trials_per_user=args.max_trials_per_user,
        random_seed=args.random_seed
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
    
    print("\nCreating model...")
    model = VGGFeatureExtractor(
        pretrained=True,
        freeze_features=args.freeze_features,
        n_classes=4
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    best_epoch = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            learn_correct_label=args.learn_correct_label
        )
        
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy_label'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy_label'])
        
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy_label']*100:.1f}%")
        print(f"  Test Loss: {test_metrics['loss']:.4f}, Test Acc (label): {test_metrics['accuracy_label']*100:.1f}%, Test Acc (response): {test_metrics['accuracy_response']*100:.1f}%")
        
        if test_metrics['accuracy_label'] > best_acc:
            best_acc = test_metrics['accuracy_label']
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_metrics['accuracy_label'],
                'config': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  Saved best model (accuracy: {best_acc*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Test Accuracy: {best_acc*100:.1f}% (epoch {best_epoch})")
    
    print("\nSaving final model and logits...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args)
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    train_logits = save_logits(model, train_dataset, device)
    test_logits = save_logits(model, test_dataset, device)
    
    np.savez(
        os.path.join(args.output_dir, 'train_logits.npz'),
        **train_logits
    )
    np.savez(
        os.path.join(args.output_dir, 'test_logits.npz'),
        **test_logits
    )
    
    np.savez(
        os.path.join(args.output_dir, 'rt_normalization_params.npz'),
        log_rt_min=train_dataset.log_rt_min,
        log_rt_max=train_dataset.log_rt_max,
        log_rt_range=train_dataset.log_rt_range,
        rt_min=train_dataset.rt_min,
        rt_max=train_dataset.rt_max,
        rt_range=train_dataset.rt_range
    )
    print(f"  Saved logits and RT normalization params to {args.output_dir}")
    
    plot_training_curves(
        history,
        os.path.join(args.output_dir, 'training_curves.png')
    )
    
    print("\nStage 1 complete! Run stage 2 for RT fitting.")


if __name__ == '__main__':
    main()
