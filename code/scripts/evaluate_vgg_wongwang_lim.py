"""
Evaluate VGG-WongWang LIM Model

Comprehensive evaluation of the two-stage model including:
- Classification accuracy (vs correct label and human response)
- RT correlation analysis
- RT by correctness analysis
- RT by congruency analysis
- Visualization of results

Usage:
    python evaluate_vgg_wongwang_lim.py --stage1_dir checkpoints/stage1 --stage2_dir checkpoints/stage2
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from typing import Dict

from vgg_wongwang_lim_data import create_lim_datasets, LIMDataset
from vgg_wongwang_lim import VGGWongWangLIM, WWWrapper, VGGFeatureExtractor


def load_complete_model(
    stage1_path: str,
    stage2_path: str,
    device: torch.device,
    n_classes: int = 4
) -> VGGWongWangLIM:
    """Load complete two-stage model."""
    stage1_ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
    stage2_ckpt = torch.load(stage2_path, map_location=device, weights_only=False)
    
    model = VGGWongWangLIM(
        pretrained=False,
        n_classes=n_classes
    )
    
    model.feature_extractor.load_state_dict(stage1_ckpt['model_state_dict'])
    model.ww_wrapper.load_state_dict(stage2_ckpt['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(
    model: VGGWongWangLIM,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    
    all_logits = []
    all_decision_times = []
    all_pred_rt = []
    all_true_rt = []
    all_labels = []
    all_responses = []
    all_correct = []
    all_congruency = []
    all_user_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            
            logits, decision_time, rt_pred = model(images)
            
            decision_times = model.get_decision_times(logits)
            
            all_logits.append(logits.cpu().numpy())
            all_decision_times.append(decision_times.cpu().numpy())
            all_pred_rt.append(rt_pred.cpu().numpy())
            all_true_rt.append(batch['rt_original'].numpy())
            all_labels.append(batch['label'].numpy())
            all_responses.append(batch['response'].numpy())
            all_correct.append(batch['correct'].numpy())
            all_congruency.append(batch['congruency'].numpy())
            all_user_ids.append(batch['user_id'].numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_decision_times = np.concatenate(all_decision_times, axis=0)
    all_pred_rt = np.concatenate(all_pred_rt, axis=0)
    all_true_rt = np.concatenate(all_true_rt, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_responses = np.concatenate(all_responses, axis=0)
    all_correct = np.concatenate(all_correct, axis=0)
    all_congruency = np.concatenate(all_congruency, axis=0)
    all_user_ids = np.concatenate(all_user_ids, axis=0)
    
    pred_directions = all_decision_times.argmin(axis=1)
    
    accuracy_label = np.mean(pred_directions == all_labels)
    accuracy_response = np.mean(pred_directions == all_responses)
    
    correlation, p_value = stats.pearsonr(all_pred_rt, all_true_rt)
    slope, intercept, r_value, _, _ = stats.linregress(all_pred_rt, all_true_rt)
    
    correct_mask = all_correct == 1
    correct_rt_pred = all_pred_rt[correct_mask]
    correct_rt_true = all_true_rt[correct_mask]
    incorrect_rt_pred = all_pred_rt[~correct_mask]
    incorrect_rt_true = all_true_rt[~correct_mask]
    
    congruent_mask = all_congruency == 0
    congruent_rt_pred = all_pred_rt[congruent_mask]
    congruent_rt_true = all_true_rt[congruent_mask]
    incongruent_rt_pred = all_pred_rt[~congruent_mask]
    incongruent_rt_true = all_true_rt[~congruent_mask]
    
    return {
        'logits': all_logits,
        'decision_times': all_decision_times,
        'pred_rt': all_pred_rt,
        'true_rt': all_true_rt,
        'labels': all_labels,
        'responses': all_responses,
        'correct': all_correct,
        'congruency': all_congruency,
        'user_ids': all_user_ids,
        'pred_directions': pred_directions,
        'accuracy_label': accuracy_label,
        'accuracy_response': accuracy_response,
        'correlation': correlation,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'correct_rt_pred': correct_rt_pred,
        'correct_rt_true': correct_rt_true,
        'incorrect_rt_pred': incorrect_rt_pred,
        'incorrect_rt_true': incorrect_rt_true,
        'congruent_rt_pred': congruent_rt_pred,
        'congruent_rt_true': congruent_rt_true,
        'incongruent_rt_pred': incongruent_rt_pred,
        'incongruent_rt_true': incongruent_rt_true,
    }


def plot_rt_correlation(results: dict, save_path: str):
    """Plot RT correlation scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(results['pred_rt'], results['true_rt'], alpha=0.3, s=5)
    
    x_line = np.array([results['pred_rt'].min(), results['pred_rt'].max()])
    y_line = results['slope'] * x_line + results['intercept']
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'y={results["slope"]:.2f}x+{results["intercept"]:.2f}')
    
    ax.set_xlabel('Predicted RT (s)', fontsize=12)
    ax.set_ylabel('True RT (s)', fontsize=12)
    ax.set_title(f'RT Correlation: r={results["correlation"]:.3f}, p={results["p_value"]:.2e}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"RT correlation plot saved to {save_path}")


def plot_rt_distributions(results: dict, save_path: str):
    """Plot RT distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(results['pred_rt'], bins=50, alpha=0.7, label='Predicted', density=True)
    axes[0, 0].hist(results['true_rt'], bins=50, alpha=0.7, label='True', density=True)
    axes[0, 0].set_xlabel('RT (s)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Overall RT Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(results['correct_rt_pred'], bins=50, alpha=0.7, label='Predicted', density=True)
    axes[0, 1].hist(results['correct_rt_true'], bins=50, alpha=0.7, label='True', density=True)
    axes[0, 1].set_xlabel('RT (s)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title(f'Correct Trials (n={len(results["correct_rt_true"])})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if len(results['incorrect_rt_true']) > 0:
        axes[1, 0].hist(results['incorrect_rt_pred'], bins=50, alpha=0.7, label='Predicted', density=True)
        axes[1, 0].hist(results['incorrect_rt_true'], bins=50, alpha=0.7, label='True', density=True)
        axes[1, 0].set_xlabel('RT (s)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title(f'Incorrect Trials (n={len(results["incorrect_rt_true"])})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(results['congruent_rt_pred'], bins=50, alpha=0.7, label='Congruent Pred', density=True)
    axes[1, 1].hist(results['incongruent_rt_pred'], bins=50, alpha=0.7, label='Incongruent Pred', density=True)
    axes[1, 1].set_xlabel('RT (s)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('RT by Congruency (Predicted)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"RT distributions plot saved to {save_path}")


def plot_confusion_matrix(results: dict, save_path: str):
    """Plot confusion matrix for direction predictions."""
    from sklearn.metrics import confusion_matrix
    
    direction_names = ['L', 'R', 'U', 'D']
    
    cm = confusion_matrix(results['labels'], results['pred_directions'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=direction_names, yticklabels=direction_names, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=direction_names, yticklabels=direction_names, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_rt_by_correctness(results: dict, save_path: str):
    """Plot RT by correctness."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    correct_data = pd.DataFrame({
        'RT': results['correct_rt_true'],
        'Type': 'True',
        'Correctness': 'Correct'
    })
    incorrect_data = pd.DataFrame({
        'RT': results['incorrect_rt_true'],
        'Type': 'True',
        'Correctness': 'Incorrect'
    })
    true_data = pd.concat([correct_data, incorrect_data])
    
    sns.boxplot(data=true_data, x='Correctness', y='RT', ax=axes[0])
    axes[0].set_title('True RT by Correctness')
    axes[0].set_ylabel('RT (s)')
    
    correct_pred_data = pd.DataFrame({
        'RT': results['correct_rt_pred'],
        'Type': 'Predicted',
        'Correctness': 'Correct'
    })
    incorrect_pred_data = pd.DataFrame({
        'RT': results['incorrect_rt_pred'],
        'Type': 'Predicted',
        'Correctness': 'Incorrect'
    })
    pred_data = pd.concat([correct_pred_data, incorrect_pred_data])
    
    sns.boxplot(data=pred_data, x='Correctness', y='RT', ax=axes[1])
    axes[1].set_title('Predicted RT by Correctness')
    axes[1].set_ylabel('RT (s)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"RT by correctness plot saved to {save_path}")


def print_summary(results: dict):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    print(f"\nClassification Performance:")
    print(f"  Accuracy (vs correct label): {results['accuracy_label']*100:.2f}%")
    print(f"  Accuracy (vs human response): {results['accuracy_response']*100:.2f}%")
    
    print(f"\nRT Prediction Performance:")
    print(f"  Pearson correlation: r={results['correlation']:.4f}, p={results['p_value']:.2e}")
    print(f"  R-squared: {results['r_squared']:.4f}")
    print(f"  Regression: y={results['slope']:.2f}x+{results['intercept']:.2f}")
    
    print(f"\nRT by Correctness (True):")
    print(f"  Correct trials: {results['correct_rt_true'].mean():.3f} ± {results['correct_rt_true'].std():.3f} s (n={len(results['correct_rt_true'])})")
    if len(results['incorrect_rt_true']) > 0:
        print(f"  Incorrect trials: {results['incorrect_rt_true'].mean():.3f} ± {results['incorrect_rt_true'].std():.3f} s (n={len(results['incorrect_rt_true'])})")
    
    print(f"\nRT by Correctness (Predicted):")
    print(f"  Correct trials: {results['correct_rt_pred'].mean():.3f} ± {results['correct_rt_pred'].std():.3f} s")
    if len(results['incorrect_rt_pred']) > 0:
        print(f"  Incorrect trials: {results['incorrect_rt_pred'].mean():.3f} ± {results['incorrect_rt_pred'].std():.3f} s")
    
    print(f"\nRT by Congruency (True):")
    print(f"  Congruent: {results['congruent_rt_true'].mean():.3f} ± {results['congruent_rt_true'].std():.3f} s (n={len(results['congruent_rt_true'])})")
    print(f"  Incongruent: {results['incongruent_rt_true'].mean():.3f} ± {results['incongruent_rt_true'].std():.3f} s (n={len(results['incongruent_rt_true'])})")
    
    print(f"\nRT by Congruency (Predicted):")
    print(f"  Congruent: {results['congruent_rt_pred'].mean():.3f} ± {results['congruent_rt_pred'].std():.3f} s")
    print(f"  Incongruent: {results['incongruent_rt_pred'].mean():.3f} ± {results['incongruent_rt_pred'].std():.3f} s")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate VGG-WongWang LIM Model')
    
    parser.add_argument('--data_dir', type=str, default='vam_data',
                        help='Path to vam_data directory')
    parser.add_argument('--graphics_dir', type=str, default='vam',
                        help='Path to graphics directory')
    parser.add_argument('--stage1_dir', type=str, default='checkpoints/stage1',
                        help='Directory containing Stage 1 model')
    parser.add_argument('--stage2_dir', type=str, default='checkpoints/stage2',
                        help='Directory containing Stage 2 model')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')
    
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
    
    print("\n" + "="*60)
    print("VGG-WongWang LIM Model Evaluation")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Stage 1 Model: {args.stage1_dir}")
    print(f"  Stage 2 Model: {args.stage2_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Device: {device}")
    
    print("\nLoading test dataset...")
    _, test_dataset = create_lim_datasets(
        args.data_dir,
        args.graphics_dir,
        image_size=args.image_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print("\nLoading model...")
    stage1_path = os.path.join(args.stage1_dir, 'best_model.pth')
    stage2_path = os.path.join(args.stage2_dir, 'best_model.pth')
    
    model = load_complete_model(stage1_path, stage2_path, device)
    
    print("\nRunning evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    print_summary(results)
    
    print("\nSaving results...")
    plot_rt_correlation(results, os.path.join(args.output_dir, 'rt_correlation.png'))
    plot_rt_distributions(results, os.path.join(args.output_dir, 'rt_distributions.png'))
    plot_confusion_matrix(results, os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_rt_by_correctness(results, os.path.join(args.output_dir, 'rt_by_correctness.png'))
    
    results_df = pd.DataFrame({
        'pred_rt': results['pred_rt'],
        'true_rt': results['true_rt'],
        'label': results['labels'],
        'response': results['responses'],
        'pred_direction': results['pred_directions'],
        'correct': results['correct'],
        'congruency': results['congruency'],
        'user_id': results['user_ids']
    })
    results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
    
    print(f"\nResults saved to {args.output_dir}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
