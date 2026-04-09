"""
Visualize generated stimulus images from LIM dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from vgg_wongwang_lim_data import LIMDataset, DIRECTION_NAMES

def visualize_stimuli(
    data_dir: str = 'vam_data',
    graphics_dir: str = 'vam',
    num_samples: int = 16,
    save_path: str = 'stimulus_examples.png'
):
    """Visualize stimulus images from the dataset."""
    
    print("Loading dataset...")
    dataset = LIMDataset(
        data_dir=data_dir,
        graphics_dir=graphics_dir,
        image_size=128,
        split='train',
        precompute_images=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image'].numpy()
        image = np.transpose(image, (1, 2, 0))
        
        label = sample['label'].item()
        response = sample['response'].item()
        correct = sample['correct'].item()
        congruency = sample['congruency'].item()
        rt = sample['rt_original'].item()
        
        axes[i].imshow(image)
        axes[i].axis('off')
        
        title = f"Target: {DIRECTION_NAMES[label]}\n"
        title += f"Response: {DIRECTION_NAMES[response]}\n"
        title += f"RT: {rt:.3f}s\n"
        title += f"{'Correct' if correct else 'Incorrect'}"
        
        color = 'green' if correct else 'red'
        axes[i].set_title(title, fontsize=8, color=color)
    
    plt.suptitle('Lost in Migration Stimulus Examples\n(Green=Correct, Red=Incorrect)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved stimulus examples to {save_path}")
    
    return fig

def visualize_by_direction(
    data_dir: str = 'vam_data',
    graphics_dir: str = 'vam',
    save_path: str = 'stimulus_by_direction.png'
):
    """Visualize stimulus images grouped by target direction."""
    
    print("\nLoading dataset...")
    dataset = LIMDataset(
        data_dir=data_dir,
        graphics_dir=graphics_dir,
        image_size=128,
        split='train',
        precompute_images=False
    )
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for direction in range(4):
        direction_indices = [i for i in range(len(dataset)) if dataset[i]['label'].item() == direction]
        np.random.seed(42)
        selected = np.random.choice(direction_indices, min(4, len(direction_indices)), replace=False)
        
        for j, idx in enumerate(selected):
            sample = dataset[idx]
            image = sample['image'].numpy()
            image = np.transpose(image, (1, 2, 0))
            
            ax = axes[direction, j]
            ax.imshow(image)
            ax.axis('off')
            
            response = sample['response'].item()
            correct = sample['correct'].item()
            rt = sample['rt_original'].item()
            
            title = f"Resp: {DIRECTION_NAMES[response]}\nRT: {rt:.2f}s"
            color = 'green' if correct else 'red'
            ax.set_title(title, fontsize=8, color=color)
        
        axes[direction, 0].set_ylabel(DIRECTION_NAMES[direction], fontsize=12, rotation=0, labelpad=40)
    
    plt.suptitle('Stimulus Images by Target Direction\n(L=Left, R=Right, U=Up, D=Down)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved direction examples to {save_path}")

def visualize_correct_vs_incorrect(
    data_dir: str = 'vam_data',
    graphics_dir: str = 'vam',
    save_path: str = 'stimulus_correct_vs_incorrect.png'
):
    """Visualize correct vs incorrect trials."""
    
    print("\nLoading dataset...")
    dataset = LIMDataset(
        data_dir=data_dir,
        graphics_dir=graphics_dir,
        image_size=128,
        split='train',
        precompute_images=False
    )
    
    correct_indices = [i for i in range(len(dataset)) if dataset[i]['correct'].item() == 1]
    incorrect_indices = [i for i in range(len(dataset)) if dataset[i]['correct'].item() == 0]
    
    np.random.seed(42)
    correct_selected = np.random.choice(correct_indices, min(8, len(correct_indices)), replace=False)
    incorrect_selected = np.random.choice(incorrect_indices, min(8, len(incorrect_indices)), replace=False)
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    
    for i, idx in enumerate(correct_selected):
        row = i // 4
        col = i % 4
        sample = dataset[idx]
        image = sample['image'].numpy()
        image = np.transpose(image, (1, 2, 0))
        
        ax = axes[row, col]
        ax.imshow(image)
        ax.axis('off')
        
        label = sample['label'].item()
        response = sample['response'].item()
        rt = sample['rt_original'].item()
        
        ax.set_title(f"T:{DIRECTION_NAMES[label]} R:{DIRECTION_NAMES[response]}\nRT:{rt:.2f}s", fontsize=7, color='green')
    
    for i, idx in enumerate(incorrect_selected):
        row = i // 4
        col = i % 4 + 4
        sample = dataset[idx]
        image = sample['image'].numpy()
        image = np.transpose(image, (1, 2, 0))
        
        ax = axes[row, col]
        ax.imshow(image)
        ax.axis('off')
        
        label = sample['label'].item()
        response = sample['response'].item()
        rt = sample['rt_original'].item()
        
        ax.set_title(f"T:{DIRECTION_NAMES[label]} R:{DIRECTION_NAMES[response]}\nRT:{rt:.2f}s", fontsize=7, color='red')
    
    axes[0, 1].text(0.5, 1.3, 'CORRECT TRIALS', transform=axes[0, 1].transAxes, 
                    fontsize=14, ha='center', fontweight='bold', color='green')
    axes[0, 5].text(0.5, 1.3, 'INCORRECT TRIALS', transform=axes[0, 5].transAxes, 
                    fontsize=14, ha='center', fontweight='bold', color='red')
    
    plt.suptitle('Correct vs Incorrect Trials\n(T=Target, R=Response)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correct vs incorrect examples to {save_path}")

if __name__ == '__main__':
    print("="*60)
    print("Visualizing LIM Stimulus Images")
    print("="*60)
    
    visualize_stimuli()
    visualize_by_direction()
    visualize_correct_vs_incorrect()
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - stimulus_examples.png")
    print("  - stimulus_by_direction.png")
    print("  - stimulus_correct_vs_incorrect.png")
