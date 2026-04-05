"""
Efficient Stage 1 + Stage 2 training for age groups.

This script:
1. Uses existing Stage 1 model to extract logits for age groups
2. Trains Stage 2 Wong-Wang model with automatic scale search
3. Compares parameters between age groups
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as transforms
from scipy import stats
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from time import perf_counter

from vgg_wongwang_lim import VGGFeatureExtractor, WWWrapper


DIRECTION_MAP = {'L': 0, 'R': 1, 'U': 2, 'D': 3}


class StimulusDataset(Dataset):
    """Dataset for loading pre-generated stimulus images."""
    
    def __init__(self, csv_path: str, image_size: int = 128):
        """
        Args:
            csv_path: Path to CSV file with stimulus_image_path column
            image_size: Image size for resizing
        """
        self.data = pd.read_csv(csv_path)
        self.image_size = image_size
        
        # Load RT stats for normalization
        data_dir = os.path.dirname(csv_path)
        with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
            self.rt_stats = json.load(f)
        
        # Calculate log RT for normalization
        self.log_rt_min = np.log(self.rt_stats['min'] + 0.001)
        self.log_rt_max = np.log(self.rt_stats['max'])
        self.log_rt_range = self.log_rt_max - self.log_rt_min
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_paths = self.data['stimulus_image_path'].astype(str).to_numpy()
        self.rts = (self.data['response_time'].to_numpy(dtype=np.float32) / 1000.0)
        self.rt_normalized = ((np.log(self.rts + 0.001) - self.log_rt_min) / self.log_rt_range).astype(np.float32)
        self.target_labels = self.data['target_direction'].map(lambda x: DIRECTION_MAP[x]).to_numpy(dtype=np.int64)
        self.response_labels = self.data['response_direction'].map(lambda x: DIRECTION_MAP[x]).to_numpy(dtype=np.int64)
        self.flanker_labels = self.data['flanker_direction'].map(lambda x: DIRECTION_MAP[x]).to_numpy(dtype=np.int64)
        self.congruency = (self.target_labels != self.flanker_labels).astype(np.int64)
        self.correct = (self.target_labels == self.response_labels).astype(np.float32)

        unique_paths = sorted(set(self.image_paths.tolist()))
        self.image_cache = {}
        zero_image = torch.zeros(3, self.image_size, self.image_size)
        for path in unique_paths:
            if os.path.exists(path):
                image = Image.open(path).convert('RGB')
                self.image_cache[path] = self.transform(image)
            else:
                self.image_cache[path] = zero_image

        print(f"Loaded {len(self.data)} samples from {csv_path} with {len(self.image_cache)} cached images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        return {
            'image': self.image_cache[image_path],
            'rt': torch.tensor(self.rts[idx], dtype=torch.float32),
            'rt_normalized': torch.tensor(self.rt_normalized[idx], dtype=torch.float32),
            'target_label': torch.tensor(self.target_labels[idx], dtype=torch.long),
            'response_label': torch.tensor(self.response_labels[idx], dtype=torch.long),
            'congruency': torch.tensor(self.congruency[idx], dtype=torch.long),
            'correct': torch.tensor(self.correct[idx], dtype=torch.float32)
        }


def extract_logits(model: nn.Module, dataloader: DataLoader, device: str) -> np.ndarray:
    """Extract logits using Stage 1 model."""
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            images = batch['image'].to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
    
    return np.concatenate(all_logits, axis=0)


def evaluate_joint_behavior(
    pred_rt: np.ndarray,
    pred_choice: np.ndarray,
    true_rt: np.ndarray,
    target_labels: np.ndarray,
    response_labels: np.ndarray,
    congruency: np.ndarray,
    human_stats: Dict,
    rt_shape_focus: bool = False,
) -> Dict:
    pred_mean = pred_rt.mean()
    pred_median = np.median(pred_rt)
    pred_skewness = stats.skew(pred_rt)
    pred_q90 = float(np.quantile(pred_rt, 0.90))
    pred_q95 = float(np.quantile(pred_rt, 0.95))
    pred_q99 = float(np.quantile(pred_rt, 0.99))
    
    true_mean = human_stats['mean']
    true_median = human_stats['median']
    true_skewness = human_stats['skewness']
    true_q90 = float(np.quantile(true_rt, 0.90))
    true_q95 = float(np.quantile(true_rt, 0.95))
    true_q99 = float(np.quantile(true_rt, 0.99))
    
    # Score components
    skewness_score = min(max(pred_skewness, 0.0) / max(true_skewness, 1.0), 1.0)
    mean_diff = abs(pred_mean - true_mean) / true_mean
    mean_score = max(0, 1 - mean_diff)
    median_diff = abs(pred_median - true_median) / true_median
    median_score = max(0, 1 - median_diff)
    pred_range = pred_rt.max() - pred_rt.min()
    true_range = human_stats['max'] - human_stats['min']
    coverage_score = min(pred_range / true_range, 1.0)
    q90_score = max(0, 1 - abs(pred_q90 - true_q90) / max(true_q90, 1e-6))
    q95_score = max(0, 1 - abs(pred_q95 - true_q95) / max(true_q95, 1e-6))
    q99_score = max(0, 1 - abs(pred_q99 - true_q99) / max(true_q99, 1e-6))
    quantile_score = (q90_score + q95_score + q99_score) / 3.0

    human_accuracy = float((response_labels == target_labels).mean())
    model_accuracy = float((pred_choice == target_labels).mean())
    response_agreement = float((pred_choice == response_labels).mean())
    acc_diff = abs(model_accuracy - human_accuracy) / max(human_accuracy, 1e-6)
    accuracy_score = max(0, 1 - acc_diff)

    human_cong_gap = float(true_rt[congruency == 1].mean() - true_rt[congruency == 0].mean())
    model_cong_gap = float(pred_rt[congruency == 1].mean() - pred_rt[congruency == 0].mean())
    if human_cong_gap > 0:
        cong_diff = abs(model_cong_gap - human_cong_gap) / human_cong_gap
        congruency_score = max(0, 1 - cong_diff)
    else:
        congruency_score = 1.0 if model_cong_gap <= 0 else 0.0
    
    if rt_shape_focus:
        rt_score = 0.30 * skewness_score + 0.10 * mean_score + 0.10 * median_score + 0.20 * coverage_score + 0.30 * quantile_score
        total_score = 0.60 * rt_score + 0.25 * accuracy_score + 0.15 * congruency_score
    else:
        rt_score = 0.3 * skewness_score + 0.3 * mean_score + 0.2 * median_score + 0.2 * coverage_score
        total_score = 0.45 * rt_score + 0.35 * accuracy_score + 0.20 * congruency_score

    mean_median_score = 0.5 * mean_score + 0.5 * median_score
    rt_shape_score = 0.35 * skewness_score + 0.35 * quantile_score + 0.30 * coverage_score
    behavior_optimal_score = (
        0.40 * rt_shape_score
        + 0.30 * response_agreement
        + 0.15 * congruency_score
        + 0.10 * mean_median_score
        + 0.05 * accuracy_score
    )
    
    return {
        'pred_mean': pred_mean,
        'pred_median': pred_median,
        'pred_skewness': pred_skewness,
        'true_mean': true_mean,
        'true_median': true_median,
        'true_skewness': true_skewness,
        'pred_q90': pred_q90,
        'pred_q95': pred_q95,
        'pred_q99': pred_q99,
        'true_q90': true_q90,
        'true_q95': true_q95,
        'true_q99': true_q99,
        'skewness_score': skewness_score,
        'mean_score': mean_score,
        'median_score': median_score,
        'coverage_score': coverage_score,
        'q90_score': q90_score,
        'q95_score': q95_score,
        'q99_score': q99_score,
        'quantile_score': quantile_score,
        'mean_median_score': mean_median_score,
        'rt_shape_score': rt_shape_score,
        'rt_score': rt_score,
        'human_accuracy': human_accuracy,
        'model_accuracy': model_accuracy,
        'response_agreement': response_agreement,
        'accuracy_score': accuracy_score,
        'human_congruency_rt_gap': human_cong_gap,
        'model_congruency_rt_gap': model_cong_gap,
        'congruency_score': congruency_score,
        'total_score': total_score,
        'behavior_optimal_score': behavior_optimal_score,
    }


def compute_stage2_outputs(model: WWWrapper, logits_batch: torch.Tensor, choice_temperature: float):
    scale_tensor = model.state_dict()['scale']
    scaled_logits = F.relu(logits_batch * scale_tensor)
    decision_times = model.ww(scaled_logits)
    final_dt = decision_times.min(dim=1)[0]

    _, trajectory, threshold = model.ww.inference(scaled_logits)
    class_strength = (trajectory - threshold).amax(dim=1)
    choice_logits = class_strength / choice_temperature

    return final_dt, choice_logits, decision_times


def behavior_optimal_key(results: Dict):
    return (
        float(results['rt_shape_score']),
        float(results['response_agreement']),
        float(results['congruency_score']),
        float(results['mean_median_score']),
        float(results['accuracy_score']),
    )


def train_stage2_with_scale(
    scale: float,
    time_steps: int,
    logits: np.ndarray,
    rts: np.ndarray,
    rts_normalized: np.ndarray,
    target_labels: np.ndarray,
    response_labels: np.ndarray,
    congruency: np.ndarray,
    human_stats: Dict,
    epochs: int = 20,
    lr: float = 1e-4,
    dt: int = 10,
    choice_temperature: float = 0.05,
    lambda_rt: float = 1.0,
    lambda_choice: float = 2.0,
    lambda_cong: float = 0.3,
    lambda_tail: float = 0.0,
    lambda_pileup: float = 0.0,
    fixed_noise_ampa: Optional[float] = None,
    congruency_margin: float = 0.01,
    device: str = 'cpu',
    log_prefix: str = '',
    rt_shape_focus: bool = False,
    tail_quantiles: Optional[np.ndarray] = None,
) -> Tuple[Dict, float, Dict]:
    """Train Stage 2 model with a specific scale value."""
    
    # Create model
    model = WWWrapper(n_classes=4, dt=dt, time_steps=time_steps)
    model.scale = torch.tensor(scale, dtype=torch.float32)
    if fixed_noise_ampa is not None:
        with torch.no_grad():
            model.ww.noise_ampa.copy_(torch.tensor(float(fixed_noise_ampa), dtype=torch.float32))
        model.ww.noise_ampa.requires_grad_(False)
    model = model.to(device)
    
    # Create tensor dataset
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    rts_tensor = torch.tensor(rts_normalized, dtype=torch.float32)
    target_tensor = torch.tensor(target_labels, dtype=torch.long)
    response_tensor = torch.tensor(response_labels, dtype=torch.long)
    congruency_tensor = torch.tensor(congruency, dtype=torch.long)

    dataset = TensorDataset(logits_tensor, rts_tensor, target_tensor, response_tensor, congruency_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if tail_quantiles is None:
        tail_quantiles = np.array([0.90, 0.95, 0.99], dtype=np.float32)
    tail_quantiles_t = torch.tensor(tail_quantiles, dtype=torch.float32, device=device)
    
    best_score = 0.0
    best_results = None
    best_params = None
    best_key = None

    print(f"{log_prefix}Starting scale={scale:.3f} for {epochs} epochs on {len(dataset)} samples")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_start = perf_counter()
        
        for batch_logits, batch_rt, batch_target, batch_response, batch_congruency in dataloader:
            batch_logits = batch_logits.to(device)
            batch_rt = batch_rt.to(device)
            batch_target = batch_target.to(device)
            batch_response = batch_response.to(device)
            batch_congruency = batch_congruency.to(device)
            
            optimizer.zero_grad()
            final_dt, choice_logits, _ = compute_stage2_outputs(model, batch_logits, choice_temperature)
            rt_loss = criterion(final_dt, batch_rt)
            choice_loss = F.cross_entropy(choice_logits, batch_response)

            if (batch_congruency == 0).any() and (batch_congruency == 1).any():
                mean_rt_cong = final_dt[batch_congruency == 0].mean()
                mean_rt_incong = final_dt[batch_congruency == 1].mean()
                cong_loss = F.relu(congruency_margin - (mean_rt_incong - mean_rt_cong))
            else:
                cong_loss = torch.tensor(0.0, device=device)

            if lambda_tail > 0:
                pred_q = torch.quantile(final_dt, tail_quantiles_t)
                true_q = torch.quantile(batch_rt, tail_quantiles_t)
                tail_loss = F.l1_loss(pred_q, true_q)
            else:
                tail_loss = torch.tensor(0.0, device=device)

            if lambda_pileup > 0:
                boundary_start = 0.95 * batch_rt.max().detach()
                pred_pile = (final_dt >= boundary_start).float().mean()
                true_pile = (batch_rt >= boundary_start).float().mean()
                pileup_loss = F.l1_loss(pred_pile, true_pile)
            else:
                pileup_loss = torch.tensor(0.0, device=device)

            loss = lambda_rt * rt_loss + lambda_choice * choice_loss + lambda_cong * cong_loss + lambda_tail * tail_loss + lambda_pileup * pileup_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        epoch_duration = perf_counter() - epoch_start
        print(f"{log_prefix}Epoch {epoch + 1:02d}/{epochs} avg_loss={avg_loss:.6f} duration={epoch_duration:.2f}s")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_rt_t, choice_logits_t, _ = compute_stage2_outputs(model, logits_tensor.to(device), choice_temperature)
                pred_rt = pred_rt_t.cpu().numpy()
                pred_choice = choice_logits_t.argmax(dim=1).cpu().numpy()
            
            results = evaluate_joint_behavior(
                pred_rt=pred_rt,
                pred_choice=pred_choice,
                true_rt=rts,
                target_labels=target_labels,
                response_labels=response_labels,
                congruency=congruency,
                human_stats=human_stats,
                rt_shape_focus=rt_shape_focus,
            )
            score = results['total_score']
            print(
                f"{log_prefix}Eval epoch {epoch + 1:02d}: "
                f"score={score:.4f} rt_score={results['rt_score']:.4f} "
                f"acc={results['model_accuracy']:.4f}/{results['human_accuracy']:.4f} "
                f"resp_agree={results['response_agreement']:.4f} "
                f"cong_gap={results['model_congruency_rt_gap']:.4f}/{results['human_congruency_rt_gap']:.4f} "
                f"rt_shape={results['rt_shape_score']:.4f} behavior_opt={results['behavior_optimal_score']:.4f}"
            )
            
            current_key = behavior_optimal_key(results)
            if best_key is None or current_key > best_key:
                best_score = score
                best_results = results.copy()
                best_params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
                best_key = current_key
                print(f"{log_prefix}New best score={best_score:.4f} at epoch {epoch + 1:02d}")
    
    if best_results is None or best_params is None:
        raise RuntimeError(f"No valid Stage 2 result found for scale={scale}")

    print(f"{log_prefix}Completed scale={scale:.3f} with best_score={best_score:.4f}")

    return best_results, best_score, best_params


def load_cached_logits_npz(npz_path: str) -> Dict[str, np.ndarray]:
    cached = np.load(npz_path)
    required_keys = ['logits', 'rts', 'rts_normalized', 'target_labels', 'response_labels', 'congruency']
    missing = [key for key in required_keys if key not in cached.files]
    if missing:
        raise KeyError(f"Cached logits file missing required keys {missing}: {npz_path}")

    return {key: cached[key] for key in required_keys}


def validate_cached_stage2_inputs(age_group: str, data_dir: str, train_logits_path: str, test_logits_path: str):
    required_files = [
        os.path.join(data_dir, 'train_data.csv'),
        os.path.join(data_dir, 'test_data.csv'),
        os.path.join(data_dir, 'rt_stats.json'),
        train_logits_path,
        test_logits_path,
    ]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required cached Stage 2 inputs for {age_group}: {missing}")

    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    train_cached = load_cached_logits_npz(train_logits_path)
    test_cached = load_cached_logits_npz(test_logits_path)

    if len(train_df) != len(train_cached['logits']):
        raise ValueError(f"Train length mismatch for {age_group}: csv={len(train_df)} logits={len(train_cached['logits'])}")
    if len(test_df) != len(test_cached['logits']):
        raise ValueError(f"Test length mismatch for {age_group}: csv={len(test_df)} logits={len(test_cached['logits'])}")

    metadata_path = os.path.join(data_dir, 'matched_branch_metadata.json')
    if 'matched' in Path(data_dir).parts and not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Matched branch metadata missing for {age_group}: {metadata_path}")

    return train_cached, test_cached


def save_partial_best_snapshot(
    output_dir: str,
    age_group: str,
    scale: float,
    epoch: int,
    score: float,
    results: Dict,
    params: Dict,
    pred_rt: np.ndarray,
    pred_choice: np.ndarray,
    target_labels: np.ndarray,
    response_labels: np.ndarray,
    congruency: np.ndarray,
    time_steps: int,
    data_root: str,
    output_root: str,
    supervision_type: str,
):
    partial_dir = os.path.join(output_dir, 'partial_best')
    os.makedirs(partial_dir, exist_ok=True)
    with open(os.path.join(partial_dir, 'best_config.partial.json'), 'w') as f:
        json.dump({
            'age_group': age_group,
            'scale': float(scale),
            'score': float(score),
            'epoch': int(epoch),
            'time_steps': int(time_steps),
            'results': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in results.items()},
        }, f, indent=2)
    np.savez(os.path.join(partial_dir, 'best_model_params.partial.npz'), **params)
    np.savez_compressed(
        os.path.join(partial_dir, 'best_test_predictions.partial.npz'),
        pred_rt=pred_rt,
        pred_choice=pred_choice,
        target_labels=target_labels,
        response_labels=response_labels,
        congruency=congruency,
    )
    with open(os.path.join(partial_dir, 'best_checkpoint_meta.json'), 'w') as f:
        json.dump({
            'age_group': age_group,
            'data_root': data_root,
            'output_root': output_root,
            'supervision_type': supervision_type,
            'scale': float(scale),
            'epoch': int(epoch),
            'score': float(score),
            'rt_score': float(results['rt_score']),
            'model_accuracy': float(results['model_accuracy']),
            'human_accuracy': float(results['human_accuracy']),
            'response_agreement': float(results['response_agreement']),
            'model_congruency_rt_gap': float(results['model_congruency_rt_gap']),
            'human_congruency_rt_gap': float(results['human_congruency_rt_gap']),
            'rt_shape_score': float(results['rt_shape_score']),
            'behavior_optimal_score': float(results['behavior_optimal_score']),
        }, f, indent=2)


def infer_predictions_from_params(
    params: Dict,
    scale: float,
    time_steps: int,
    logits: np.ndarray,
    device: str,
    choice_temperature: float = 0.05,
):
    model = WWWrapper(n_classes=4, dt=10, time_steps=time_steps)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in params:
            state_dict[key] = torch.tensor(params[key], dtype=torch.float32)
    state_dict['scale'] = torch.tensor(scale, dtype=torch.float32)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred_rt_t, choice_logits_t, _ = compute_stage2_outputs(model, torch.tensor(logits, dtype=torch.float32).to(device), choice_temperature)
    return pred_rt_t.cpu().numpy(), choice_logits_t.argmax(dim=1).cpu().numpy()


def fit_stage2_from_logits(
    age_group: str,
    output_dir: str,
    human_stats: Dict,
    train_cached: Dict[str, np.ndarray],
    test_cached: Dict[str, np.ndarray],
    device: str = 'cpu',
    scales: Optional[np.ndarray] = None,
    data_root: str = 'data_age_groups',
    output_root: str = 'checkpoints_age_groups',
    supervision_type: str = 'response_labels',
    epochs: int = 20,
    lambda_rt: float = 1.0,
    lambda_choice: float = 2.0,
    lambda_cong: float = 0.3,
    choice_temperature: float = 0.05,
    lambda_tail: float = 0.0,
    lambda_pileup: float = 0.0,
    fixed_noise_ampa: Optional[float] = None,
    time_steps_factor: float = 1.0,
    rt_shape_focus: bool = False,
    tail_quantiles: Optional[np.ndarray] = None,
):
    print(f"\n{'='*60}")
    print(f"Processing age group from cached logits: {age_group}")
    print(f"{'='*60}")
    print(f"Human stats: Mean={human_stats['mean']:.3f}s, Median={human_stats['median']:.3f}s")
    print(f"Choice supervision: {supervision_type}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Epochs: {epochs}")
    print(f"lambda_rt={lambda_rt} lambda_choice={lambda_choice} lambda_cong={lambda_cong} lambda_tail={lambda_tail} lambda_pileup={lambda_pileup} choice_temperature={choice_temperature}")
    if fixed_noise_ampa is not None:
        print(f"Fixed noise_ampa: {fixed_noise_ampa}")

    time_steps = int(np.ceil(human_stats['percentile_99'] * 100 * time_steps_factor))
    print(f"Using time_steps={time_steps} (max RT={time_steps*10/1000:.2f}s)")
    print(f"RT shape focus: {rt_shape_focus}")

    train_logits = train_cached['logits']
    train_rts = train_cached['rts']
    train_rts_norm = train_cached['rts_normalized']
    train_target = train_cached['target_labels']
    train_response = train_cached['response_labels']
    train_congruency = train_cached['congruency']
    test_target = test_cached['target_labels']
    test_response = test_cached['response_labels']
    test_congruency = test_cached['congruency']

    if scales is None:
        scales = np.linspace(0.1, 0.5, 5)

    print("\nSearching for optimal scale using cached logits...")
    results_list = []
    total_scales = len(scales)
    global_best_score = -np.inf

    for scale_idx, scale in enumerate(scales, start=1):
        scale_start = perf_counter()
        log_prefix = f"[{age_group} scale {scale_idx}/{total_scales}] "
        print(f"{log_prefix}Beginning optimization")
        results, score, params = train_stage2_with_scale(
            scale=scale,
            time_steps=time_steps,
            logits=train_logits,
            rts=train_rts,
            rts_normalized=train_rts_norm,
            target_labels=train_target,
            response_labels=train_response,
            congruency=train_congruency,
            human_stats=human_stats,
            epochs=epochs,
            lambda_rt=lambda_rt,
            lambda_choice=lambda_choice,
            lambda_cong=lambda_cong,
            lambda_tail=lambda_tail,
            lambda_pileup=lambda_pileup,
            fixed_noise_ampa=fixed_noise_ampa,
            choice_temperature=choice_temperature,
            rt_shape_focus=rt_shape_focus,
            tail_quantiles=tail_quantiles,
            device=device,
            log_prefix=log_prefix,
        )

        results_list.append({
            'scale': scale,
            'score': score,
            'results': results,
            'params': params,
        })

        test_pred_rt, test_pred_choice = infer_predictions_from_params(
            params=params,
            scale=scale,
            time_steps=time_steps,
            logits=test_cached['logits'],
            device=device,
        )

        if score > global_best_score:
            global_best_score = score
            save_partial_best_snapshot(
                output_dir=output_dir,
                age_group=age_group,
                scale=scale,
                epoch=20,
                score=score,
                results=results,
                params=params,
                pred_rt=test_pred_rt,
                pred_choice=test_pred_choice,
                target_labels=test_target,
                response_labels=test_response,
                congruency=test_congruency,
                time_steps=time_steps,
                data_root=data_root,
                output_root=output_root,
                supervision_type=supervision_type,
            )

        scale_duration = perf_counter() - scale_start
        print(
            f"{log_prefix}Finished in {scale_duration:.2f}s | "
            f"Score={score:.4f}, PredMean={results['pred_mean']:.3f}s, "
            f"Acc={results['model_accuracy']:.4f}, Cong={results['model_congruency_rt_gap']:.4f}"
        )

    best_idx = np.argmax([r['score'] for r in results_list])
    best_config = results_list[best_idx]

    print(f"\n{'='*50}")
    print(f"Best scale for {age_group}: {best_config['scale']:.3f}")
    print(f"Score: {best_config['score']:.4f}")
    print(f"Pred Mean RT: {best_config['results']['pred_mean']:.3f}s")
    print(f"Human Mean RT: {human_stats['mean']:.3f}s")
    print(f"Model Acc: {best_config['results']['model_accuracy']:.4f}")
    print(f"Human Acc: {best_config['results']['human_accuracy']:.4f}")
    print(f"Response agreement: {best_config['results']['response_agreement']:.4f}")
    print(f"RT-shape score: {best_config['results']['rt_shape_score']:.4f}")
    print(f"Behavior-optimal score: {best_config['results']['behavior_optimal_score']:.4f}")
    print(f"{'='*50}")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump({
            'scale': float(best_config['scale']),
            'score': float(best_config['score']),
            'time_steps': time_steps,
            'results': {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in best_config['results'].items()}
        }, f, indent=2)

    np.savez(os.path.join(output_dir, 'best_model_params.npz'), **best_config['params'])

    np.savez(
        os.path.join(output_dir, 'train_logits.npz'),
        **train_cached,
    )

    np.savez(
        os.path.join(output_dir, 'test_logits.npz'),
        **test_cached,
    )

    return best_config


def process_age_group(
    age_group: str,
    stage1_model_path: str,
    data_dir: str,
    output_dir: str,
    device: str = 'cpu'
):
    """Process a single age group: extract logits and train Stage 2."""
    
    print(f"\n{'='*60}")
    print(f"Processing age group: {age_group}")
    print(f"{'='*60}")
    
    # Load human stats
    with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
        human_stats = json.load(f)
    
    print(f"Human stats: Mean={human_stats['mean']:.3f}s, Median={human_stats['median']:.3f}s")
    
    # Determine time_steps based on human RT
    time_steps = int(np.ceil(human_stats['percentile_99'] * 100))
    print(f"Using time_steps={time_steps} (max RT={time_steps*10/1000:.2f}s)")
    
    # Load Stage 1 model
    print("Loading Stage 1 model...")
    model = VGGFeatureExtractor(pretrained=False, n_classes=4)
    checkpoint = torch.load(stage1_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    # Create datasets
    train_dataset = StimulusDataset(os.path.join(data_dir, 'train_data.csv'))
    test_dataset = StimulusDataset(os.path.join(data_dir, 'test_data.csv'))
    
    loader_kwargs = {
        'batch_size': 64,
        'shuffle': False,
        'num_workers': min(4, os.cpu_count() or 1),
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    
    # Extract logits
    print("\nExtracting train logits...")
    train_logits = extract_logits(model, train_loader, device)
    
    print("Extracting test logits...")
    test_logits = extract_logits(model, test_loader, device)
    
    # Get RTs
    train_rts = train_dataset.rts
    train_rts_norm = train_dataset.rt_normalized
    train_target = train_dataset.target_labels
    train_response = train_dataset.response_labels
    train_congruency = train_dataset.congruency
    
    # Search for optimal scale
    print("\nSearching for optimal scale...")
    scales = np.linspace(0.1, 0.5, 5)
    results_list = []
    
    for scale in scales:
        results, score, params = train_stage2_with_scale(
            scale=scale,
            time_steps=time_steps,
            logits=train_logits,
            rts=train_rts,
            rts_normalized=train_rts_norm,
            target_labels=train_target,
            response_labels=train_response,
            congruency=train_congruency,
            human_stats=human_stats,
            epochs=20,
            device=device
        )
        
        results_list.append({
            'scale': scale,
            'score': score,
            'results': results,
            'params': params
        })
        
        print(f"  Scale={scale:.3f}: Score={score:.4f}, PredMean={results['pred_mean']:.3f}s")
    
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
    
    # Save logits
    np.savez(
        os.path.join(output_dir, 'train_logits.npz'),
        logits=train_logits,
        rts=train_rts,
        rts_normalized=train_rts_norm
    )
    
    np.savez(
        os.path.join(output_dir, 'test_logits.npz'),
        logits=test_logits,
        rts=test_dataset.rts,
        rts_normalized=test_dataset.rt_normalized
    )
    
    return best_config


def main():
    parser = argparse.ArgumentParser(description='Train age-group Stage 2 models')
    parser.add_argument('--age_group', choices=['20-29', '80-89'], help='Run only one age group')
    parser.add_argument('--cached_only', action='store_true', help='Use cached logits instead of extracting from Stage 1')
    parser.add_argument('--train_logits_path', type=str, help='Path to cached train logits npz')
    parser.add_argument('--test_logits_path', type=str, help='Path to cached test logits npz')
    parser.add_argument('--data_root', default='data_age_groups', help='Root directory containing age-group data')
    parser.add_argument('--output_root', default='checkpoints_age_groups', help='Root directory for Stage 2 outputs')
    parser.add_argument('--scale_values', type=str, help='Comma-separated scale values, e.g. 0.1 or 0.1,0.3,0.5')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per scale')
    parser.add_argument('--lambda_rt', type=float, default=1.0, help='Weight for RT loss')
    parser.add_argument('--lambda_choice', type=float, default=2.0, help='Weight for choice loss')
    parser.add_argument('--lambda_cong', type=float, default=0.3, help='Weight for congruency loss')
    parser.add_argument('--lambda_tail', type=float, default=0.0, help='Weight for tail quantile loss')
    parser.add_argument('--lambda_pileup', type=float, default=0.0, help='Weight for anti-boundary-pileup loss')
    parser.add_argument('--fixed_noise_ampa', type=float, help='If set, override and freeze Wong-Wang noise_ampa')
    parser.add_argument('--choice_temperature', type=float, default=0.05, help='Choice temperature for logits')
    parser.add_argument('--time_steps_factor', type=float, default=1.0, help='Multiplier on time_steps derived from human percentile_99')
    parser.add_argument('--rt_shape_focus', action='store_true', help='Upweight skewness and tail coverage in checkpoint selection')
    parser.add_argument('--tail_quantiles', type=str, default='0.9,0.95,0.99', help='Comma-separated quantiles for tail loss')
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    stage1_model_path = 'checkpoints_test/stage1/best_model.pth'
    if not os.path.exists(stage1_model_path):
        print(f"Stage 1 model not found at {stage1_model_path}")
        return
    
    age_groups = [args.age_group] if args.age_group else ['20-29', '80-89']
    custom_scales = None
    if args.scale_values:
        custom_scales = np.array([float(x.strip()) for x in args.scale_values.split(',') if x.strip()], dtype=np.float32)
    tail_quantiles = np.array([float(x.strip()) for x in args.tail_quantiles.split(',') if x.strip()], dtype=np.float32)
    all_results = {}
    
    for age_group in age_groups:
        data_dir = os.path.join(args.data_root, age_group)
        output_dir = os.path.join(args.output_root, age_group, 'stage2')
        
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            continue
        
        if args.cached_only:
            train_logits_path = args.train_logits_path or os.path.join(output_dir, 'train_logits.npz')
            test_logits_path = args.test_logits_path or os.path.join(output_dir, 'test_logits.npz')

            with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
                human_stats = json.load(f)

            print(f"Using cached logits for {age_group}:")
            print(f"  train: {train_logits_path}")
            print(f"  test: {test_logits_path}")
            print("  choice supervision: response_labels")

            train_cached, test_cached = validate_cached_stage2_inputs(
                age_group=age_group,
                data_dir=data_dir,
                train_logits_path=train_logits_path,
                test_logits_path=test_logits_path,
            )

            result = fit_stage2_from_logits(
                age_group=age_group,
                output_dir=output_dir,
                human_stats=human_stats,
                train_cached=train_cached,
                test_cached=test_cached,
                device=device,
                scales=custom_scales,
                data_root=args.data_root,
                output_root=args.output_root,
                supervision_type='response_labels',
                epochs=args.epochs,
                lambda_rt=args.lambda_rt,
                lambda_choice=args.lambda_choice,
                lambda_cong=args.lambda_cong,
                lambda_tail=args.lambda_tail,
                lambda_pileup=args.lambda_pileup,
                fixed_noise_ampa=args.fixed_noise_ampa,
                choice_temperature=args.choice_temperature,
                time_steps_factor=args.time_steps_factor,
                rt_shape_focus=args.rt_shape_focus,
                tail_quantiles=tail_quantiles,
            )
        else:
            result = process_age_group(
                age_group=age_group,
                stage1_model_path=stage1_model_path,
                data_dir=data_dir,
                output_dir=output_dir,
                device=device
            )
        
        all_results[age_group] = result
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN AGE GROUPS")
    print(f"{'='*60}")
    
    for age_group, result in all_results.items():
        print(f"\n{age_group}:")
        print(f"  Best scale: {result['scale']:.3f}")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Pred Mean RT: {result['results']['pred_mean']:.3f}s")
        print(f"  Pred Skewness: {result['results']['pred_skewness']:.3f}")


if __name__ == "__main__":
    main()
