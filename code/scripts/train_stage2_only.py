"""
Stage 2 training only - uses pre-extracted logits.

This script:
1. Loads pre-extracted logits from Stage 1
2. Trains Stage 2 Wong-Wang model with automatic scale search
3. Compares parameters between age groups
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from tqdm import tqdm
from typing import Dict, Tuple

from vgg_wongwang_lim import WWWrapper

def evaluate_rt_distribution(pred_rt: np.ndarray, rts: np.ndarray, human_stats: Dict) -> Dict:
    """Evaluate how well predicted RT matches human RT distribution."""
    
    pred_mean = pred_rt.mean()
    pred_median = np.median(pred_rt)
    pred_skewness = stats.skew(pred_rt)
    
    true_mean = human_stats['mean']
    true_median = human_stats['median']
    true_skewness = human_stats['skewness']
    
    # Score components
    skewness_score = min(pred_skewness / max(true_skewness, 1.0), 1.0) if pred_skewness > 0.5 else 0.0
    mean_diff = abs(pred_mean - true_mean) / true_mean
    mean_score = max(0, 1 - mean_diff)
    median_diff = abs(pred_median - true_median) / true_median
    median_score = max(0, 1 - median_diff)
    pred_range = pred_rt.max() - pred_rt.min()
    true_range = human_stats['max'] - human_stats['min']
    coverage_score = min(pred_range / true_range, 1.0)
    
    total_score = 0.3 * skewness_score + 0.3 * mean_score + 0.2 * median_score + 0.2 * coverage_score
    
    return {
        'pred_mean': pred_mean,
        'pred_median': pred_median,
        'pred_skewness': pred_skewness,
        'true_mean': true_mean,
        'true_median': true_median,
        'true_skewness': true_skewness,
        'ske