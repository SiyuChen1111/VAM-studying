"""
VGG-WongWang LIM Dataset Module

Data loading for Lost in Migration task using vam_data directory.
Generates stimulus images from bird images and behavioral data.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
import argparse


DIRECTION_MAP = {'L': 0, 'R': 1, 'U': 2, 'D': 3}
DIRECTION_NAMES = ['L', 'R', 'U', 'D']

LAYOUT_SPACERS = {
    0: [51, 0],
    1: [0, 51],
    2: [51, 51],
    3: [34, 34],
    4: [34, 34],
    5: [34, 34],
    6: [34, 34],
}

CANVAS_SIZE = (640, 480)


def get_distractor_positions(targ_pos: Tuple[float, float], layout: int, spacer: List[float]) -> List[Tuple[float, float]]:
    """Calculate distractor bird positions based on layout."""
    if layout == 0:
        return [
            (targ_pos[0] - 2 * spacer[0], targ_pos[1]),
            (targ_pos[0] - spacer[0], targ_pos[1]),
            (targ_pos[0] + spacer[0], targ_pos[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1]),
        ]
    elif layout == 1:
        return [
            (targ_pos[0], targ_pos[1] - 2 * spacer[1]),
            (targ_pos[0], targ_pos[1] - spacer[1]),
            (targ_pos[0], targ_pos[1] + spacer[1]),
            (targ_pos[0], targ_pos[1] + 2 * spacer[1]),
        ]
    elif layout == 2:
        return [
            (targ_pos[0] - spacer[0], targ_pos[1]),
            (targ_pos[0] + spacer[0], targ_pos[1]),
            (targ_pos[0], targ_pos[1] - spacer[1]),
            (targ_pos[0], targ_pos[1] + spacer[1]),
        ]
    elif layout == 3:
        return [
            (targ_pos[0] + spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    elif layout == 4:
        return [
            (targ_pos[0] - spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] - spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    elif layout == 5:
        return [
            (targ_pos[0] - spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
        ]
    elif layout == 6:
        return [
            (targ_pos[0] - spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    return []


def create_stimulus_image(
    background: Image.Image,
    target_bird: Image.Image,
    distractor_bird: Image.Image,
    target_pos: Tuple[float, float],
    distractor_positions: List[Tuple[float, float]],
    canvas_size: Tuple[int, int] = CANVAS_SIZE
) -> Image.Image:
    """Create a stimulus image by compositing birds onto background."""
    img = background.copy()
    
    center_x = canvas_size[0] // 2
    center_y = canvas_size[1] // 2
    
    for dx, dy in distractor_positions:
        bird_x = center_x + dx - distractor_bird.width // 2
        bird_y = center_y - dy - distractor_bird.height // 2
        img.paste(distractor_bird, (int(bird_x), int(bird_y)), distractor_bird)
    
    target_x = center_x + target_pos[0] - target_bird.width // 2
    target_y = center_y - target_pos[1] - target_bird.height // 2
    img.paste(target_bird, (int(target_x), int(target_y)), target_bird)
    
    return img


class LIMDataset(Dataset):
    """
    Lost in Migration Dataset for VGG-WongWang training.
    
    Uses LOG NORMALIZATION for RT values.
    
    Each sample contains:
    - image: 3-channel stimulus image (128x128)
    - label: correct direction (0-3 for L/R/U/D)
    - response: human response direction (0-3)
    - rt_normalized: LOG-normalized reaction time
    - correct: whether response matches target
    - congruency: whether flankers match target (0=congruent, 1=incongruent)
    """
    
    def __init__(
        self,
        data_dir: str,
        graphics_dir: str,
        users: Optional[List[int]] = None,
        max_trials_per_user: int = 25000,
        min_rt: int = 250,
        image_size: int = 128,
        split: str = 'train',
        train_ratio: float = 0.8,
        random_seed: int = 42,
        use_log_norm: bool = True,
        precompute_images: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_dir: path to vam_data directory
            graphics_dir: path to graphics directory (bird images)
            users: list of user IDs to include (None = all users)
            max_trials_per_user: maximum trials to load per user
            min_rt: minimum RT in ms to include
            image_size: size to resize images to
            split: 'train' or 'test'
            train_ratio: ratio of data for training
            random_seed: random seed for reproducibility
            use_log_norm: use log normalization for RT
            precompute_images: whether to precompute all images
            cache_dir: directory to cache precomputed images
        """
        self.data_dir = data_dir
        self.graphics_dir = graphics_dir
        self.image_size = image_size
        self.split = split
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.use_log_norm = use_log_norm
        self.precompute_images = precompute_images
        
        if cache_dir is None:
            cache_dir = os.path.join(data_dir, 'processed_cache')
        self.cache_dir = cache_dir
        
        self._load_graphics()
        
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            if users is None:
                users = self.metadata['user_id'].tolist()
        
        self.users = users
        self._load_behavioral_data(max_trials_per_user, min_rt)
        self._compute_rt_normalization()
        
        if self.precompute_images:
            self._precompute_all_images()
        
        self._print_stats()
    
    def _load_graphics(self):
        """Load bird images and background."""
        self.background = Image.open(os.path.join(self.graphics_dir, 'bkgrnd.png')).convert('RGB')
        self.birds = {}
        for direction, idx in DIRECTION_MAP.items():
            self.birds[idx] = Image.open(os.path.join(self.graphics_dir, f'bird{idx}.png')).convert('RGBA')
    
    def _load_behavioral_data(self, max_trials: int, min_rt: int):
        """Load and preprocess behavioral data from all users."""
        all_data = []
        
        for user_id in self.users:
            csv_path = os.path.join(self.data_dir, f'user{user_id}df.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping")
                continue
            
            df = pd.read_csv(csv_path, nrows=max_trials)
            df = df[df['response_time'] >= min_rt]
            df['user_id'] = user_id
            all_data.append(df)
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        self.raw_data = pd.concat(all_data, ignore_index=True)
        self._filter_rt_outliers()
        self._add_derived_columns()
        self._split_data()
    
    def _filter_rt_outliers(self, mult: int = 10):
        """Filter RT outliers using median absolute deviation."""
        rt = self.raw_data['response_time'].values
        median = np.median(rt)
        mad = np.median(np.abs(rt - median))
        
        mask = np.abs(rt - median) < mult * mad
        n_filtered = len(self.raw_data) - mask.sum()
        
        if n_filtered > 0:
            print(f"Filtered {n_filtered} RT outliers")
            self.raw_data = self.raw_data[mask].reset_index(drop=True)
    
    def _add_derived_columns(self):
        """Add derived columns to the dataframe."""
        self.raw_data['target_dir'] = self.raw_data['target_direction'].map(DIRECTION_MAP)
        self.raw_data['flanker_dir'] = self.raw_data['flanker_direction'].map(DIRECTION_MAP)
        self.raw_data['response_dir'] = self.raw_data['response_direction'].map(DIRECTION_MAP)
        
        self.raw_data['correct'] = (self.raw_data['target_dir'] == self.raw_data['response_dir']).astype(int)
        self.raw_data['congruency'] = (self.raw_data['target_dir'] != self.raw_data['flanker_dir']).astype(int)
        self.raw_data['rt_seconds'] = self.raw_data['response_time'] / 1000.0
        
        self.raw_data['xpos_centered'] = self.raw_data['xpos'] - CANVAS_SIZE[0] / 2
        self.raw_data['ypos_centered'] = -self.raw_data['ypos'] + CANVAS_SIZE[1] / 2
    
    def _split_data(self):
        """Split data into train/test sets by users (not trials)."""
        np.random.seed(self.random_seed)
        
        unique_users = self.raw_data['user_id'].unique()
        np.random.shuffle(unique_users)
        
        n_users = len(unique_users)
        n_train_users = int(n_users * self.train_ratio)
        
        train_users = unique_users[:n_train_users]
        test_users = unique_users[n_train_users:]
        
        if self.split == 'train':
            self.data = self.raw_data[self.raw_data['user_id'].isin(train_users)].reset_index(drop=True)
        else:
            self.data = self.raw_data[self.raw_data['user_id'].isin(test_users)].reset_index(drop=True)
    
    def _compute_rt_normalization(self):
        """Compute RT normalization parameters using LOG normalization."""
        rt_values = self.data['rt_seconds'].values
        
        self.rt_min = np.min(rt_values)
        self.rt_max = np.max(rt_values)
        self.rt_range = self.rt_max - self.rt_min
        
        if self.use_log_norm:
            log_rt_values = np.log(rt_values)
            self.log_rt_min = np.min(log_rt_values)
            self.log_rt_max = np.max(log_rt_values)
            self.log_rt_range = self.log_rt_max - self.log_rt_min
            
            self.data['rt_normalized'] = (log_rt_values - self.log_rt_min) / self.log_rt_range
        else:
            self.data['rt_normalized'] = (rt_values - self.rt_min) / self.rt_range
    
    def _precompute_all_images(self):
        """Precompute and cache all stimulus images."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.image_cache = {}
        cache_file = os.path.join(self.cache_dir, f'{self.split}_images.npy')
        index_file = os.path.join(self.cache_dir, f'{self.split}_index.npy')
        
        if os.path.exists(cache_file) and os.path.exists(index_file):
            print(f"Loading cached images from {cache_file}")
            self.image_cache = np.load(cache_file, allow_pickle=True).item()
            return
        
        print(f"Precomputing {len(self.data)} images...")
        
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            img = self._create_stimulus(row)
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            self.image_cache[idx] = img_array
        
        np.save(cache_file, self.image_cache)
        np.save(index_file, np.array(list(self.image_cache.keys())))
        
        print(f"Cached images saved to {cache_file}")
    
    def _create_stimulus(self, row) -> Image.Image:
        """Create a stimulus image for a single trial."""
        target_dir = row['target_dir']
        flanker_dir = row['flanker_dir']
        layout = row['stimulus_layout']
        xpos = row['xpos_centered']
        ypos = row['ypos_centered']
        
        target_bird = self.birds[target_dir]
        distractor_bird = self.birds[flanker_dir]
        
        spacer = LAYOUT_SPACERS[layout]
        distractor_positions = get_distractor_positions((xpos, ypos), layout, spacer)
        
        img = create_stimulus_image(
            self.background, target_bird, distractor_bird,
            (xpos, ypos), distractor_positions
        )
        
        return img
    
    def _print_stats(self):
        """Print dataset statistics."""
        print(f"\nLIM Dataset ({self.split}):")
        print(f"  Total trials: {len(self.data)}")
        print(f"  Users: {self.data['user_id'].nunique()}")
        print(f"  Correct trials: {self.data['correct'].sum()} ({self.data['correct'].mean()*100:.1f}%)")
        print(f"  Congruent trials: {(self.data['congruency']==0).sum()} ({(self.data['congruency']==0).mean()*100:.1f}%)")
        print(f"  RT range: {self.rt_min:.3f} - {self.rt_max:.3f} seconds")
        print(f"  Mean RT: {self.data['rt_seconds'].mean():.3f} seconds")
        
        if self.use_log_norm:
            print(f"  Log RT range: {self.log_rt_min:.3f} - {self.log_rt_max:.3f}")
    
    def denormalize_rt(self, normalized_rt):
        """Convert normalized RT back to seconds."""
        if isinstance(normalized_rt, torch.Tensor):
            normalized_rt = normalized_rt.cpu().numpy()
        
        if self.use_log_norm:
            log_rt = normalized_rt * self.log_rt_range + self.log_rt_min
            return np.exp(log_rt)
        else:
            return normalized_rt * self.rt_range + self.rt_min
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.precompute_images:
            image = self.image_cache[idx]
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            img = self._create_stimulus(row)
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        
        return {
            'image': image,
            'label': torch.tensor(row['target_dir'], dtype=torch.long),
            'response': torch.tensor(row['response_dir'], dtype=torch.long),
            'rt_normalized': torch.tensor(row['rt_normalized'], dtype=torch.float32),
            'rt_original': torch.tensor(row['rt_seconds'], dtype=torch.float32),
            'correct': torch.tensor(row['correct'], dtype=torch.bool),
            'congruency': torch.tensor(row['congruency'], dtype=torch.long),
            'user_id': torch.tensor(row['user_id'], dtype=torch.long),
        }


class LogitsDataset(Dataset):
    """Dataset for stage 2 RT fitting with precomputed logits."""
    
    def __init__(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        responses: np.ndarray,
        rts: np.ndarray,
        rt_normalized: np.ndarray
    ):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.responses = torch.tensor(responses, dtype=torch.long)
        self.rts = torch.tensor(rts, dtype=torch.float32)
        self.rt_normalized = torch.tensor(rt_normalized, dtype=torch.float32)
    
    def __len__(self):
        return len(self.logits)
    
    def __getitem__(self, idx):
        return {
            'logits': self.logits[idx],
            'label': self.labels[idx],
            'response': self.responses[idx],
            'rt': self.rts[idx],
            'rt_normalized': self.rt_normalized[idx],
        }


def create_lim_datasets(
    data_dir: str,
    graphics_dir: str,
    **kwargs
) -> Tuple[LIMDataset, LIMDataset]:
    """Create train and test datasets."""
    train_dataset = LIMDataset(data_dir, graphics_dir, split='train', **kwargs)
    test_dataset = LIMDataset(data_dir, graphics_dir, split='test', **kwargs)
    
    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='Test LIM Dataset')
    parser.add_argument('--data_dir', type=str, default='vam_data')
    parser.add_argument('--graphics_dir', type=str, default='vam')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Testing LIM Dataset")
    print("="*60)
    
    train_dataset = LIMDataset(
        args.data_dir,
        args.graphics_dir,
        image_size=args.image_size,
        split='train'
    )
    
    test_dataset = LIMDataset(
        args.data_dir,
        args.graphics_dir,
        image_size=args.image_size,
        split='test'
    )
    
    sample = train_dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label: {sample['label']}")
    print(f"Response: {sample['response']}")
    print(f"RT (normalized): {sample['rt_normalized']:.4f}")
    print(f"RT (original): {sample['rt_original']:.4f} seconds")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(train_loader))
    print(f"\nBatch image shape: {batch['image'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    
    print("\n" + "="*60)
    print("Dataset Test Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
