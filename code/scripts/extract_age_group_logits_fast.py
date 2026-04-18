import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_age_groups_efficient import StimulusDataset, VGGFeatureExtractor, extract_logits


def get_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def build_loader(csv_path: str) -> tuple[StimulusDataset, DataLoader]:
    dataset = StimulusDataset(csv_path)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
    )
    return dataset, loader


def save_logits_npz(path: str, logits: np.ndarray, dataset: StimulusDataset):
    np.savez_compressed(
        path,
        logits=logits,
        rts=dataset.rts,
        rts_normalized=dataset.rt_normalized,
        target_labels=dataset.target_labels,
        response_labels=dataset.response_labels,
        flanker_labels=dataset.flanker_labels,
        congruency=dataset.congruency,
    )


def main():
    parser = argparse.ArgumentParser(description='Extract Stage 1 logits for age groups')
    parser.add_argument('--age_group', choices=['20-29', '80-89'], help='Extract logits for a single age group')
    parser.add_argument('--data_root', default='data_age_groups', help='Root directory containing age-group train/test CSVs')
    parser.add_argument('--output_root', default='checkpoints_age_groups', help='Root directory for extracted logits outputs')
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    model = VGGFeatureExtractor(pretrained=False, n_classes=4)
    ckpt = torch.load('checkpoints_test/stage1/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)

    age_groups = [args.age_group] if args.age_group else ['20-29', '80-89']

    for age_group in age_groups:
        print(f"\n{'='*60}")
        print(f'Extracting logits only for age group: {age_group}')
        print(f"{'='*60}")
        stage2_dir = Path(args.output_root) / age_group / 'stage2'
        stage2_dir.mkdir(parents=True, exist_ok=True)

        train_csv = str(Path(args.data_root) / age_group / 'train_data.csv')
        test_csv = str(Path(args.data_root) / age_group / 'test_data.csv')

        train_dataset, train_loader = build_loader(train_csv)
        train_logits = extract_logits(model, train_loader, device)
        save_logits_npz(str(stage2_dir / 'train_logits.npz'), train_logits, train_dataset)
        print(f'Saved train logits: {train_logits.shape}')

        test_dataset, test_loader = build_loader(test_csv)
        test_logits = extract_logits(model, test_loader, device)
        save_logits_npz(str(stage2_dir / 'test_logits.npz'), test_logits, test_dataset)
        print(f'Saved test logits: {test_logits.shape}')


if __name__ == '__main__':
    main()
