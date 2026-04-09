# Reproduction Guide: Age-Related Differences in Decision-Making Dynamics

This guide provides step-by-step instructions to reproduce the experiments on age-related differences in Wong-Wang model parameters.

## Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (recommended for faster training)
- Required packages: numpy, pandas, scipy, matplotlib, Pillow, torchvision

## Data Preparation

### Step 1: Prepare Age Group Data

```bash
python prepare_age_group_data.py
```

This creates:
- `data_age_groups/20-29/` - Young age group data
- `data_age_groups/80-89/` - Old age group data
- Each directory contains: `train_data.csv`, `test_data.csv`, `rt_stats.json`

### Step 2: Create Stimulus Image Mapping

```bash
python create_stimulus_mapping.py
```

This creates:
- `data_age_groups/20-29/stimulus_images/` - 28 unique stimulus images
- `data_age_groups/*/stimulus_mapping.csv` - Mapping from (stimulus_layout, flanker_direction) to image

## Stage 1: Feature Extraction

### Option A: Use Existing Stage 1 Model

If you already have a trained Stage 1 model:

```bash
python train_age_groups_efficient.py
```

This will:
1. Load existing Stage 1 model from `checkpoints_test/stage1/best_model.pth`
2. Extract logits for each age group
3. Train Stage 2 with automatic scale search
4. Save results to `checkpoints_age_groups/*/stage2/`

### Option B: Train Stage 1 from Scratch

If you need to train Stage 1:

```bash
python train_stage1_classification.py --data_dir vam_data --output_dir checkpoints_age_groups/20-29/stage1 --epochs 50
```

## Stage 2: RT Fitting with Automatic Scale Search

The `train_age_groups_efficient.py` script automatically:
1. Extracts logits using Stage 1 model
2. Searches for optimal scale (0.1 to 0.5, 5 values)
3. Trains Wong-Wang model for each scale
4. Selects best scale based on RT distribution matching

## Output Structure

```
checkpoints_age_groups/
├── 20-29/
│   ├── stage1/
│   │   ├── train_logits.npz
│   │   └── test_logits.npz
│   └── stage2/
│       ├── best_config.json      # Best scale and results
│       ├── best_model_params.npz # Trained Wong-Wang parameters
│       ├── train_logits.npz
│       └── test_logits.npz
└── 80-89/
    └── ... (same structure)
```

## Key Files

| File | Purpose |
|------|---------|
| `prepare_age_group_data.py` | Split data by age group |
| `create_stimulus_mapping.py` | Create 28 stimulus images |
| `train_age_groups_efficient.py` | Main training script |
| `analyze_human_data.py` | Human data analysis |
| `research_plan.md` | Research plan and progress |

## Running on GPU

To use GPU, ensure CUDA is installed and set:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

The script automatically detects and uses GPU if available.

## Expected Results

### Human Data Analysis

| Age Group | Mean RT | Median RT | Skewness |
|-----------|---------|-----------|----------|
| 20-29 | 0.605s | 0.580s | 40.219 |
| 80-89 | 0.939s | 0.879s | 23.946 |

### Hypothesis Testing

After training, compare Wong-Wang parameters between age groups:

- **H2a**: Noise parameter (`noise_ampa`) should be higher in older adults
- **H2b**: Threshold parameter may differ
- **H2c**: External input weight (`J_ext`) may differ

## Troubleshooting

### Slow Training on CPU

- Use GPU for faster training
- Reduce batch size if memory is limited
- Use fewer epochs for initial testing

### Missing Images

- Run `create_stimulus_mapping.py` to generate stimulus images
- Check that `stimulus_image_path` column exists in CSV files

### Model Loading Errors

- Ensure Stage 1 model exists at `checkpoints_test/stage1/best_model.pth`
- Check PyTorch version compatibility

## Current Progress

| Step | Status |
|------|--------|
| Data preparation | ✓ Completed |
| Stimulus mapping | ✓ Completed |
| Stage 1 logits (20-29) | ⏳ In Progress (2%) |
| Stage 2 training (20-29) | Pending |
| Stage 1 + Stage 2 (80-89) | Pending |
| Parameter comparison | Pending |

## Next Steps

1. Wait for Stage 1 logits extraction to complete
2. Run Stage 2 training for 20-29 age group
3. Repeat for 80-89 age group
4. Compare Wong-Wang parameters between groups
5. Validate hypotheses
