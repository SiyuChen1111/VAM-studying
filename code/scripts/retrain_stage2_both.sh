#!/bin/bash

# Stage 2 Retraining Script - Both Age Groups
# Uses latest logits (20-29 from 16:24, 80-89 from current run)

set -e

echo "=========================================="
echo "Stage 2 Retraining - Both Age Groups"
echo "=========================================="
echo ""

# Step 1: Train 20-29 with new logits
echo "[Step 1/4] Training 20-29 age group..."
python3 train_age_groups_efficient.py \\
    --age_group 20-29 \\
    --use_cached_logits \\
    --scale_search 0.05,0.15,0.25,0.35

if [ $? -eq 0 ]; then
    echo "✓ 20-29 training completed"
else
    echo "✗ 20-29 training failed"
    exit 1
fi

echo ""

# Step 2: Train 80-89 with new logits (after extraction)
echo "[Step 2/4] Training 80-89 age group..."
python3 train_age_groups_efficient.py \\
    --age_group 80-89 \\
    --use_cached_logits \\
    --scale_search 0.10,0.20,0.30,0.40

if [ $? -eq 0 ]; then
    echo "✓ 80-89 training completed"
else
    echo "✗ 80-89 training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

# Show results
echo ""
echo "=== 20-29 Results ==="
cat checkpoints_age_groups/20-29/stage2/best_config.json

echo ""
echo "=== 80-89 Results ==="
cat checkpoints_age_groups/80-89/stage2/best_config.json

echo ""
echo "=== Parameter Comparison ==="
python3 -c "
import numpy as np
p1 = np.load('checkpoints_age_groups/20-29/stage2/best_model_params.npz')
p2 = np.load('checkpoints_age_groups/80-89/stage2/best_model_params.npz')

print('Parameter | 20-29 | 80-89 | Difference')
print('---|-------|-------|---')
print(f'scale | {p1[\"scale\"][0]:.3f} | {p2[\"scale\"][0]:.3f} | {((p2[\"scale\"][0]-p1[\"scale\"][0])/p1[\"scale\"][0]*100):.1f}%')
print(f'noise_ampa | {p1[\"ww.noise_ampa\"][0]:.4f} | {p2[\"ww.noise_ampa\"][0]:.4f} | {((p2[\"ww.noise_ampa\"][0]-p1[\"ww.noise_ampa\"][0])/p1[\"ww.noise_ampa\"][0]*100):.1f}%')
print(f'threshold | {p1[\"ww.threshold\"][0]:.3f} | {p2[\"ww.threshold\"][0]:.3f} | {((p2[\"ww.threshold\"][0]-p1[\"ww.threshold\"][0])/p1[\"ww.threshold\"][0]*100):.1f}%')
print(f'J_ext | {p1[\"ww.J_ext\"][0]:.4f} | {p2[\"ww.J_ext\"][0]:.4f} | {((p2[\"ww.J_ext\"][0]-p1[\"ww.J_ext\"][0])/p1[\"ww.J_ext\"][0]*100):.1f}%')
print(f'I_0 | {p1[\"ww.I_0\"][0]:.3f} | {p2[\"ww.I_0\"][0]:.3f} | {((p2[\"ww.I_0\"][0]-p1[\"ww.I_0\"][0])/p1[\"ww.I_0\"][0]*100):.1f}%')
"
