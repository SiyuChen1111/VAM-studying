#!/bin/bash

# Monitor Stage 2 RT Fitting

LOG_FILE="test_pipeline.log"

echo "=============================================="
echo "Stage 2 RT Fitting Monitor"
echo "=============================================="
echo ""

# Check if Stage 2 is running
STAGE2_PID=$(ps aux | grep "train_stage2" | grep -v grep | awk '{print $2}')

if [ -z "$STAGE2_PID" ]; then
    echo "Status: Stage 2 is NOT running"
    echo ""
    echo "Stage 1 logits:"
    ls -la checkpoints_test/stage1/*.npz 2>/dev/null
    echo ""
    echo "Stage 2 checkpoints:"
    ls -la checkpoints_test/stage2/*.pth 2>/dev/null
    exit 0
fi

echo "Status: RUNNING (PID: $STAGE2_PID)"
echo ""

# Show Stage 2 specific logs
echo "Stage 2 training progress:"
echo "----------------------------------------"
grep -E "(Stage 2|Epoch [0-9]+/|Train Corr|Test Corr|correlation|Loading|Train Loss|Test Loss)" "$LOG_FILE" 2>/dev/null | tail -30
echo "----------------------------------------"
echo ""

# Show checkpoints
echo "Stage 2 checkpoints:"
ls -la checkpoints_test/stage2/*.pth 2>/dev/null || echo "No checkpoints yet"
echo ""

echo "Commands:"
echo "  Monitor: ./monitor_stage2.sh"
echo "  Stop: kill $STAGE2_PID"
