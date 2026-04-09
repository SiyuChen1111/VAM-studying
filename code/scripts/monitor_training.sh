#!/bin/bash
# Monitor both extraction and training processes

echo "=== Status Monitor ==="
echo ""

# Check 80-89 extraction
if ps -p 74699 >/dev/null 2>&1; then
    echo "✓ 80-89 extraction: RUNNING (PID 74699)"
    # Show recent progress
    tail -5 extract_80_89_new.log 2>/dev/null | grep -E "(Extracting|Batch|100%)" || echo "  (no recent progress)"
else
    echo "✓ 80-89 extraction: NOT RUNNING"
fi

echo ""

# Check 20-29 training
if ps -p \$! >/dev/null 2>&1 | grep -E "train_age_groups_efficient.py.*20-29"; then
    echo "✓ 20-29 training: RUNNING"
    tail -10 train_20_29_stage2.log 2>/dev/null | grep -E "(Processing|Best|Epoch|Error)" || echo "  (no output yet)"
else
    echo "✓ 20-29 training: NOT RUNNING"
fi

echo ""
echo "Use 'bash monitor_training.sh' to refresh"
