#!/bin/bash
echo "============================================================"
echo "Training Status Monitor"
date "+%Y-%m-%d %H:%M:%S"
echo "============================================================"
echo ""

# Check 80-89 extraction
if ps -p 74699 >/dev/null 2>&1; then
    PROGRESS=$(tail -5 extract_80_89_new.log 2>/dev/null | grep "Extracting:" | tail -1)
    echo "🔄 80-89 extraction: RUNNING"
    echo "   Latest: $PROGRESS"
else
    echo "✓ 80-89 extraction: DONE"
    echo ""
    # Show completion status if done
    tail -3 extract_80_89_new.log 2>/dev/null | grep -E "(100%|Completed|Error)"
fi

echo ""

# Check 20-29 training
if ps -p 75713 >/dev/null 2>&1; then
    PROGRESS=$(tail -10 train_20_29_stage2.log 2>/dev/null | grep -E "(Processing|Epoch|Best)" | tail -2)
    echo "🔄 20-29 training: RUNNING"
    echo "   Latest:"
    echo "$PROGRESS" | head -3
else
    echo "✓ 20-29 training: DONE"
    echo ""
    # Show results if done
    if [ -f checkpoints_age_groups/20-29/stage2/best_config.json ]; then
        echo "   Results available"
    fi
fi

echo ""
echo "============================================================"
echo "Refresh in 10 seconds with: bash monitor_both.sh"
echo "============================================================"
