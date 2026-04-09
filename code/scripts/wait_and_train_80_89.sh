#!/bin/bash
echo "=== Wait for 80-89 extraction to finish ==="
echo ""

while true; do
    # Check if completed
    if tail -3 extract_80_89_new.log | grep -q "(100%|Completed|Finished)"; then
        echo "✓ 80-89 extraction completed!"
        break
    fi
    
    echo ""
    echo "Extraction status:"
    tail -3 extract_80_89_new.log | grep -E "(Extracting)" | tail -1
done
