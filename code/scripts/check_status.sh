#!/bin/bash

# Check training pipeline status
# Usage: ./check_status.sh [test|full]

MODE=${1:-full}

if [ "$MODE" = "test" ]; then
    PID_FILE="test_pipeline.pid"
    LOG_FILE="test_pipeline.log"
    OUTPUT_BASE="checkpoints_test"
    RESULTS_DIR="results_test"
    TITLE="Test Pipeline"
else
    PID_FILE="training_pipeline.pid"
    LOG_FILE="training_pipeline.log"
    OUTPUT_BASE="checkpoints"
    RESULTS_DIR="results"
    TITLE="Full Training Pipeline"
fi

echo "=============================================="
echo "VGG-WongWang ${TITLE} Status"
echo "=============================================="
echo ""

if [ -f ${PID_FILE} ]; then
    PID=$(cat ${PID_FILE})
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "Status: RUNNING (PID: ${PID})"
        echo ""
        echo "Recent log entries:"
        echo "----------------------------------------"
        tail -30 ${LOG_FILE}
        echo "----------------------------------------"
        echo ""
        echo "To monitor live: tail -f ${LOG_FILE}"
        echo "To stop: kill ${PID}"
    else
        echo "Status: STOPPED (stale PID file found)"
        rm ${PID_FILE}
        echo ""
        if [ -f ${LOG_FILE} ]; then
            echo "Last log entries:"
            echo "----------------------------------------"
            tail -30 ${LOG_FILE}
            echo "----------------------------------------"
        fi
    fi
else
    echo "Status: NOT RUNNING"
    echo ""
    if [ -f ${LOG_FILE} ]; then
        echo "Last log entries:"
        echo "----------------------------------------"
        tail -30 ${LOG_FILE}
        echo "----------------------------------------"
    else
        echo "No log file found. Pipeline has not been started."
    fi
fi

echo ""
echo "Checkpoints:"
if [ -d "${OUTPUT_BASE}/stage1" ]; then
    echo "  Stage 1: $(ls ${OUTPUT_BASE}/stage1/*.pth 2>/dev/null | wc -l | tr -d ' ') files"
fi
if [ -d "${OUTPUT_BASE}/stage2" ]; then
    echo "  Stage 2: $(ls ${OUTPUT_BASE}/stage2/*.pth 2>/dev/null | wc -l | tr -d ' ') files"
fi

echo ""
echo "Results:"
if [ -d "${RESULTS_DIR}" ]; then
    echo "  Directory: ${RESULTS_DIR}"
    ls -la ${RESULTS_DIR} 2>/dev/null | head -10
fi

echo ""
echo "=============================================="
echo "Commands:"
echo "  Start test:    ./run_test.sh"
echo "  Start full:    ./run_pipeline_background.sh"
echo "  Check test:    ./check_status.sh test"
echo "  Check full:    ./check_status.sh"
echo "  Monitor log:   tail -f ${LOG_FILE}"
echo "=============================================="
