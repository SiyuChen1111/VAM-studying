#!/bin/bash

# VGG-WongWang LIM Complete Training Pipeline
# This script runs the full two-stage training workflow
# Supports background execution with nohup

set -e

# Configuration
DATA_DIR="vam_data"
GRAPHICS_DIR="vam"
OUTPUT_BASE="checkpoints"
STAGE1_DIR="${OUTPUT_BASE}/stage1"
STAGE2_DIR="${OUTPUT_BASE}/stage2"
RESULTS_DIR="results"

# Training parameters
EPOCHS_STAGE1=30
EPOCHS_STAGE2=10000
BATCH_SIZE_STAGE1=64
BATCH_SIZE_STAGE2=1024
LR_STAGE1=1e-4
LR_STAGE2=1e-4
IMAGE_SIZE=128

# Wong-Wang parameters
DT=10
TIME_STEPS=500

# Log file
LOG_FILE="training_pipeline.log"
REUSE_STAGE1_IF_AVAILABLE="${REUSE_STAGE1_IF_AVAILABLE:-1}"

has_reusable_stage1_artifacts() {
    local required_files=(
        "${STAGE1_DIR}/best_model.pth"
        "${STAGE1_DIR}/train_logits.npz"
        "${STAGE1_DIR}/test_logits.npz"
        "${STAGE1_DIR}/rt_normalization_params.npz"
    )

    for required_file in "${required_files[@]}"; do
        if [ ! -f "${required_file}" ]; then
            return 1
        fi
    done

    return 0
}

# Create directories
mkdir -p ${STAGE1_DIR}
mkdir -p ${STAGE2_DIR}
mkdir -p ${RESULTS_DIR}

echo "=============================================="
echo "VGG-WongWang LIM Training Pipeline"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data Directory: ${DATA_DIR}"
echo "  Graphics Directory: ${GRAPHICS_DIR}"
echo "  Stage 1 Output: ${STAGE1_DIR}"
echo "  Stage 2 Output: ${STAGE2_DIR}"
echo "  Results Directory: ${RESULTS_DIR}"
echo "  Log File: ${LOG_FILE}"
echo "  Reuse Stage 1 Artifacts: ${REUSE_STAGE1_IF_AVAILABLE}"
echo ""
echo "Training Parameters:"
echo "  Stage 1 Epochs: ${EPOCHS_STAGE1}"
echo "  Stage 2 Epochs: ${EPOCHS_STAGE2}"
echo ""

# Check if data exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory ${DATA_DIR} not found!"
    exit 1
fi

if [ ! -d "${GRAPHICS_DIR}" ]; then
    echo "Error: Graphics directory ${GRAPHICS_DIR} not found!"
    exit 1
fi

# Function to run training with logging
run_training() {
    if [ "${REUSE_STAGE1_IF_AVAILABLE}" = "1" ] && has_reusable_stage1_artifacts; then
        echo "=============================================="
        echo "Stage 1: Reusing Existing Artifacts"
        echo "=============================================="
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found reusable Stage 1 artifacts in ${STAGE1_DIR}; skipping Stage 1 training." | tee -a ${LOG_FILE}
        echo ""
    else
        echo "=============================================="
        echo "Stage 1: VGG Classification Training"
        echo "=============================================="
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Stage 1..." | tee -a ${LOG_FILE}

        python train_stage1_classification.py \
            --data_dir ${DATA_DIR} \
            --graphics_dir ${GRAPHICS_DIR} \
            --output_dir ${STAGE1_DIR} \
            --epochs ${EPOCHS_STAGE1} \
            --batch_size ${BATCH_SIZE_STAGE1} \
            --lr ${LR_STAGE1} \
            --image_size ${IMAGE_SIZE} 2>&1 | tee -a ${LOG_FILE}

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Stage 1 training failed!" | tee -a ${LOG_FILE}
            exit 1
        fi

        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 complete!" | tee -a ${LOG_FILE}
        echo ""
    fi

    if ! has_reusable_stage1_artifacts; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Stage 1 artifacts are incomplete in ${STAGE1_DIR}. Expected best_model.pth, train_logits.npz, test_logits.npz, and rt_normalization_params.npz." | tee -a ${LOG_FILE}
        exit 1
    fi

    echo "=============================================="
    echo "Stage 2: Wong-Wang RT Fitting"
    echo "=============================================="
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Stage 2..." | tee -a ${LOG_FILE}

    python train_stage2_rt_fitting.py \
        --logits_dir ${STAGE1_DIR} \
        --output_dir ${STAGE2_DIR} \
        --epochs ${EPOCHS_STAGE2} \
        --batch_size ${BATCH_SIZE_STAGE2} \
        --lr ${LR_STAGE2} \
        --dt ${DT} \
        --time_steps ${TIME_STEPS} 2>&1 | tee -a ${LOG_FILE}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Stage 2 training failed!" | tee -a ${LOG_FILE}
        exit 1
    fi

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2 complete!" | tee -a ${LOG_FILE}
    echo ""

    echo "=============================================="
    echo "Model Evaluation"
    echo "=============================================="
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Evaluation..." | tee -a ${LOG_FILE}

    python evaluate_vgg_wongwang_lim.py \
        --data_dir ${DATA_DIR} \
        --graphics_dir ${GRAPHICS_DIR} \
        --stage1_dir ${STAGE1_DIR} \
        --stage2_dir ${STAGE2_DIR} \
        --output_dir ${RESULTS_DIR} \
        --image_size ${IMAGE_SIZE} 2>&1 | tee -a ${LOG_FILE}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Evaluation failed!" | tee -a ${LOG_FILE}
        exit 1
    fi

    echo ""
    echo "=============================================="
    echo "Training Pipeline Complete!"
    echo "=============================================="
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline finished successfully!" | tee -a ${LOG_FILE}
    echo ""
    echo "Results saved to: ${RESULTS_DIR}"
    echo ""
    echo "Files generated:"
    echo "  Stage 1 model: ${STAGE1_DIR}/best_model.pth"
    echo "  Stage 2 model: ${STAGE2_DIR}/best_model.pth"
    echo "  Evaluation results: ${RESULTS_DIR}/evaluation_results.csv"
    echo "  Visualizations: ${RESULTS_DIR}/*.png"
    echo "  Training log: ${LOG_FILE}"
    echo ""
}

# Main execution
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline started" | tee ${LOG_FILE}

# Run training
run_training
