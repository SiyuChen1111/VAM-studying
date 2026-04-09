#!/bin/bash

# VGG-WongWang LIM Full Training Pipeline (Background Mode)
# Uses nohup to run in background - continues even when laptop is closed
# All output is logged to training_pipeline.log

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
PID_FILE="training_pipeline.pid"
REUSE_STAGE1_IF_AVAILABLE="${REUSE_STAGE1_IF_AVAILABLE:-1}"

# Create directories
mkdir -p ${STAGE1_DIR}
mkdir -p ${STAGE2_DIR}
mkdir -p ${RESULTS_DIR}

# Check if already running
if [ -f ${PID_FILE} ]; then
    PID=$(cat ${PID_FILE})
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "Training pipeline is already running (PID: ${PID})"
        echo "To stop it, run: kill ${PID}"
        echo "To view progress: tail -f ${LOG_FILE}"
        exit 1
    else
        rm ${PID_FILE}
    fi
fi

# Check if data exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory ${DATA_DIR} not found!"
    exit 1
fi

if [ ! -d "${GRAPHICS_DIR}" ]; then
    echo "Error: Graphics directory ${GRAPHICS_DIR} not found!"
    exit 1
fi

echo "=============================================="
echo "VGG-WongWang LIM Training Pipeline (Background)"
echo "=============================================="
echo ""
echo "Starting training in background..."
echo ""
echo "Configuration:"
echo "  Data Directory: ${DATA_DIR}"
echo "  Stage 1 Epochs: ${EPOCHS_STAGE1}"
echo "  Stage 2 Epochs: ${EPOCHS_STAGE2}"
echo "  Log File: ${LOG_FILE}"
echo "  PID File: ${PID_FILE}"
echo "  Reuse Stage 1 Artifacts: ${REUSE_STAGE1_IF_AVAILABLE}"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To stop training:"
echo "  kill \$(cat ${PID_FILE})"
echo ""

# Create the training script content
cat > run_training_inner.sh << 'INNER_SCRIPT'
#!/bin/bash
set -e

DATA_DIR="vam_data"
GRAPHICS_DIR="vam"
OUTPUT_BASE="checkpoints"
STAGE1_DIR="${OUTPUT_BASE}/stage1"
STAGE2_DIR="${OUTPUT_BASE}/stage2"
RESULTS_DIR="results"
EPOCHS_STAGE1=30
EPOCHS_STAGE2=10000
BATCH_SIZE_STAGE1=64
BATCH_SIZE_STAGE2=1024
LR_STAGE1=1e-4
LR_STAGE2=1e-4
IMAGE_SIZE=128
DT=10
TIME_STEPS=500
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

if [ "${REUSE_STAGE1_IF_AVAILABLE}" = "1" ] && has_reusable_stage1_artifacts; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1: Reusing Existing Artifacts" | tee -a ${LOG_FILE}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found reusable Stage 1 artifacts in ${STAGE1_DIR}; skipping Stage 1 training." | tee -a ${LOG_FILE}
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1: VGG Classification Training" | tee -a ${LOG_FILE}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}

    python train_stage1_classification.py \
        --data_dir ${DATA_DIR} \
        --graphics_dir ${GRAPHICS_DIR} \
        --output_dir ${STAGE1_DIR} \
        --epochs ${EPOCHS_STAGE1} \
        --batch_size ${BATCH_SIZE_STAGE1} \
        --lr ${LR_STAGE1} \
        --image_size ${IMAGE_SIZE} 2>&1 | tee -a ${LOG_FILE}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Stage 1 failed!" | tee -a ${LOG_FILE}
        exit 1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 complete!" | tee -a ${LOG_FILE}
fi

if ! has_reusable_stage1_artifacts; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Stage 1 artifacts are incomplete in ${STAGE1_DIR}. Expected best_model.pth, train_logits.npz, test_logits.npz, and rt_normalization_params.npz." | tee -a ${LOG_FILE}
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2: Wong-Wang RT Fitting" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}

python train_stage2_rt_fitting.py \
    --logits_dir ${STAGE1_DIR} \
    --output_dir ${STAGE2_DIR} \
    --epochs ${EPOCHS_STAGE2} \
    --batch_size ${BATCH_SIZE_STAGE2} \
    --lr ${LR_STAGE2} \
    --dt ${DT} \
    --time_steps ${TIME_STEPS} 2>&1 | tee -a ${LOG_FILE}

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Stage 2 failed!" | tee -a ${LOG_FILE}
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2 complete!" | tee -a ${LOG_FILE}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model Evaluation" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}

python evaluate_vgg_wongwang_lim.py \
    --data_dir ${DATA_DIR} \
    --graphics_dir ${GRAPHICS_DIR} \
    --stage1_dir ${STAGE1_DIR} \
    --stage2_dir ${STAGE2_DIR} \
    --output_dir ${RESULTS_DIR} \
    --image_size ${IMAGE_SIZE} 2>&1 | tee -a ${LOG_FILE}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TRAINING PIPELINE COMPLETE!" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}

rm -f training_pipeline.pid
INNER_SCRIPT

chmod +x run_training_inner.sh

# Start in background with nohup
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training pipeline started" | tee ${LOG_FILE}

nohup ./run_training_inner.sh > /dev/null 2>&1 &
PID=$!
echo ${PID} > ${PID_FILE}

echo "Training started in background with PID: ${PID}"
echo ""
echo "Monitor with: tail -f ${LOG_FILE}"
echo "Stop with: kill ${PID}"
