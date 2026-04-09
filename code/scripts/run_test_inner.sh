#!/bin/bash
set -e

DATA_DIR="vam_data"
GRAPHICS_DIR="vam"
OUTPUT_BASE="checkpoints_test"
STAGE1_DIR="${OUTPUT_BASE}/stage1"
STAGE2_DIR="${OUTPUT_BASE}/stage2"
RESULTS_DIR="results_test"
EPOCHS_STAGE1=3
EPOCHS_STAGE2=100
BATCH_SIZE_STAGE1=32
BATCH_SIZE_STAGE2=256
LR_STAGE1=1e-4
LR_STAGE2=1e-4
IMAGE_SIZE=128
MAX_TRIALS=1000
DT=10
TIME_STEPS=500
LOG_FILE="test_pipeline.log"
PID_FILE="test_pipeline.pid"
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found reusable Stage 1 artifacts in ${STAGE1_DIR}; skipping Stage 1 test." | tee -a ${LOG_FILE}
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1: VGG Classification Test" | tee -a ${LOG_FILE}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}

    python train_stage1_classification.py \
        --data_dir ${DATA_DIR} \
        --graphics_dir ${GRAPHICS_DIR} \
        --output_dir ${STAGE1_DIR} \
        --epochs ${EPOCHS_STAGE1} \
        --batch_size ${BATCH_SIZE_STAGE1} \
        --lr ${LR_STAGE1} \
        --image_size ${IMAGE_SIZE} \
        --max_trials_per_user ${MAX_TRIALS} 2>&1 | tee -a ${LOG_FILE}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Stage 1 test failed!" | tee -a ${LOG_FILE}
        rm -f ${PID_FILE}
        exit 1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 test complete!" | tee -a ${LOG_FILE}
fi

if ! has_reusable_stage1_artifacts; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Stage 1 artifacts are incomplete in ${STAGE1_DIR}. Expected best_model.pth, train_logits.npz, test_logits.npz, and rt_normalization_params.npz." | tee -a ${LOG_FILE}
    rm -f ${PID_FILE}
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2: Wong-Wang RT Fitting Test" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}

python train_stage2_rt_fitting.py \
    --logits_dir ${STAGE1_DIR} \
    --output_dir ${STAGE2_DIR} \
    --epochs ${EPOCHS_STAGE2} \
    --batch_size ${BATCH_SIZE_STAGE2} \
    --lr ${LR_STAGE2} \
    --dt ${DT} \
    --time_steps ${TIME_STEPS} \
    --eval_every 20 \
    --save_every 50 2>&1 | tee -a ${LOG_FILE}

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Stage 2 test failed!" | tee -a ${LOG_FILE}
    rm -f ${PID_FILE}
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2 test complete!" | tee -a ${LOG_FILE}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model Evaluation Test" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}

python evaluate_vgg_wongwang_lim.py \
    --data_dir ${DATA_DIR} \
    --graphics_dir ${GRAPHICS_DIR} \
    --stage1_dir ${STAGE1_DIR} \
    --stage2_dir ${STAGE2_DIR} \
    --output_dir ${RESULTS_DIR} \
    --image_size ${IMAGE_SIZE} 2>&1 | tee -a ${LOG_FILE}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST PIPELINE COMPLETE!" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==============================================" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] If test passed, run full training with:" | tee -a ${LOG_FILE}
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   ./run_pipeline_background.sh" | tee -a ${LOG_FILE}

rm -f ${PID_FILE}
