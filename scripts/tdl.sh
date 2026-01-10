#!/bin/bash

# ==========================================
# 1. basic configurations
# ==========================================
EXP_NAME="baseline_tdl" 
CONFIG="config/assess/baseline_tdl.yaml"
ASSESS_CONFIG="config/assess/1_PES_cap_base_0.16_4.yaml"
GPU_ID=0
NUM_WORKERS=4
MAX_EPOCHS=32

# for evaluation
EVAL_ASSESS_CONFIG="config/assess/0_eval_PES_unseen_0.16_4.yaml"
CHECKPOINT="/path/to/your/checkpoint.ckpt"

# ==========================================
# 2. train/eval logic
# ==========================================

STAGE=$1

if [ "$STAGE" == "train" ]; then
    echo "Starting [TRAINING] for experiment: $EXP_NAME"
    python ./assess_train.py \
        --exp_name "$EXP_NAME" \
        --config "$CONFIG" \
        --assess_config "$ASSESS_CONFIG" \
        --gpu $GPU_ID \
        --early_stop 4 \
        --validate_interval 2 \
        --max_epochs $MAX_EPOCHS \
        --batch_size 8 \
        --num_workers $NUM_WORKERS \
        --fast_eval

elif [ "$STAGE" == "eval" ]; then
    echo "Starting [EVALUATION] for experiment: $EXP_NAME"
    python ./assess_train.py \
        --exp_name "$EXP_NAME" \
        --config "$CONFIG" \
        --assess_config "$EVAL_ASSESS_CONFIG" \
        --num_workers $NUM_WORKERS \
        --gpu $GPU_ID \
        --test_only \
        --fast_eval \
        --checkpoint "$CHECKPOINT"

else
    echo "Usage: $0 {train|eval}"
    echo "Example: sh $0 train"
    exit 1
fi