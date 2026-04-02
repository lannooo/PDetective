#!/bin/bash

# ==========================================
# 1. basic configurations
# ==========================================
EXP_NAME="PDetective"
CONFIG="config/model/PDetective.yaml"
ASSESS_CONFIG="config/assess/1_PES_cap_aug_all_0.16_4.yaml"
GPU_ID=0
# increase num_workers carefully, may cause memory issue when enable data augmentation (DiffWave or WaveGlow)
NUM_WORKERS=1
MAX_EPOCHS=8

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
        --validate_interval 1 \
        --max_epochs $MAX_EPOCHS \
        --batch_size 8 \
        --num_workers $NUM_WORKERS \

elif [ "$STAGE" == "eval" ]; then
    echo "Starting [EVALUATION] for experiment: $EXP_NAME"
    python ./assess_train.py \
        --exp_name "$EXP_NAME" \
        --config "$CONFIG" \
        --assess_config "$EVAL_ASSESS_CONFIG" \
        --num_workers $NUM_WORKERS \
        --gpu $GPU_ID \
        --test_only \
        --checkpoint "$CHECKPOINT"

else
    echo "Usage: $0 {train|eval}"
    echo "Example: sh $0 train"
    exit 1
fi