#!/bin/bash

set -xe

# Modify the following variables according to your settings
PRETRAINED_MODEL_NAME_OR_PATH=
RLHF_REWARD_MODEL_PATH=
RLLR_REWARD_MODEL_PATH=
OUTPUT_DIR=

# global_batch_size=16

accelerate launch \
    src/train_bash.py \
    --stage ppo \
    --model_name_or_path ${PRETRAINED_MODEL_NAME_OR_PATH} \
    --do_train \
    --dataset unsupervised \
    --template llama2 \
    --cutoff_len 1536 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --resume_lora_training False \
    --reward_model_rationale ${RLHF_REWARD_MODEL_PATH} \
    --reward_model ${RLLR_REWARD_MODEL_PATH} \
    --value_model ${RLLR_REWARD_MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 2 \
    --ppo_mini_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-6 \
    --num_train_epochs 1.0 \
    --label_reward_threshold 3.9 \
    --plot_loss \
    --max_new_tokens 256 \
    --do_sample True \
    --temperature 1.0 \
    --top_p 1.0 \
    --ppo_score_norm True \
    --ppo_use_separate_value_model True \
    --ppo_logger tensorboard
