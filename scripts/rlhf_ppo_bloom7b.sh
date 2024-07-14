#!/bin/bash

set -xe

# Modify the following variables according to your settings
PRETRAINED_MODEL_NAME_OR_PATH=
REWARD_MODEL_PATH=
OUTPUT_DIR=

# global_batch_size=32

accelerate launch \
    src/train_bash.py \
    --stage ppo \
    --model_name_or_path ${PRETRAINED_MODEL_NAME_OR_PATH} \
    --do_train \
    --dataset unsupervised \
    --template default \
    --cutoff_len 1536 \
    --finetuning_type lora \
    --lora_target query_key_value \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --resume_lora_training False \
    --reward_model ${REWARD_MODEL_PATH} \
    --value_model ${REWARD_MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 4 \
    --ppo_mini_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-6 \
    --num_train_epochs 1.0 \
    --fp16 \
    --plot_loss \
    --max_new_tokens 256 \
    --do_sample True \
    --temperature 1.0 \
    --top_p 1.0 \
    --ppo_score_norm True \
    --ppo_use_separate_value_model True \
    --ppo_logger tensorboard
