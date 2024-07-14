#!/bin/bash

set -xe

# Modify the following variables according to your settings
PRETRAINED_MODEL_NAME_OR_PATH=
OUTPUT_DIR=

# global_batch_size=64

accelerate launch \
    src/train_bash.py \
    --stage rm \
    --model_name_or_path ${PRETRAINED_MODEL_NAME_OR_PATH} \
    --do_train \
    --dataset rllr_reward_train \
    --template mistral \
    --cutoff_len 1536 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --resume_lora_training False \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
