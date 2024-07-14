#!/bin/bash

set -xe

# Modify the following variables according to your settings
PRETRAINED_MODEL_NAME_OR_PATH=
OUTPUT_DIR=

# global_batch_size=128

accelerate launch \
    src/train_bash.py \
    --stage sft \
    --model_name_or_path ${PRETRAINED_MODEL_NAME_OR_PATH} \
    --do_train \
    --dataset sft_with_rat_train \
    --template baichuan2 \
    --cutoff_len 512 \
    --finetuning_type lora \
    --lora_target W_pack \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --resume_lora_training False \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 300 \
    --learning_rate 1e-3 \
    --num_train_epochs 10.0 \
    --fp16 \
    --plot_loss
