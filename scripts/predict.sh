#!/bin/bash

set -xe

# Modify the following variables according to your settings
PRETRAINED_MODEL_NAME_OR_PATH=
CHECKPOINT_PATH=
OUTPUT_DIR=

accelerate launch \
    src/train_bash.py \
    --stage sft \
    --model_name_or_path ${PRETRAINED_MODEL_NAME_OR_PATH} \
    --checkpoint_dir ${CHECKPOINT_PATH} \
    --do_predict \
    --dataset test \
    --template mistral \
    --cutoff_len 1536 \
    --finetuning_type lora \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --max_samples 1000000 \
    --predict_with_generate \
    --do_sample False \
    --fp16
