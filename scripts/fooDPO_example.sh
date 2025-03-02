#!/bin/bash

# 使用fooDPO算法进行训练的示例脚本
# 此脚本基于DPO算法改编，仅将训练类型改为fooDPO

# 配置参数
MODEL_PATH="llama2-7b"           # 模型路径
DATASET_PATH="rm_data"           # 数据集路径
DATASET_NAME="comparison_gpt4_en"  # 数据集名称
OUTPUT_DIR="output/fooDPO_test"    # 输出目录

# 基本训练参数
MICRO_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
EPOCHS=3
MAX_SEQ_LEN=2048
WARMUP_RATIO=0.03

# 运行训练
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --model_name_or_path $MODEL_PATH \
    --dataset_dir $DATASET_PATH \
    --dataset $DATASET_NAME \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --cutoff_len $MAX_SEQ_LEN \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $EPOCHS \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type cosine \
    --stage foodpo \
    --pref_beta 0.1 \
    --per_device_eval_batch_size 2 \
    --gradient_checkpointing \
    --bf16

# 注意：此脚本与DPO训练完全相同，只是把stage参数改为"foodpo"
# 可以根据需要调整参数 