#!/bin/bash

# 测试脚本 - 使用Qwen1.5-0.5B模型和dpo_zh_demo数据集测试动态β版本的fooDPO算法
# 此脚本使用本地的Qwen1.5-0.5B模型进行测试，适用于macOS (MPS)
# 动态β版本: β(x) = c · log(PPL(x)) · β，其中c通过pref_beta_scale参数控制

# 基本配置参数
MODEL_PATH=~/models/Qwen1.5-0.5B
DATASET_PATH="data"
OUTPUT_DIR="output/qwen_dynamic_foodpo_test"

# 基本训练参数
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-5
EPOCHS=1
MAX_SEQ_LEN=1024
WARMUP_RATIO=0.03

# FooDPO特定参数
BETA=0.1           # 基础β值
BETA_SCALE=1.0     # β(x)公式中的c系数，控制困惑度对β的影响程度

# 运行训练命令 - 使用MPS设备而非CUDA
llamafactory-cli train \
    --model_name_or_path $MODEL_PATH \
    --dataset "dpo_zh_demo" \
    --dataset_dir $DATASET_PATH \
    --eval_dataset "dpo_zh_demo" \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --per_device_eval_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --cutoff_len $MAX_SEQ_LEN \
    --warmup_ratio $WARMUP_RATIO \
    --max_samples 20 \
    --logging_steps 5 \
    --save_steps 20 \
    --save_total_limit 1 \
    --do_train true \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --stage foodpo \
    --pref_beta $BETA \
    --pref_beta_scale $BETA_SCALE \
    --fp16 false \
    --use_mps_device true \
    --report_to none 