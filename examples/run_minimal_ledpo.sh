#!/bin/bash

# 使用Qwen1.5-0.5B模型进行LEDPO测试
MODEL_PATH="/home/models/Qwen1.5-0.5B"
OUTPUT_DIR="saves/qwen1.5-0.5b/lora/ledpo_baseline_n1000_epoch1.0"

# 使用conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llama

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 运行测试脚本
CUDA_VISIBLE_DEVICES=0 python examples/minimal_ledpo_test.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_samples 1000 \
    --batch_size 2 \
    --beta 0.1 \
    --value_head_lr 1e-3 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_length 512 \
    --max_prompt_length 256 \
    --use_lora

echo "LEDPO 测试完成！" 