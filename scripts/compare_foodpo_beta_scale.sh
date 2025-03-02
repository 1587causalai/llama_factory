#!/bin/bash

# 比较测试脚本 - 使用不同的β比例值测试动态β版本的fooDPO算法
# 此脚本运行多个不同的β比例值配置，以比较其效果

# 基本配置参数
MODEL_PATH=~/models/Qwen1.5-0.5B
DATASET_PATH="data"
BASE_OUTPUT_DIR="output/foodpo_beta_scale_compare"

# 基本训练参数
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-5
EPOCHS=1
MAX_SEQ_LEN=1024
WARMUP_RATIO=0.03
MAX_SAMPLES=20

# FooDPO基础参数
BETA=0.1
BETA_SCALE=0.1

# 创建比较实验目录
mkdir -p "$BASE_OUTPUT_DIR"

# 测试不同的β比例值
run_test() {
    local scale=$1
    local output_dir="${BASE_OUTPUT_DIR}/scale_${scale}"
    
    echo "====================================================="
    echo "开始测试 β比例值 = $scale"
    echo "输出目录: $output_dir"
    echo "====================================================="
    
    llamafactory-cli train \
        --model_name_or_path $MODEL_PATH \
        --dataset "dpo_zh_demo" \
        --dataset_dir $DATASET_PATH \
        --eval_dataset "dpo_zh_demo" \
        --output_dir $output_dir \
        --per_device_train_batch_size $MICRO_BATCH_SIZE \
        --per_device_eval_batch_size $MICRO_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $EPOCHS \
        --cutoff_len $MAX_SEQ_LEN \
        --warmup_ratio $WARMUP_RATIO \
        --max_samples $MAX_SAMPLES \
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
        --pref_beta_scale $scale \
        --fp16 false \
        --use_mps_device true \
        --report_to none
    
    echo "测试完成: β比例值 = $scale"
    echo ""
}

# 运行多个比例值的测试
run_test 0.0   # 相当于标准DPO
run_test 0.5   # 中等影响
run_test 1.0   # 正常影响
run_test 2.0   # 加强影响

echo "所有测试完成！结果保存在 $BASE_OUTPUT_DIR" 