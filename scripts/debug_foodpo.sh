#!/bin/bash

# 调试脚本 - 用于解决fooDPO训练过程中的问题
# 此脚本添加了更多的调试信息和简化的配置

# 基本配置参数
MODEL_PATH=~/models/Qwen1.5-0.5B
DATASET_PATH="data"
OUTPUT_DIR="output/qwen_foodpo_debug"

# 基本训练参数 - 最小化配置以便调试
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=5e-5
EPOCHS=1
MAX_SEQ_LEN=512  # 减小长度加快处理
WARMUP_RATIO=0.03

# 设置更详细的日志
export LLAMAFACTORY_VERBOSE=1  # 启用详细日志
export TRANSFORMERS_VERBOSITY=info
export WANDB_DISABLED=true  # 暂时禁用wandb，减少干扰

echo "====== 开始调试fooDPO训练 ======"
echo "模型路径: $MODEL_PATH"
echo "数据集路径: $DATASET_PATH"
echo "输出目录: $OUTPUT_DIR"

# 清理可能存在的旧检查点
if [ -d "$OUTPUT_DIR/checkpoint-10" ]; then
    echo "删除旧检查点..."
    rm -rf "$OUTPUT_DIR/checkpoint-10"
fi

# 运行训练命令 - 使用更多调试参数
llamafactory-cli train \
    --model_name_or_path $MODEL_PATH \
    --dataset "dpo_zh_demo" \
    --dataset_dir $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --cutoff_len $MAX_SEQ_LEN \
    --warmup_ratio $WARMUP_RATIO \
    --max_samples 10 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 1 \
    --do_train true \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --stage foodpo \
    --pref_beta 0.1 \
    --fp16 false \
    --use_mps_device true \
    --plot_loss true \
    --ddp_find_unused_parameters false \
    --group_by_length false \
    --dataloader_num_workers 0 \
    --log_level detail \
    --disable_tqdm false \
    --optim adamw_torch \
    --remove_unused_columns false \
    --overwrite_output_dir

echo "====== 调试完成 ======"
echo "请检查输出日志中的错误信息" 