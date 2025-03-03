#!/bin/bash

# ===========================================================================================
# Qwen1.5-0.5B + fooDPO 算法 测试脚本
# ===========================================================================================
#
# 【脚本说明】
# 此脚本用于使用fooDPO算法在小数据集上训练Qwen1.5-0.5B模型
# 适用于macOS MPS设备(Apple Silicon)，数据集很小的情况
#
# 【小数据集适应性调整】
# 1. 减小batch_size和gradient_accumulation_steps，避免过度更新
# 2. 提高learning_rate以加速小数据集上的收敛
# 3. 减少训练epochs，防止过拟合
# 4. 增加 logging_steps 和 eval_steps 频率，便于监控训练进度
# 5. 设置max_samples限制数据集大小用于快速测试
# 6. 禁用group_by_length，确保所有样本都被使用
# 7. 设置dataloader_num_workers=0解决MPS设备多进程问题
#
# 【内存优化措施】
# 1. 减小MAX_SEQ_LEN以降低内存占用
# 2. 启用梯度检查点(gradient_checkpointing)以减少显存使用
# 3. 减小lora_rank参数减少额外参数量
# 4. 使用8位优化器减少优化器状态内存占用
# 5. 设置环境变量PYTORCH_MPS_HIGH_WATERMARK_RATIO控制MPS内存使用
#
# 【wandb配置】
# 使用Weights & Biases进行训练监控
# 运行前确保已经登录: wandb login
#
# 【注意事项】
# - 对于非常小的数据集(<100样本)，可能需要更高的learning_rate和更少的epochs
# - 监控训练过程中的损失曲线，如果出现震荡或无法收敛，考虑调整learning_rate
# - 小数据集训练时wandb可能只有少量数据点，这是正常现象
# ===========================================================================================

# 设置MPS内存使用限制
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 基本配置参数
MODEL_PATH=~/models/Qwen1.5-0.5B
DATASET_PATH="data"
OUTPUT_DIR="output/qwen_foodpo_test"

# 训练参数 - 适合小数据集且优化内存使用
MICRO_BATCH_SIZE=1            # 小批量大小，避免梯度更新过大
GRADIENT_ACCUMULATION_STEPS=1 # 减少梯度累积步数，降低内存压力
LEARNING_RATE=1e-4            # 提高学习率以便在少量数据上快速学习
EPOCHS=1                      # 增加轮次以便在小数据集上充分学习
MAX_SEQ_LEN=512               # 减小序列最大长度以减少内存使用
WARMUP_RATIO=0.1              # 增加预热比例以稳定初期训练

# 设置wandb配置
export WANDB_PROJECT="qwen-foodpo" 
export WANDB_NAME="qwen1.5-0.5B-foodpo-small-data-memory-optimized"

# 清理上次运行的输出目录（如果需要）
if [ -d "$OUTPUT_DIR" ]; then
    echo "清理旧输出目录: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

echo "开始运行内存优化版的fooDPO训练..."

# 运行训练命令 - 使用MPS设备而非CUDA，添加内存优化选项
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
    --max_samples 10 \
    --logging_steps 1 \
    --save_steps 5 \
    --save_total_limit 1 \
    --do_train true \
    --template default \
    --finetuning_type lora \
    --lora_rank 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --stage foodpo \
    --pref_beta 0.1 \
    --fp16 false \
    --use_mps_device true \
    --plot_loss true \
    --report_to wandb \
    --ddp_find_unused_parameters false \
    --eval_strategy steps \
    --eval_steps 5 \
    --group_by_length false \
    --dataloader_num_workers 0 \
    --log_level info \
    --disable_tqdm false \
    --remove_unused_columns false \
    --optim adamw_torch \
    --gradient_checkpointing true

echo "训练完成！检查输出目录: $OUTPUT_DIR" 