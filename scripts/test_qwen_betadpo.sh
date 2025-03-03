#!/bin/bash
# 测试脚本 - 使用Qwen1.5-0.5B模型和dpo_zh_demo数据集测试BetaDPO算法

# 基础配置
MODEL_PATH=~/models/Qwen1.5-0.5B
DATA_PATH=data/dpo_zh_demo.json
OUTPUT_DIR="output/qwen_betadpo_test"

# 设置日志记录
mkdir -p $OUTPUT_DIR
exec &> >(tee -a "$OUTPUT_DIR/training_log.txt")

echo "开始BetaDPO训练任务: $(date)"
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"

# BetaDPO特定参数
echo "BetaDPO训练参数:"
echo "- beta策略: adaptive"
echo "- beta范围: 0.1-5.0"

# 运行训练
CUDA_VISIBLE_DEVICES=-1 python src/llamafactory/train.py \
    --model_name_or_path $MODEL_PATH \
    --do_train \
    --dataset $DATA_PATH \
    --finetuning_type lora \
    --lora_target all \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 false \
    --max_grad_norm 1.0 \
    --seed 42 \
    --stage betadpo \
    --beta_strategy adaptive \
    --beta_min 0.1 \
    --beta_max 5.0 \
    --pref_beta 0.3 \
    2>&1

echo "BetaDPO训练完成: $(date)"

# 运行简单的推理测试
echo "运行推理测试..."
python scripts/test_inference.py 