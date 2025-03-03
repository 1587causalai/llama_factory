#!/bin/bash
# 比较测试脚本 - 使用不同的beta策略测试BetaDPO算法

# 基础配置
MODEL_PATH=~/models/Qwen1.5-0.5B
DATA_PATH=data/dpo_zh_demo.json
BASE_OUTPUT_DIR="output/betadpo_strategies_compare"

# 确保输出目录存在
mkdir -p $BASE_OUTPUT_DIR

# 记录开始时间
echo "开始BetaDPO策略比较实验: $(date)" | tee $BASE_OUTPUT_DIR/comparison_log.txt

# BetaDPO基础参数
COMMON_ARGS=(
    --model_name_or_path $MODEL_PATH
    --do_train
    --dataset $DATA_PATH
    --finetuning_type lora
    --lora_target all
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 4
    --lr_scheduler_type cosine
    --logging_steps 10
    --save_steps 50
    --learning_rate 5e-5
    --num_train_epochs 2.0
    --plot_loss
    --fp16 false
    --max_grad_norm 1.0
    --seed 42
    --stage betadpo
    --pref_beta 0.3
)

# 测试不同的beta策略
STRATEGIES=("constant" "adaptive" "exponential" "cosine")

for strategy in "${STRATEGIES[@]}"; do
    OUTPUT_DIR="$BASE_OUTPUT_DIR/${strategy}"
    mkdir -p $OUTPUT_DIR
    
    echo "开始测试 $strategy 策略: $(date)" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
    
    # 运行训练，根据策略设置参数
    if [ "$strategy" == "constant" ]; then
        # 对于constant策略，只使用默认的pref_beta
        EXTRA_ARGS=(
            --beta_strategy constant
        )
    else
        # 对于其他策略，设置beta范围
        EXTRA_ARGS=(
            --beta_strategy $strategy
            --beta_min 0.1
            --beta_max 5.0
        )
    fi
    
    # 执行训练
    CUDA_VISIBLE_DEVICES=-1 python src/llamafactory/train.py \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        --output_dir $OUTPUT_DIR \
        2>&1 | tee -a $OUTPUT_DIR/training_log.txt
    
    echo "完成 $strategy 策略测试: $(date)" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
    echo "--------------------------------" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
done

echo "所有BetaDPO策略比较实验完成: $(date)" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt

# 收集结果和比较
echo "生成结果比较报告..." | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
python -c '
import os
import json
import matplotlib.pyplot as plt
import numpy as np

base_dir = "output/betadpo_strategies_compare"
strategies = ["constant", "adaptive", "exponential", "cosine"]
results = {}

# 收集训练损失和奖励准确率
for strategy in strategies:
    strategy_dir = os.path.join(base_dir, strategy)
    train_file = os.path.join(strategy_dir, "trainer_state.json")
    if os.path.exists(train_file):
        with open(train_file, "r") as f:
            data = json.load(f)
            
            # 提取loss和rewards/accuracies
            steps = [log["step"] for log in data["log_history"] if "loss" in log]
            loss = [log["loss"] for log in data["log_history"] if "loss" in log]
            rewards_acc = [log.get("rewards/accuracies", None) for log in data["log_history"] if "loss" in log]
            
            results[strategy] = {
                "steps": steps,
                "loss": loss,
                "rewards_acc": rewards_acc
            }

# 绘制损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for strategy in strategies:
    if strategy in results:
        plt.plot(results[strategy]["steps"], results[strategy]["loss"], label=strategy)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)

# 绘制奖励准确率曲线
plt.subplot(1, 2, 2)
for strategy in strategies:
    if strategy in results:
        # 过滤掉None值
        steps = []
        acc = []
        for i, val in enumerate(results[strategy]["rewards_acc"]):
            if val is not None:
                steps.append(results[strategy]["steps"][i])
                acc.append(val)
        if steps and acc:
            plt.plot(steps, acc, label=strategy)
plt.xlabel("Steps")
plt.ylabel("Reward Accuracy")
plt.title("Reward Accuracy Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "comparison_results.png"))
print(f"保存了比较结果到: {os.path.join(base_dir, \"comparison_results.png\")}")
' | tee -a $BASE_OUTPUT_DIR/comparison_log.txt

echo "实验结果保存在: $BASE_OUTPUT_DIR/comparison_results.png" 