#!/bin/bash

# 设置实验名称
EXP_NAME=$1
if [ -z "$EXP_NAME" ]; then
  EXP_NAME="default"
fi

# 获取当前时间戳作为运行ID
TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN_ID="${EXP_NAME}_${TIMESTAMP}"

# 复制配置文件并替换输出目录
cp qwen_lora_dpo_mac_fixed.yaml "qwen_lora_dpo_${RUN_ID}.yaml"
sed -i '' "s|output_dir:.*|output_dir: output/qwen-0.5B/lora/dpo_${RUN_ID}|g" "qwen_lora_dpo_${RUN_ID}.yaml"

# 运行训练
echo "开始训练实验: ${RUN_ID}"
echo "配置文件: qwen_lora_dpo_${RUN_ID}.yaml"
echo "输出目录: output/qwen-0.5B/lora/dpo_${RUN_ID}"

# 执行训练命令
llamafactory-cli train "qwen_lora_dpo_${RUN_ID}.yaml"

# 输出提示信息
echo "训练完成，结果保存在: output/qwen-0.5B/lora/dpo_${RUN_ID}"
echo "可以使用以下命令查看训练日志:"
echo "tensorboard --logdir output/qwen-0.5B/lora/dpo_${RUN_ID}" 