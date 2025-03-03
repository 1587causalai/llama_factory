#!/bin/bash

# 设置实验名称和beta值
EXP_NAME=$1
BETA_VALUE=$2

# 检查参数
if [ -z "$EXP_NAME" ]; then
  EXP_NAME="default"
  echo "未提供实验名称，使用默认名称: $EXP_NAME"
fi

if [ -z "$BETA_VALUE" ]; then
  BETA_VALUE="0.1"
  echo "未提供beta值，使用默认值: $BETA_VALUE"
fi

# 获取当前时间戳作为运行ID
TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN_ID="${EXP_NAME}_beta${BETA_VALUE}_${TIMESTAMP}"

# 创建配置文件
CONFIG_FILE="configs/dpo/qwen_dpo_${RUN_ID}.yaml"
cp configs/dpo/qwen_lora_dpo_mac_fixed.yaml "$CONFIG_FILE"

# 替换配置中的值
sed -i '' "s|output_dir:.*|output_dir: output/qwen-0.5B/lora/dpo_${RUN_ID}|g" "$CONFIG_FILE"
sed -i '' "s|pref_beta:.*|pref_beta: ${BETA_VALUE}  # 已设置beta值|g" "$CONFIG_FILE"
sed -i '' "s|run_name:.*|run_name: ${EXP_NAME}_beta_${BETA_VALUE}|g" "$CONFIG_FILE"

# 运行训练前，确保已登录wandb（可选）
if command -v wandb &>/dev/null; then
  echo "正在验证wandb登录状态..."
  wandb status > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "请先登录Weights & Biases:"
    wandb login
  fi
  export WANDB_PROJECT="qwen_dpo"
  export WANDB_NAME="${EXP_NAME}_beta_${BETA_VALUE}"
fi

# 运行训练
echo "==============================================="
echo "开始DPO训练实验"
echo "实验名称: ${EXP_NAME}"
echo "Beta值: ${BETA_VALUE}"
echo "配置文件: ${CONFIG_FILE}"
echo "输出目录: output/qwen-0.5B/lora/dpo_${RUN_ID}"
echo "==============================================="

# 执行训练命令
llamafactory-cli train "$CONFIG_FILE"

# 输出提示信息
echo "==============================================="
echo "训练完成，结果保存在: output/qwen-0.5B/lora/dpo_${RUN_ID}"
if command -v wandb &>/dev/null; then
  echo "可以在W&B界面查看训练进度和结果: https://wandb.ai"
fi
echo "===============================================" 