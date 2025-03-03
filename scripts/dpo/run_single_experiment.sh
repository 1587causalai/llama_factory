#!/bin/bash

# 脚本位置
SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")
cd "$ROOT_DIR"

# 运行单个DPO实验
echo "========================================================"
echo "         开始运行Qwen1.5-0.5B的DPO单个实验"
echo "========================================================"

# 运行DPO实验（可选参数：实验名称、beta值）
./scripts/dpo/run_qwen_dpo_beta.sh "single_run" "0.1"

# ./scripts/dpo/run_qwen_dpo_beta.sh "my_experiment" "0.01"


echo "实验完成！" 