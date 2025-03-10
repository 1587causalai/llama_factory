#!/bin/bash

# 确保当前所在目录是 LLaMA-Factory 的根目录
cd $(dirname $0)/../../

# 不要使用 conda activate，因为这在非交互环境可能不工作
# 直接使用绝对路径执行 python
python=/root/.conda/envs/llama/bin/python

# 设置 WANDB
export WANDB_PROJECT="neural_ledpo"

# 运行实验 - value_head only
CUDA_VISIBLE_DEVICES=0 $python src/train.py \
    experiments/valuehead_only/qwen1_5_0_5b_valuehead_only.yaml

echo "只训练 value_head 的实验已完成"