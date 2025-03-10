#!/bin/bash

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 激活环境
source ~/.bashrc
conda activate llama

# 进入项目目录
cd /root/LLaMA-Factory

# 1. 只训练 ValueHead 的实验
echo "========= 开始实验 1: 只训练 ValueHead ========="
python src/train_bash.py \
    --config experiments/valuehead_only_training/qwen1_5_0_5b_valuehead_only.yaml \
    --deepspeed deepspeed/zero2.json

# 2. 只训练 Policy 的实验
echo "========= 开始实验 2: 只训练 Policy ========="
python src/train_bash.py \
    --config experiments/valuehead_only_training/qwen1_5_0_5b_policy_only.yaml \
    --deepspeed deepspeed/zero2.json

# 3. 正常训练两者的实验
echo "========= 开始实验 3: 正常训练 Policy 和 ValueHead ========="
python src/train_bash.py \
    --config experiments/valuehead_only_training/qwen1_5_0_5b_normal_training.yaml \
    --deepspeed deepspeed/zero2.json

echo "所有实验完成！" 