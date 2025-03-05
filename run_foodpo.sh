#!/bin/bash
# 设置wandb项目和运行名称
export WANDB_PROJECT="foodpo"
export WANDB_NAME="qwen1.5-0.5b-foodpo-test"

# 运行FooDPO训练
python dpo_baseline/run_foodpo_detailed.py examples/train_lora/qwen1_5_0_5b_lora_foodpo.yaml 