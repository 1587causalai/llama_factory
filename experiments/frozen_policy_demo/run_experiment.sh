#!/bin/bash
# 冻结与非冻结策略模型对比实验运行脚本

set -e  # 遇到错误立即退出

EXPERIMENT_DIR=$(dirname "$0")
cd $(git rev-parse --show-toplevel)  # 切换到项目根目录

echo "===== 开始冻结策略模型实验 ====="
python ledpo_progressive_dev/run_train_and_plot.py --config $EXPERIMENT_DIR/frozen_config.yaml --wandb_project frozen_policy_demo

echo "===== 开始非冻结策略模型实验 ====="
python ledpo_progressive_dev/run_train_and_plot.py --config $EXPERIMENT_DIR/unfrozen_config.yaml --wandb_project frozen_policy_demo

# 运行结果对比脚本
echo "===== 生成实验结果对比图表 ====="
python $EXPERIMENT_DIR/compare_results.py

echo "===== 实验完成! ====="
echo "冻结模型结果: $EXPERIMENT_DIR/results/frozen"
echo "非冻结模型结果: $EXPERIMENT_DIR/results/unfrozen"
echo "对比结果: $EXPERIMENT_DIR/comparison" 