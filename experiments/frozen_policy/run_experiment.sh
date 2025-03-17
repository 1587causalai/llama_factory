#!/bin/bash
# 冻结与非冻结策略模型正式对比实验运行脚本
# 运行命令: bash experiments/frozen_policy/run_experiment.sh

set -e  # 遇到错误立即退出

EXPERIMENT_DIR=$(dirname "$0")
cd $(git rev-parse --show-toplevel)  # 切换到项目根目录

# 创建结果目录
mkdir -p $EXPERIMENT_DIR/results/frozen
mkdir -p $EXPERIMENT_DIR/results/frozen_diff_ref
mkdir -p $EXPERIMENT_DIR/results/unfrozen
mkdir -p $EXPERIMENT_DIR/analysis

# 设置wandb项目名称
WANDB_PROJECT="ledpo_frozen_policy"

echo "===== 开始冻结策略模型实验 ====="
python ledpo_progressive_dev/run_train_and_plot.py --config $EXPERIMENT_DIR/frozen_config.yaml --wandb_project $WANDB_PROJECT

echo "===== 开始冻结策略模型-新ref模型实验 ====="
python ledpo_progressive_dev/run_train_and_plot.py --config $EXPERIMENT_DIR/frozen_config_diff_ref.yaml --wandb_project $WANDB_PROJECT

echo "===== 开始非冻结策略模型实验 ====="
python ledpo_progressive_dev/run_train_and_plot.py --config $EXPERIMENT_DIR/unfrozen_config.yaml --wandb_project $WANDB_PROJECT

# 运行结果分析脚本
echo "===== 分析实验结果 ====="
python $EXPERIMENT_DIR/analyze_logs.py \
  --frozen_dir $EXPERIMENT_DIR/results/frozen \
  --frozen_diff_ref_dir $EXPERIMENT_DIR/results/frozen_diff_ref \
  --unfrozen_dir $EXPERIMENT_DIR/results/unfrozen \
  --output_dir $EXPERIMENT_DIR/analysis

echo "===== 实验完成! ====="
echo "冻结模型结果: $EXPERIMENT_DIR/results/frozen"
echo "冻结模型-新ref模型结果: $EXPERIMENT_DIR/results/frozen_diff_ref"
echo "非冻结模型结果: $EXPERIMENT_DIR/results/unfrozen"
echo "分析结果: $EXPERIMENT_DIR/analysis" 