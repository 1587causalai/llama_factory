#!/bin/bash
# 标准DPO与LEDPO对比实验运行脚本
# 运行命令: bash experiments/dpo_vs_ledpo_demo/run_experiment.sh

set -e  # 遇到错误立即退出

EXPERIMENT_DIR=$(dirname "$0")
cd $(git rev-parse --show-toplevel)  # 切换到项目根目录

# 创建结果目录
mkdir -p $EXPERIMENT_DIR/results/dpo
mkdir -p $EXPERIMENT_DIR/results/ledpo
mkdir -p $EXPERIMENT_DIR/analysis

# 设置wandb项目名称
WANDB_PROJECT="dpo_vs_ledpo"

# 解析命令行参数
SKIP_DPO=0
SKIP_LEDPO=0

for arg in "$@"; do
  case $arg in
    --skip-dpo)
      SKIP_DPO=1
      shift
      ;;
    --skip-ledpo)
      SKIP_LEDPO=1
      shift
      ;;
  esac
done

# 运行标准DPO实验
if [ $SKIP_DPO -eq 0 ]; then
  echo "===== 开始标准DPO实验 ====="
  python ledpo_progressive_dev/run_train_and_plot.py --config $EXPERIMENT_DIR/dpo_config.yaml --wandb_project $WANDB_PROJECT
else
  echo "===== 跳过标准DPO实验 ====="
fi

# 运行LEDPO实验
if [ $SKIP_LEDPO -eq 0 ]; then
  echo "===== 开始LEDPO实验 ====="
  python ledpo_progressive_dev/run_train_and_plot.py --config $EXPERIMENT_DIR/ledpo_config.yaml --wandb_project $WANDB_PROJECT
else
  echo "===== 跳过LEDPO实验 ====="
fi

# 运行结果分析脚本
echo "===== 分析实验结果 ====="
python $EXPERIMENT_DIR/analyze_logs.py \
  --dpo_dir $EXPERIMENT_DIR/results/dpo \
  --ledpo_dir $EXPERIMENT_DIR/results/ledpo \
  --output_dir $EXPERIMENT_DIR/analysis

echo "===== 实验完成! ====="
echo "标准DPO结果: $EXPERIMENT_DIR/results/dpo"
echo "LEDPO结果: $EXPERIMENT_DIR/results/ledpo"
echo "分析结果: $EXPERIMENT_DIR/analysis" 