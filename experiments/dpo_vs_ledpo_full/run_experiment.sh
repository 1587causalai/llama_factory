#!/bin/bash
# DPO vs LEDPO 完整实验运行脚本

set -e

# 实验根目录
EXPERIMENT_DIR="experiments/dpo_vs_ledpo_full"
# 解析参数
SKIP_DPO=false
SKIP_LEDPO=false

for arg in "$@"
do
    case $arg in
        --skip-dpo)
        SKIP_DPO=true
        shift
        ;;
        --skip-ledpo)
        SKIP_LEDPO=true
        shift
        ;;
        *)
        # 未知参数
        shift
        ;;
    esac
done

# 显示实验信息
echo "======================================================"
echo "        DPO vs LEDPO 完整对比实验                     "
echo "======================================================"
echo "运行配置:"
echo "- 跳过标准DPO: $SKIP_DPO"
echo "- 跳过LEDPO: $SKIP_LEDPO"
echo "- 实验目录: $EXPERIMENT_DIR"
echo "======================================================"

# 创建结果目录
mkdir -p "$EXPERIMENT_DIR/results/dpo"
mkdir -p "$EXPERIMENT_DIR/results/ledpo"
mkdir -p "$EXPERIMENT_DIR/analysis"
mkdir -p "$EXPERIMENT_DIR/samples"

# 运行标准DPO实验
if [ "$SKIP_DPO" = false ]; then
    echo "开始运行标准DPO实验..."
    
    llamafactory-cli train "$EXPERIMENT_DIR/dpo_config.yaml"
    
    echo "标准DPO实验完成!"
else
    echo "已跳过标准DPO实验"
fi

# 运行LEDPO实验
if [ "$SKIP_LEDPO" = false ]; then
    echo "开始运行LEDPO实验..."
    
    llamafactory-cli train "$EXPERIMENT_DIR/ledpo_config.yaml"
    
    echo "LEDPO实验完成!"
else
    echo "已跳过LEDPO实验"
fi

# 若两个实验都运行，则进行分析
if [ "$SKIP_DPO" = false ] && [ "$SKIP_LEDPO" = false ]; then
    echo "开始分析实验结果..."
    
    # 分析日志，生成报告
    python "$EXPERIMENT_DIR/analyze_logs.py" \
        --dpo_dir "$EXPERIMENT_DIR/results/dpo" \
        --ledpo_dir "$EXPERIMENT_DIR/results/ledpo" \
        --output_dir "$EXPERIMENT_DIR/analysis"
    
    echo "实验分析完成!"
fi

# 创建样本生成
if [ "$SKIP_DPO" = false ] || [ "$SKIP_LEDPO" = false ]; then
    echo "生成模型回答样本..."
    
    # 生成样本脚本
    python "$EXPERIMENT_DIR/generate_samples.py" \
        --dpo_dir "$EXPERIMENT_DIR/results/dpo" \
        --ledpo_dir "$EXPERIMENT_DIR/results/ledpo" \
        --output_dir "$EXPERIMENT_DIR/samples" \
        --skip_dpo "$SKIP_DPO" \
        --skip_ledpo "$SKIP_LEDPO"
    
    echo "样本生成完成!"
fi

echo "======================================================"
echo "        DPO vs LEDPO 完整对比实验完成                 "
echo "======================================================" 