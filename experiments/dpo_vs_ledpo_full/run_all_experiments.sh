#!/bin/bash
# DPO vs LEDPO 全部实验一键运行脚本
# 运行Qwen1.5-0.5B和Qwen2.5-1.5B-Instruct两个模型的DPO与LEDPO对比实验

set -e

# 实验根目录与配置
EXPERIMENT_DIR="experiments/dpo_vs_ledpo_full"
WANDB_PROJECT="ledpo_fullscale_experiment"
WANDB_OPTION="--wandb_project $WANDB_PROJECT"

# 解析参数
SKIP_QWEN15=false
SKIP_QWEN25=false
SKIP_DPO=false
SKIP_LEDPO=false

for arg in "$@"
do
    case $arg in
        --skip-qwen15)
        SKIP_QWEN15=true
        shift
        ;;
        --skip-qwen25)
        SKIP_QWEN25=true
        shift
        ;;
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
echo "        DPO vs LEDPO 完整对比实验套件                 "
echo "======================================================"
echo "运行配置:"
echo "- Wandb项目: $WANDB_PROJECT"
echo "- 跳过Qwen1.5: $SKIP_QWEN15"
echo "- 跳过Qwen2.5: $SKIP_QWEN25"
echo "- 跳过标准DPO: $SKIP_DPO"
echo "- 跳过LEDPO: $SKIP_LEDPO"
echo "======================================================"

# 创建所有结果目录
mkdir -p "$EXPERIMENT_DIR/results/dpo"
mkdir -p "$EXPERIMENT_DIR/results/ledpo"
mkdir -p "$EXPERIMENT_DIR/results/dpo_qwen25"
mkdir -p "$EXPERIMENT_DIR/results/ledpo_qwen25"
mkdir -p "$EXPERIMENT_DIR/analysis"
mkdir -p "$EXPERIMENT_DIR/analysis_qwen25"

# ====== Qwen1.5-0.5B实验 ======
if [ "$SKIP_QWEN15" = false ]; then
    echo "======================================================"
    echo "          开始运行 Qwen1.5-0.5B 模型实验             "
    echo "======================================================"
    
    # 运行标准DPO实验
    if [ "$SKIP_DPO" = false ]; then
        echo "开始运行Qwen1.5-0.5B标准DPO实验..."
        python ledpo_progressive_dev/run_train_and_plot.py --config "$EXPERIMENT_DIR/dpo_config.yaml" $WANDB_OPTION
        echo "Qwen1.5-0.5B标准DPO实验完成!"
    else
        echo "已跳过Qwen1.5-0.5B标准DPO实验"
    fi
    
    # 运行LEDPO实验
    if [ "$SKIP_LEDPO" = false ]; then
        echo "开始运行Qwen1.5-0.5B LEDPO实验..."
        python ledpo_progressive_dev/run_train_and_plot.py --config "$EXPERIMENT_DIR/ledpo_config.yaml" $WANDB_OPTION
        echo "Qwen1.5-0.5B LEDPO实验完成!"
    else
        echo "已跳过Qwen1.5-0.5B LEDPO实验"
    fi
    
    # 分析Qwen1.5-0.5B实验结果
    if [ "$SKIP_DPO" = false ] && [ "$SKIP_LEDPO" = false ]; then
        echo "开始分析Qwen1.5-0.5B实验结果..."
        python "$EXPERIMENT_DIR/analyze_logs.py" \
            --dpo_dir "$EXPERIMENT_DIR/results/dpo" \
            --ledpo_dir "$EXPERIMENT_DIR/results/ledpo" \
            --output_dir "$EXPERIMENT_DIR/analysis"
        echo "Qwen1.5-0.5B实验分析完成!"
    fi
else
    echo "已跳过所有Qwen1.5-0.5B实验"
fi

# ====== Qwen2.5-1.5B-Instruct实验 ======
if [ "$SKIP_QWEN25" = false ]; then
    echo "======================================================"
    echo "        开始运行 Qwen2.5-1.5B-Instruct 模型实验      "
    echo "======================================================"
    
    # 运行标准DPO实验
    if [ "$SKIP_DPO" = false ]; then
        echo "开始运行Qwen2.5-1.5B-Instruct标准DPO实验..."
        python ledpo_progressive_dev/run_train_and_plot.py --config "$EXPERIMENT_DIR/dpo_qwen25_config.yaml" $WANDB_OPTION
        echo "Qwen2.5-1.5B-Instruct标准DPO实验完成!"
    else
        echo "已跳过Qwen2.5-1.5B-Instruct标准DPO实验"
    fi
    
    # 运行LEDPO实验
    if [ "$SKIP_LEDPO" = false ]; then
        echo "开始运行Qwen2.5-1.5B-Instruct LEDPO实验..."
        python ledpo_progressive_dev/run_train_and_plot.py --config "$EXPERIMENT_DIR/ledpo_qwen25_config.yaml" $WANDB_OPTION
        echo "Qwen2.5-1.5B-Instruct LEDPO实验完成!"
    else
        echo "已跳过Qwen2.5-1.5B-Instruct LEDPO实验"
    fi
    
    # 分析Qwen2.5-1.5B-Instruct实验结果
    if [ "$SKIP_DPO" = false ] && [ "$SKIP_LEDPO" = false ]; then
        echo "开始分析Qwen2.5-1.5B-Instruct实验结果..."
        python "$EXPERIMENT_DIR/analyze_logs.py" \
            --dpo_dir "$EXPERIMENT_DIR/results/dpo_qwen25" \
            --ledpo_dir "$EXPERIMENT_DIR/results/ledpo_qwen25" \
            --output_dir "$EXPERIMENT_DIR/analysis_qwen25"
        echo "Qwen2.5-1.5B-Instruct实验分析完成!"
    fi
else
    echo "已跳过所有Qwen2.5-1.5B-Instruct实验"
fi

echo "======================================================"
echo "        DPO vs LEDPO 完整对比实验套件完成             "
echo "======================================================" 