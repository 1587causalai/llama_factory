#!/bin/bash

# ===================================================
# pref_beta 实验脚本 (run_pref_beta_experiments.sh)
# ===================================================
#
# 本脚本用于测试不同pref_beta值对DPO实验结果的影响
# 默认测试值：0.05, 0.1, 0.5, 1.0, 2.0
#
# 使用方法：
# 1. 添加执行权限：chmod +x run_pref_beta_experiments.sh
# 2. 默认运行：./run_pref_beta_experiments.sh
# 3. 自定义参数：./run_pref_beta_experiments.sh --model /path/to/model --dataset custom_dataset
#
# 示例:
# # 使用默认参数运行
# ./xdpo/experiments/run_pref_beta_experiments.sh
#
# # 指定模型和数据集
# ./xdpo/experiments/run_pref_beta_experiments.sh -m /root/models/Qwen1.5-1.8B -d hh_rlhf_zh
#
# # 同时运行实验1和实验2
# ./xdpo/experiments/run_pref_beta_experiments.sh --exps "1 2"
#
# # 完整自定义
# ./xdpo/experiments/run_pref_beta_experiments.sh -m /root/models/Qwen1.5-0.5B -d hh_rlhf_en -p 2.0 --exps "1 2" -o custom_results --no-timestamp
#
# 可用选项：
# -m, --model PATH        设置模型路径 (默认: /root/models/Qwen1.5-0.5B)
# -d, --dataset NAME      设置训练数据集 (默认: hh_rlhf_en)
# -p, --epochs NUM        设置训练轮数 (默认: 1.0)
# -o, --output PATH       设置输出目录基础路径 (默认: results/pref_beta_experiments)
# --exps "NUM1 NUM2..."   设置要运行的实验类型，空格分隔 (默认: "1 2")
# -t, --no-timestamp      禁用时间戳
# -h, --help              显示帮助信息
# ===================================================

USE_TIMESTAMP=true

# 定义要测试的 pref_beta 值数组
BETA_VALUES=(0.05 0.1 0.5 1.0 2.0)

# 默认参数值
MODEL_PATH="/root/models/Qwen1.5-0.5B"
DATASET="hh_rlhf_en"
EPOCHS=1.0
OUTPUT_BASE="results/pref_beta_experiments"
EXPERIMENTS="1 2"  # 默认同时运行实验1和实验2

# 创建时间戳
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 显示帮助信息
function show_help {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -m, --model PATH        设置模型路径 (默认: $MODEL_PATH)"
    echo "  -d, --dataset NAME      设置训练数据集 (默认: $DATASET)"
    echo "  -p, --epochs NUM        设置训练轮数 (默认: $EPOCHS)"
    echo "  -o, --output PATH       设置输出目录基础路径 (默认: $OUTPUT_BASE)"
    echo "  --exps \"NUM1 NUM2...\"   设置要运行的实验类型，空格分隔 (默认: \"$EXPERIMENTS\")"
    echo "  -t, --no-timestamp      禁用时间戳"
    echo "  -h, --help              显示帮助信息"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -p|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --exps)
            EXPERIMENTS="$2"
            shift 2
            ;;
        -t|--no-timestamp)
            USE_TIMESTAMP=false
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "未知选项: $1"
            show_help
            ;;
    esac
done

# 确定输出目录
if [[ "$USE_TIMESTAMP" = true ]]; then
    OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"
else
    OUTPUT_DIR="${OUTPUT_BASE}"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 创建实验记录文件
cat > "${OUTPUT_DIR}/README.md" << EOF
# pref_beta 实验结果汇总 - $(date '+%Y-%m-%d %H:%M:%S')

## 实验设置
- **测试的 pref_beta 值**: ${BETA_VALUES[*]}
- **模型路径**: $MODEL_PATH
- **数据集**: $DATASET
- **训练轮数**: $EPOCHS
- **实验类型**: $EXPERIMENTS

## 目录结构
$(for exp in $EXPERIMENTS; do
    echo "- \`config_${exp}/\` - $(get_experiment_description $exp)"
    echo "  - beta值子实验:"
    for beta in "${BETA_VALUES[@]}"; do
        echo "    - \`beta_${beta}/\` - pref_beta = $beta"
    done
done)

## 实验结果
EOF

echo "======================================"
echo "开始 pref_beta 实验系列"
echo "测试的 pref_beta 值: ${BETA_VALUES[*]}"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET"
echo "实验类型: $EXPERIMENTS"
echo "输出目录: $OUTPUT_DIR"
echo "======================================"

# 获取实验描述的辅助函数
get_experiment_description() {
    local exp_num=$1
    case $exp_num in
        1)
            echo "use_dynamic_beta=false, disco_pref=false (标准DPO)"
            ;;
        2)
            echo "use_dynamic_beta=false, disco_pref=true (Disco-DPO)"
            ;;
        3)
            echo "use_dynamic_beta=true, disco_pref=false (动态Beta标准DPO)"
            ;;
        4)
            echo "use_dynamic_beta=true, disco_pref=true (动态Beta Disco-DPO)"
            ;;
    esac
}

# 遍历所有实验类型
for exp in $EXPERIMENTS; do
    echo "======================================"
    echo "运行实验类型: $(get_experiment_description $exp)"
    echo "======================================"
    
    # 构建实验类型目录
    exp_dir="${OUTPUT_DIR}/config_${exp}"
    mkdir -p "$exp_dir"
    
    # 记录当前实验类型的开始时间
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "- **实验类型 $exp**: 开始于 $start_time" >> "${OUTPUT_DIR}/README.md"
    
    # 遍历所有 beta 值运行实验
    for beta in "${BETA_VALUES[@]}"; do
        echo "----------------------------------------"
        echo "运行 pref_beta = $beta 的实验"
        echo "----------------------------------------"
        
        # 构建输出子目录
        beta_dir="${exp_dir}/beta_${beta}"
        mkdir -p "$beta_dir"
        
        # 记录当前beta值实验的开始时间
        beta_start_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "  - **pref_beta = $beta**: 开始于 $beta_start_time" >> "${OUTPUT_DIR}/README.md"
        
        # 运行实验
        ./xdpo/run_experiments.sh \
            --beta "$beta" \
            --model "$MODEL_PATH" \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --output-dir "$beta_dir" \
            --exp "$exp" \
            --timestamp false
        
        # 记录beta值实验结束时间
        beta_end_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "    - 结束于 $beta_end_time" >> "${OUTPUT_DIR}/README.md"
        echo "    - [详细结果](./beta_${beta})" >> "${OUTPUT_DIR}/README.md"
        echo "" >> "${OUTPUT_DIR}/README.md"
        
        echo "pref_beta = $beta 的实验完成"
        echo ""
    done
    
    # 记录实验类型结束时间
    end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "  - 结束于 $end_time" >> "${OUTPUT_DIR}/README.md"
    echo "" >> "${OUTPUT_DIR}/README.md"
    
    echo "实验类型 $exp 的所有beta值实验完成"
    echo ""
done

echo "======================================"
echo "所有 pref_beta 实验已完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "======================================"