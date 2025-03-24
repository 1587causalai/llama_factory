#!/bin/bash

# ===================================================
# pref_beta 实验脚本 (run_pref_beta_experiments.sh)
# ===================================================
#
# 本脚本用于测试不同pref_beta值对DPO实验结果的影响
# 默认测试值：0.05, 0.1, 0.5, 1.0, 2.0
#
# 使用方法：
# 1. 添加执行权限：chmod +x xdpo/quick_exp/run_pref_beta_experiments.sh
# 2. 默认运行：./xdpo/quick_exp/run_pref_beta_experiments.sh
# 3. 自定义参数：./xdpo/quick_exp/run_pref_beta_experiments.sh --model /path/to/model --dataset custom_dataset
#
# 示例:
# # 使用默认参数运行
# ./xdpo/quick_exp/run_pref_beta_experiments.sh
#
# # 指定模型和数据集
# ./xdpo/quick_exp/run_pref_beta_experiments.sh -m /root/models/Qwen1.5-1.8B -d hh_rlhf_zh
#
# # 同时运行实验1和实验2
# ./xdpo/quick_exp/run_pref_beta_experiments.sh --exps "1 2"
#
# # 完整自定义
# ./xdpo/quick_exp/run_pref_beta_experiments.sh -m /root/models/Qwen1.5-0.5B -d hh_rlhf_en -p 2.0 --exps "1 2" -o custom_results --no-timestamp
#
# 可用选项：
# -m, --model PATH        设置模型路径 (默认: /root/models/Qwen1.5-0.5B)
# -d, --dataset NAME      设置训练数据集 (默认: hh_rlhf_en)
# -e, --eval NAME         设置评估数据集 (默认: hh_rlhf_en)
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
WANDB_PROJECT="xdpo_pref_beta_exp"
OUTPUT_BASE="results/pref_beta_experiments"
EXPERIMENTS="1 2"  # 默认同时运行实验1和实验2

# 保存原始命令行参数，用于记录
ORIGINAL_ARGS="$*"

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

# 如果启用时间戳，则生成时间戳
TIMESTAMP=""
if [[ "$USE_TIMESTAMP" = true ]]; then
    TIMESTAMP="_$(date '+%Y%m%d_%H%M%S')"
    echo "已启用时间戳: $TIMESTAMP"
    
    # 更新wandb项目名称
    WANDB_PROJECT="${WANDB_PROJECT}${TIMESTAMP}"
fi

# 创建统一的实验目录，所有配置都将放在这里
EXPERIMENT_DIR="${OUTPUT_BASE}${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

echo "参数设置如下:"
echo "模型路径: $MODEL_PATH"
echo "训练数据集: $DATASET"
echo "训练轮数: $EPOCHS"
echo "WANDB项目: $WANDB_PROJECT"
echo "输出目录: $EXPERIMENT_DIR"
echo "启用时间戳: $USE_TIMESTAMP"
echo "将运行的实验编号: $EXPERIMENTS"
echo "测试的 pref_beta 值: ${BETA_VALUES[*]}"
echo ""

# 更新配置文件的函数
update_config() {
    local original_config_file=$1
    local config_num=$2
    local beta=$3
    
    # 创建实验输出目录（调整目录结构）
    local config_dir="${EXPERIMENT_DIR}/config_${config_num}/beta_${beta//./_}"
    local configs_dir="${config_dir}/configs"
    mkdir -p "$configs_dir"
    
    # 生成新配置文件路径
    local new_config_file="${configs_dir}/config_${config_num}.yaml"
    
    # 复制原始配置文件到新位置
    cp "$original_config_file" "$new_config_file"
    
    # 使用sed更新新配置文件中的参数
    sed -i "s|dataset:.*|dataset: $DATASET|g" $new_config_file
    sed -i "s|num_train_epochs:.*|num_train_epochs: $EPOCHS|g" $new_config_file
    sed -i "s|model_name_or_path:.*|model_name_or_path: $MODEL_PATH|g" $new_config_file
    sed -i "s|pref_beta:.*|pref_beta: $beta|g" $new_config_file
    
    # 更新输出目录设置
    sed -i "s|output_dir:.*|output_dir: $config_dir|g" $new_config_file
    sed -i "s|overwrite_output_dir:.*|overwrite_output_dir: true|g" $new_config_file
    
    # 创建README.md文件，记录实验信息
    cat > "${config_dir}/README.md" << EOF
# 实验 ${config_num} 配置信息 (pref_beta: $beta)

## 参数设置
- **实验类型**: 
  - 实验${config_num}: $(get_experiment_description $config_num)
- **训练数据集**: $DATASET
- **训练轮数**: $EPOCHS
- **模型路径**: $MODEL_PATH
- **WANDB项目**: $WANDB_PROJECT
- **偏好β值**: $beta

## 重现命令
要重现此实验，请运行以下命令:
\`\`\`bash
python xdpo/run_demo.py --config $new_config_file --wandb_project $WANDB_PROJECT
\`\`\`

## 配置文件
配置文件路径: \`$new_config_file\`

## 实验时间
生成时间: $(date '+%Y-%m-%d %H:%M:%S')
EOF
    
    # 打印状态信息（不会影响返回值）
    echo "已创建配置文件: $new_config_file" >&2
    echo "  - 输出目录: $config_dir" >&2
    
    # 仅返回新配置文件路径，不包含其他输出
    printf "%s" "$new_config_file"
}

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

# 创建并更新所有配置文件
# 为每个beta值创建配置文件
declare -A CONFIG_FILES
for beta in "${BETA_VALUES[@]}"; do
    for exp_num in $EXPERIMENTS; do
        CONFIG_FILES["${beta}_${exp_num}"]=$(update_config "xdpo/config_${exp_num}.yaml" "${exp_num}" "$beta")
    done
done

# 显示已创建的配置文件
echo "======================================"
echo "已准备配置文件:"
for beta in "${BETA_VALUES[@]}"; do
    echo "pref_beta: $beta"
    for exp_num in $EXPERIMENTS; do
        echo "  配置${exp_num}: ${CONFIG_FILES["${beta}_${exp_num}"]}"
    done
done
echo "======================================"

echo "======================================"
echo "开始运行选定的实验"
echo "======================================"

# 运行选定的实验
for beta in "${BETA_VALUES[@]}"; do
    echo "======================================"
    echo "运行 pref_beta = $beta 的实验"
    echo "======================================"
    
    for exp_num in $EXPERIMENTS; do
        case $exp_num in
            1)
                echo "运行标准DPO实验..."
                python xdpo/run_demo.py --config "${CONFIG_FILES["${beta}_${exp_num}"]}" --wandb_project $WANDB_PROJECT
                echo "标准DPO实验完成"
                echo ""
                ;;
            2)
                echo "运行Disco-DPO实验..."
                python xdpo/run_demo.py --config "${CONFIG_FILES["${beta}_${exp_num}"]}" --wandb_project $WANDB_PROJECT
                echo "Disco-DPO实验完成"
                echo ""
                ;;
            3)
                echo "运行动态Beta标准DPO实验..."
                python xdpo/run_demo.py --config "${CONFIG_FILES["${beta}_${exp_num}"]}" --wandb_project $WANDB_PROJECT
                echo "动态Beta标准DPO实验完成"
                echo ""
                ;;
            4)
                echo "运行动态Beta Disco-DPO实验..."
                python xdpo/run_demo.py --config "${CONFIG_FILES["${beta}_${exp_num}"]}" --wandb_project $WANDB_PROJECT
                echo "动态Beta Disco-DPO实验完成"
                echo ""
                ;;
        esac
    done
done

echo "所有实验已完成！"
echo "实验结果保存在: $EXPERIMENT_DIR"
echo "可通过 ${EXPERIMENT_DIR}/README.md 查看实验详情和重现命令。"

# 在统一实验目录创建说明文件
cat > "${EXPERIMENT_DIR}/README.md" << EOF
# pref_beta 实验组合 - $(date '+%Y-%m-%d %H:%M:%S')

## 实验概述
这个目录包含一组使用不同pref_beta值的DPO实验，对比标准DPO和Disco-DPO在不同beta值下的性能。

## pref_beta值列表
$(for beta in "${BETA_VALUES[@]}"; do
    echo "- $beta"
done)

## 包含的实验配置
$(for exp_num in $EXPERIMENTS; do
    echo "- 实验${exp_num}: $(get_experiment_description $exp_num)"
done)

## 全局参数设置
- **训练数据集**: $DATASET
- **训练轮数**: $EPOCHS
- **模型路径**: $MODEL_PATH
- **WANDB项目**: $WANDB_PROJECT

## 目录结构
$(for exp_num in $EXPERIMENTS; do
    echo "- \`config_${exp_num}/\` - $(get_experiment_description $exp_num)"
    echo "  - beta值子实验:"
    for beta in "${BETA_VALUES[@]}"; do
        echo "    - \`beta_${beta//./_}/\` - pref_beta = $beta"
    done
done)

## 原始运行命令
\`$0 $ORIGINAL_ARGS\`

## 实验时间
实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')
实验结束时间: $(date '+%Y-%m-%d %H:%M:%S')
EOF