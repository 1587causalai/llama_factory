#!/bin/bash

# ===================================================
# 学习速率实验脚本 (run_lr_experiments.sh)
# ===================================================
#
# 本脚本用于运行不同学习速率的DPO实验，支持以下功能：
# 1. 通过命令行参数修改配置文件中的各项参数
# 2. 选择性地运行特定类型的实验
# 3. 自动更新所有配置文件并执行实验
# 4. 保存实验脚本副本到输出目录，方便后续复现
#
# 基本用法：
# ./xdpo/quick_exp/run_lr_experiments.sh [选项]
#
# 示例:
# # 使用默认参数运行所有实验
# ./xdpo/run_lr_experiments.sh
#
# # 只运行特定学习速率的实验
# ./xdpo/quick_exp/run_lr_experiments.sh --lr 5e-5
#
# # 修改数据集和评估数据集
# ./xdpo/quick_exp/run_lr_experiments.sh -d custom_dataset -e test_dataset
#
# # 只运行use_dynamic_beta=true的实验
# ./xdpo/quick_exp/run_lr_experiments.sh --dynamic-beta
#
# # 只运行disco_pref=true的实验
# ./xdpo/run_lr_experiments.sh --disco-pref
#
# # 只运行特定编号的实验
# ./xdpo/quick_exp/run_lr_experiments.sh --exp 2
#
# # 调整训练参数
# ./xdpo/quick_exp/run_lr_experiments.sh --beta 0.5 --batch 4
#
# # 使用时间戳防止结果被覆盖
# ./xdpo/quick_exp/run_lr_experiments.sh --timestamp
#
# 可用选项：
# -s, --samples NUM       设置样本数量 
# -d, --dataset NAME      设置训练数据集
# -e, --eval NAME         设置评估数据集
# -p, --epochs NUM        设置训练轮数
# -m, --model PATH        设置模型路径
# -w, --wandb NAME        设置wandb项目名称
# -b, --beta NUM          设置pref_beta值
# -l, --lr NUM            设置学习率
# --batch NUM             设置训练批量大小
# --grad-accum NUM        设置梯度累积步数
# --lora-rank NUM         设置LoRA秩
# --cutoff NUM            设置最大上下文长度
# --warmup NUM            设置warmup比例
# --log-steps NUM         设置日志记录步数
# --save-steps NUM        设置模型保存步数
# --output-dir PATH       设置输出目录的基础路径
# -t, --timestamp         启用时间戳（防止实验结果覆盖）
# --dynamic-beta          只运行use_dynamic_beta=true的实验
# --static-beta           只运行use_dynamic_beta=false的实验
# --disco-pref            只运行disco_pref=true的实验
# --base-pref             只运行disco_pref=false的实验
# --exp NUM               只运行指定编号的实验(1-4)
# -h, --help              显示帮助信息
#
# 实验配置说明:
# 实验1：use_dynamic_beta=false, disco_pref=false (标准DPO)
# 实验2：use_dynamic_beta=false, disco_pref=true (Disco-DPO)
# 实验3：use_dynamic_beta=true, disco_pref=false (动态Beta标准DPO)
# 实验4：use_dynamic_beta=true, disco_pref=true (动态Beta Disco-DPO)
# ===================================================

# 定义默认参数
USE_TIMESTAMP=true
MAX_SAMPLES=1000
DATASET="hh_rlhf_en"
EPOCHS=1.0
MODEL_PATH="/root/models/Qwen1.5-0.5B"
WANDB_PROJECT="xdpo_lr_exp"
OUTPUT_DIR_BASE="results/lr_experiments"

# 新增控制参数
PREF_BETA=0.7
# 定义学习率列表（修复语法错误）
LEARNING_RATES=("1.0e-3" "1.0e-4" "5.0e-5" "1.0e-5")
BATCH_SIZE=2
GRAD_ACCUM_STEPS=4
LORA_RANK=8
CUTOFF_LEN=1024
WARMUP_RATIO=0.1
LOGGING_STEPS=5
SAVE_STEPS=200

# 保存原始命令行参数，用于记录
ORIGINAL_ARGS="$*"

# 显示帮助信息
function show_help {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -s, --samples NUM       设置样本数量 (默认: $MAX_SAMPLES)"
    echo "  -d, --dataset NAME      设置训练数据集 (默认: $DATASET)"
    echo "  -e, --eval NAME         设置评估数据集 (默认: $EVAL_DATASET)"
    echo "  -p, --epochs NUM        设置训练轮数 (默认: $EPOCHS)"
    echo "  -m, --model PATH        设置模型路径 (默认: $MODEL_PATH)"
    echo "  -w, --wandb NAME        设置wandb项目名称 (默认: $WANDB_PROJECT)"
    echo "  -b, --beta NUM          设置pref_beta值 (默认: $PREF_BETA)"
    echo "  -l, --lr NUM            设置单个学习率 (默认: ${LEARNING_RATES[0]})"
    echo "  --lrs NUM1,NUM2,...     设置学习率列表 (默认: ${LEARNING_RATES[*]})"
    echo "  --batch NUM             设置训练批量大小 (默认: $BATCH_SIZE)"
    echo "  --grad-accum NUM        设置梯度累积步数 (默认: $GRAD_ACCUM_STEPS)"
    echo "  --lora-rank NUM         设置LoRA秩 (默认: $LORA_RANK)"
    echo "  --cutoff NUM            设置最大上下文长度 (默认: $CUTOFF_LEN)"
    echo "  --warmup NUM            设置warmup比例 (默认: $WARMUP_RATIO)"
    echo "  --log-steps NUM         设置日志记录步数 (默认: $LOGGING_STEPS)"
    echo "  --save-steps NUM        设置模型保存步数 (默认: $SAVE_STEPS)"
    echo "  --output-dir PATH       设置输出目录的基础路径 (默认: $OUTPUT_DIR_BASE)"
    echo "  -t, --timestamp         启用时间戳（防止实验结果覆盖）"
    echo "  --dynamic-beta          只运行use_dynamic_beta=true的实验"
    echo "  --static-beta           只运行use_dynamic_beta=false的实验"
    echo "  --disco-pref            只运行disco_pref=true的实验"
    echo "  --base-pref             只运行disco_pref=false的实验"
    echo "  --exp NUM               只运行指定编号的实验(1-4)"
    echo "  -h, --help              显示帮助信息"
    exit 1
}

# 用于存储要运行的实验编号
declare -a experiments_to_run=(1 2)  # 默认只运行标准DPO和Disco-DPO

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -e|--eval)
            EVAL_DATASET="$2"
            shift 2
            ;;
        -p|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -w|--wandb)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -b|--beta)
            PREF_BETA="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATES=("$2")
            shift 2
            ;;
        --lrs)
            IFS=',' read -ra LEARNING_RATES <<< "$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --cutoff)
            CUTOFF_LEN="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --log-steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --save-steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_BASE="$2"
            shift 2
            ;;
        -t|--timestamp)
            if [[ "$2" == "false" ]]; then
                USE_TIMESTAMP=false
                shift 2
            else
                USE_TIMESTAMP=true
                shift
            fi
            ;;
        --dynamic-beta)
            experiments_to_run=(3 4)
            shift
            ;;
        --static-beta)
            experiments_to_run=(1 2)
            shift
            ;;
        --disco-pref)
            experiments_to_run=(2 4)
            shift
            ;;
        --base-pref)
            experiments_to_run=(1 3)
            shift
            ;;
        --exp)
            # 只运行指定编号的实验
            experiments_to_run=("$2")
            shift 2
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
EXPERIMENT_DIR="${OUTPUT_DIR_BASE}${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

echo "参数设置如下:"
echo "样本数量: $MAX_SAMPLES"
echo "训练数据集: $DATASET"
echo "评估数据集: $EVAL_DATASET"
echo "训练轮数: $EPOCHS"
echo "模型路径: $MODEL_PATH"
echo "WANDB项目: $WANDB_PROJECT"
echo "输出目录: $EXPERIMENT_DIR"
echo "启用时间戳: $USE_TIMESTAMP"
echo "偏好β值: $PREF_BETA"
echo "学习率列表: ${LEARNING_RATES[*]}"
echo "训练批量大小: $BATCH_SIZE"
echo "梯度累积步数: $GRAD_ACCUM_STEPS"
echo "LoRA秩: $LORA_RANK"
echo "最大上下文长度: $CUTOFF_LEN"
echo "Warmup比例: $WARMUP_RATIO"
echo "日志记录步数: $LOGGING_STEPS"
echo "模型保存步数: $SAVE_STEPS"
echo "将运行的实验编号: ${experiments_to_run[*]}"
echo ""

# 更新配置文件的函数
update_config() {
    local original_config_file=$1
    local config_num=$2
    local lr=$3
    
    # 创建实验输出目录（调整目录结构）
    local config_dir="${EXPERIMENT_DIR}/config_${config_num}/lr_${lr//./_}"
    local configs_dir="${config_dir}/configs"
    mkdir -p "$configs_dir"
    
    # 生成新配置文件路径
    local new_config_file="${configs_dir}/config_${config_num}.yaml"
    
    # 复制原始配置文件到新位置
    cp "$original_config_file" "$new_config_file"
    
    # 使用sed更新新配置文件中的参数
    sed -i "s|max_samples:.*|max_samples: $MAX_SAMPLES|g" $new_config_file
    sed -i "s|dataset:.*|dataset: $DATASET|g" $new_config_file
    sed -i "s|eval_dataset:.*|eval_dataset: $EVAL_DATASET|g" $new_config_file
    # 如果设置了eval_dataset，则删除val_size行以避免冲突
    if [[ -n "$EVAL_DATASET" ]]; then
        sed -i '/val_size:/d' $new_config_file
    fi
    sed -i "s|num_train_epochs:.*|num_train_epochs: $EPOCHS|g" $new_config_file
    sed -i "s|model_name_or_path:.*|model_name_or_path: $MODEL_PATH|g" $new_config_file
    sed -i "s|pref_beta:.*|pref_beta: $PREF_BETA|g" $new_config_file
    sed -i "s|learning_rate:.*|learning_rate: $lr|g" $new_config_file
    sed -i "s|per_device_train_batch_size:.*|per_device_train_batch_size: $BATCH_SIZE|g" $new_config_file
    sed -i "s|gradient_accumulation_steps:.*|gradient_accumulation_steps: $GRAD_ACCUM_STEPS|g" $new_config_file
    sed -i "s|lora_rank:.*|lora_rank: $LORA_RANK|g" $new_config_file
    sed -i "s|cutoff_len:.*|cutoff_len: $CUTOFF_LEN|g" $new_config_file
    sed -i "s|warmup_ratio:.*|warmup_ratio: $WARMUP_RATIO|g" $new_config_file
    sed -i "s|logging_steps:.*|logging_steps: $LOGGING_STEPS|g" $new_config_file
    sed -i "s|save_steps:.*|save_steps: $SAVE_STEPS|g" $new_config_file
    
    # 更新输出目录设置
    sed -i "s|output_dir:.*|output_dir: $config_dir|g" $new_config_file
    sed -i "s|overwrite_output_dir:.*|overwrite_output_dir: true|g" $new_config_file
    
    # 创建README.md文件，记录实验信息
    cat > "${config_dir}/README.md" << EOF
# 实验 ${config_num} 配置信息 (学习率: $lr)

## 参数设置
- **实验类型**: 
  - 实验${config_num}: $(get_experiment_description $config_num)
- **样本数量**: $MAX_SAMPLES
- **训练数据集**: $DATASET
- **评估数据集**: $EVAL_DATASET
- **训练轮数**: $EPOCHS
- **模型路径**: $MODEL_PATH
- **WANDB项目**: $WANDB_PROJECT
- **偏好β值**: $PREF_BETA
- **学习率**: $lr
- **训练批量大小**: $BATCH_SIZE
- **梯度累积步数**: $GRAD_ACCUM_STEPS
- **LoRA秩**: $LORA_RANK
- **最大上下文长度**: $CUTOFF_LEN
- **Warmup比例**: $WARMUP_RATIO
- **日志记录步数**: $LOGGING_STEPS
- **模型保存步数**: $SAVE_STEPS

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
# 为每个学习率创建配置文件
declare -A CONFIG_FILES
for lr in "${LEARNING_RATES[@]}"; do
    for exp_num in "${experiments_to_run[@]}"; do
        CONFIG_FILES["${lr}_${exp_num}"]=$(update_config "xdpo/config_${exp_num}.yaml" "${exp_num}" "$lr")
    done
done

# 显示已创建的配置文件
echo "======================================"
echo "已准备配置文件:"
for lr in "${LEARNING_RATES[@]}"; do
    echo "学习率: $lr"
    for exp_num in "${experiments_to_run[@]}"; do
        echo "  配置${exp_num}: ${CONFIG_FILES["${lr}_${exp_num}"]}"
    done
done
echo "======================================"

echo "======================================"
echo "开始运行选定的实验"
echo "======================================"

# 运行选定的实验
for lr in "${LEARNING_RATES[@]}"; do
    echo "======================================"
    echo "运行学习率 $lr 的实验"
    echo "======================================"
    
    for exp_num in "${experiments_to_run[@]}"; do
        case $exp_num in
            1)
                echo "运行标准DPO实验..."
                python xdpo/run_demo.py --config "${CONFIG_FILES["${lr}_${exp_num}"]}" --wandb_project $WANDB_PROJECT
                echo "标准DPO实验完成"
                echo ""
                ;;
            2)
                echo "运行Disco-DPO实验..."
                python xdpo/run_demo.py --config "${CONFIG_FILES["${lr}_${exp_num}"]}" --wandb_project $WANDB_PROJECT
                echo "Disco-DPO实验完成"
                echo ""
                ;;
            3)
                echo "运行动态Beta标准DPO实验..."
                python xdpo/run_demo.py --config "${CONFIG_FILES["${lr}_${exp_num}"]}" --wandb_project $WANDB_PROJECT
                echo "动态Beta标准DPO实验完成"
                echo ""
                ;;
            4)
                echo "运行动态Beta Disco-DPO实验..."
                python xdpo/run_demo.py --config "${CONFIG_FILES["${lr}_${exp_num}"]}" --wandb_project $WANDB_PROJECT
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
# 学习速率实验组合 - $(date '+%Y-%m-%d %H:%M:%S')

## 实验概述
这个目录包含一组使用不同学习速率的DPO实验，对比标准DPO和Disco-DPO在不同学习率下的性能。

## 学习率列表
$(for lr in "${LEARNING_RATES[@]}"; do
    echo "- $lr"
done)

## 包含的实验配置
$(for exp_num in "${experiments_to_run[@]}"; do
    echo "- 实验${exp_num}: $(get_experiment_description $exp_num)"
done)

## 全局参数设置
- **样本数量**: $MAX_SAMPLES
- **训练数据集**: $DATASET
- **评估数据集**: $EVAL_DATASET
- **训练轮数**: $EPOCHS
- **模型路径**: $MODEL_PATH
- **WANDB项目**: $WANDB_PROJECT
- **偏好β值**: $PREF_BETA
- **训练批量大小**: $BATCH_SIZE
- **梯度累积步数**: $GRAD_ACCUM_STEPS
- **LoRA秩**: $LORA_RANK
- **最大上下文长度**: $CUTOFF_LEN
- **Warmup比例**: $WARMUP_RATIO
- **日志记录步数**: $LOGGING_STEPS
- **模型保存步数**: $SAVE_STEPS

## 目录结构
$(for exp_num in "${experiments_to_run[@]}"; do
    echo "- \`config_${exp_num}/\` - $(get_experiment_description $exp_num)"
    echo "  - 学习率子实验:"
    for lr in "${LEARNING_RATES[@]}"; do
        echo "    - \`lr_${lr//./_}/\` - 学习率 $lr"
    done
done)

## 原始运行命令
\`$0 $ORIGINAL_ARGS\`

## 实验时间
实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')
实验结束时间: $(date '+%Y-%m-%d %H:%M:%S')
EOF 