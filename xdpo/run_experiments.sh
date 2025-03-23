#!/bin/bash

# ===================================================
# DPO实验脚本 (run_experiments.sh)
# ===================================================
#
# 本脚本用于运行不同参数组合的DPO实验，支持以下功能：
# 1. 通过命令行参数修改配置文件中的各项参数
# 2. 选择性地运行特定类型的实验
# 3. 自动更新所有配置文件并执行实验
# 4. 保存实验脚本副本到输出目录，方便后续复现
# 5. 支持多模型对比实验
#
# 基本用法：
# ./xdpo/run_experiments.sh [选项]
#
# 示例:
# # 使用默认参数运行所有实验
# ./xdpo/run_experiments.sh
#
# # 只运行样本数为500的实验
# ./xdpo/run_experiments.sh --samples 500
#
# # 修改数据集和评估数据集
# ./xdpo/run_experiments.sh -d custom_dataset -e test_dataset
#
# # 只运行use_dynamic_beta=true的实验（实验3和4）
# ./xdpo/run_experiments.sh --dynamic-beta
#
# # 只运行disco_pref=true的实验（实验2和4）
# ./xdpo/run_experiments.sh --disco-pref  # or base-pref
#
# # 只运行特定编号的实验
# ./xdpo/run_experiments.sh --exp 2
#
# # 调整训练参数
# ./xdpo/run_experiments.sh --beta 0.5 --lr 5e-5 --batch 4
#
# # 综合使用多个参数
# ./xdpo/run_experiments.sh -s 800 -d hh_rlhf_zh -p 2.0 --beta 0.3 --base-pref
#
# # 使用时间戳防止结果被覆盖
# ./xdpo/run_experiments.sh --timestamp
#
# # 运行Qwen1.5-0.5B和Qwen2.5-1.5B-Instruct的对比实验
# ./xdpo/run_experiments.sh --compare-models
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
# --compare-models        运行Qwen1.5-0.5B和Qwen2.5-1.5B-Instruct的对比实验
# --dynamic-beta          只运行use_dynamic_beta=true的实验（实验3和4）
# --static-beta           只运行use_dynamic_beta=false的实验（实验1和2）
# --disco-pref            只运行disco_pref=true的实验（实验2和4）
# --base-pref             只运行disco_pref=false的实验（实验1和3）
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
USE_TIMESTAMP=false
MAX_SAMPLES=1000
DATASET="hh_rlhf_en" # dpo_en_demo, dpo_zh_demo
# EVAL_DATASET="hh_rlhf_en"   
EPOCHS=1.0
MODEL_PATH="/root/models/Qwen1.5-0.5B"
# WANDB_PROJECT="xdpo_demo"
WANDB_PROJECT="xdpo_pref_beta_exp"
OUTPUT_DIR_BASE="results/dpo_baseline_qwen0.5b"

# 新增控制参数
PREF_BETA=0.7
LEARNING_RATE=1.0e-4
BATCH_SIZE=2
GRAD_ACCUM_STEPS=4
LORA_RANK=8
CUTOFF_LEN=1024
WARMUP_RATIO=0.1
LOGGING_STEPS=5
SAVE_STEPS=200

# 保存原始命令行参数，用于记录
ORIGINAL_ARGS="$*"

# 添加模型对比参数
MODEL_COMPARE=false

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
    echo "  -l, --lr NUM            设置学习率 (默认: $LEARNING_RATE)"
    echo "  --batch NUM             设置训练批量大小 (默认: $BATCH_SIZE)"
    echo "  --grad-accum NUM        设置梯度累积步数 (默认: $GRAD_ACCUM_STEPS)"
    echo "  --lora-rank NUM         设置LoRA秩 (默认: $LORA_RANK)"
    echo "  --cutoff NUM            设置最大上下文长度 (默认: $CUTOFF_LEN)"
    echo "  --warmup NUM            设置warmup比例 (默认: $WARMUP_RATIO)"
    echo "  --log-steps NUM         设置日志记录步数 (默认: $LOGGING_STEPS)"
    echo "  --save-steps NUM        设置模型保存步数 (默认: $SAVE_STEPS)"
    echo "  --output-dir PATH       设置输出目录的基础路径 (默认: $OUTPUT_DIR_BASE)"
    echo "  -t, --timestamp         启用时间戳（防止实验结果覆盖）"
    echo "  --compare-models        运行Qwen1.5-0.5B和Qwen2.5-1.5B-Instruct的对比实验"
    echo "  --dynamic-beta          只运行use_dynamic_beta=true的实验"
    echo "  --static-beta           只运行use_dynamic_beta=false的实验"
    echo "  --disco-pref            只运行disco_pref=true的实验"
    echo "  --base-pref             只运行disco_pref=false的实验"
    echo "  --exp NUM               只运行指定编号的实验(1-4)"
    echo "  -h, --help              显示帮助信息"
    exit 1
}

# 用于存储要运行的实验编号
declare -a experiments_to_run=(1 2 3 4)

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
            LEARNING_RATE="$2"
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
        --compare-models)
            MODEL_COMPARE=true
            shift
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
echo "学习率: $LEARNING_RATE"
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
    
    # 创建实验输出目录
    local config_dir="${EXPERIMENT_DIR}/config_${config_num}"
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
    sed -i "s|learning_rate:.*|learning_rate: $LEARNING_RATE|g" $new_config_file
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
# 实验 ${config_num} 配置信息

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
- **学习率**: $LEARNING_RATE
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

# 在特定配置下运行模型实验的函数
run_model_for_config() {
    local config_dir=$1    # 配置目录路径
    local config_num=$2    # 配置编号
    local model_name=$3    # 模型名称
    local model_path=$4    # 模型路径
    local model_dir_name=$5 # 模型目录名
    
    # 创建模型输出目录
    local model_output_dir="${config_dir}/${model_dir_name}"
    mkdir -p "$model_output_dir"
    
    echo "======================================"
    echo "配置${config_num}下运行模型: $model_name"
    echo "输出目录: $model_output_dir"
    echo "======================================"
    
    # 创建configs目录
    local configs_dir="${model_output_dir}/configs"
    mkdir -p "$configs_dir"
    
    # 复制和修改配置文件
    local config_file="${configs_dir}/config_${config_num}.yaml"
    cp "xdpo/config_${config_num}.yaml" "$config_file"
    
    # 使用sed更新配置文件参数
    sed -i "s|max_samples:.*|max_samples: $MAX_SAMPLES|g" $config_file
    sed -i "s|dataset:.*|dataset: $DATASET|g" $config_file
    sed -i "s|eval_dataset:.*|eval_dataset: $EVAL_DATASET|g" $config_file
    sed -i "s|num_train_epochs:.*|num_train_epochs: $EPOCHS|g" $config_file
    sed -i "s|model_name_or_path:.*|model_name_or_path: $model_path|g" $config_file
    sed -i "s|pref_beta:.*|pref_beta: $PREF_BETA|g" $config_file
    sed -i "s|learning_rate:.*|learning_rate: $LEARNING_RATE|g" $config_file
    sed -i "s|per_device_train_batch_size:.*|per_device_train_batch_size: $BATCH_SIZE|g" $config_file
    sed -i "s|gradient_accumulation_steps:.*|gradient_accumulation_steps: $GRAD_ACCUM_STEPS|g" $config_file
    sed -i "s|lora_rank:.*|lora_rank: $LORA_RANK|g" $config_file
    sed -i "s|cutoff_len:.*|cutoff_len: $CUTOFF_LEN|g" $config_file
    sed -i "s|warmup_ratio:.*|warmup_ratio: $WARMUP_RATIO|g" $config_file
    sed -i "s|logging_steps:.*|logging_steps: $LOGGING_STEPS|g" $config_file
    sed -i "s|save_steps:.*|save_steps: $SAVE_STEPS|g" $config_file
    
    # 更新输出目录
    sed -i "s|output_dir:.*|output_dir: $model_output_dir|g" $config_file
    sed -i "s|overwrite_output_dir:.*|overwrite_output_dir: true|g" $config_file
    
    # 运行实验
    echo "开始运行实验..."
    python xdpo/run_demo.py --config "$config_file" --wandb_project "$WANDB_PROJECT"
    
    echo "模型 $model_name 在配置${config_num}下的实验完成"
    echo ""
}

# 如果是模型对比模式，则运行两个模型的实验并退出
if [[ "$MODEL_COMPARE" = true ]]; then
    echo "======================================"
    echo "进行模型对比实验"
    echo "======================================"
    
    # 创建对比实验的根目录
    COMPARE_DIR="${OUTPUT_DIR_BASE}${TIMESTAMP}"
    mkdir -p "$COMPARE_DIR"
    
    # 保存原始OUTPUT_DIR_BASE
    ORIGINAL_OUTPUT_DIR="$OUTPUT_DIR_BASE"
    
    # 创建总实验README文件
    cat > "$COMPARE_DIR/README.md" << EOF
# 模型对比实验 - $(date '+%Y-%m-%d %H:%M:%S')

## 实验概述
本目录包含不同模型在相同配置下的DPO实验结果，便于直接比较模型性能。

## 测试模型
- Qwen1.5-0.5B: \`/root/models/Qwen1.5-0.5B\`
- Qwen2.5-1.5B-Instruct: \`/root/models/Qwen2.5-1.5B-Instruct\`

## 实验配置类型
- 配置1 (config_1): use_dynamic_beta=false, disco_pref=false (标准DPO)
- 配置2 (config_2): use_dynamic_beta=false, disco_pref=true (Disco-DPO)

## 全局参数设置
- **样本数量**: $MAX_SAMPLES
- **训练数据集**: $DATASET
- **评估数据集**: $EVAL_DATASET
- **训练轮数**: $EPOCHS
- **WANDB项目**: $WANDB_PROJECT
- **偏好β值**: $PREF_BETA
- **学习率**: $LEARNING_RATE
- **训练批量大小**: $BATCH_SIZE
- **梯度累积步数**: $GRAD_ACCUM_STEPS
- **LoRA秩**: $LORA_RANK
- **最大上下文长度**: $CUTOFF_LEN
- **Warmup比例**: $WARMUP_RATIO
- **日志记录步数**: $LOGGING_STEPS
- **模型保存步数**: $SAVE_STEPS

## 目录结构
- \`config_1/\` - 标准DPO配置实验结果
  - \`qwen1_5_0_5b/\` - Qwen1.5-0.5B模型在配置1下的结果
  - \`qwen2_5_1_5b/\` - Qwen2.5-1.5B-Instruct模型在配置1下的结果
- \`config_2/\` - Disco-DPO配置实验结果
  - \`qwen1_5_0_5b/\` - Qwen1.5-0.5B模型在配置2下的结果
  - \`qwen2_5_1_5b/\` - Qwen2.5-1.5B-Instruct模型在配置2下的结果

## 原始实验命令
\`$0 $ORIGINAL_ARGS\`

## 实验时间
开始时间: $(date '+%Y-%m-%d %H:%M:%S')
EOF

    echo "将在两个静态beta配置下对比模型性能..."
    
    # 为每个配置创建目录并运行实验
    for config_num in 1 2; do
        config_dir="$COMPARE_DIR/config_$config_num"
        mkdir -p "$config_dir"
        
        # 获取配置描述
        config_desc=$(get_experiment_description $config_num)
        echo "======================================"
        echo "配置$config_num: $config_desc"
        echo "======================================"
        
        # 创建配置README
        cat > "$config_dir/README.md" << EOF
# 配置$config_num: $config_desc

本目录包含不同模型在配置$config_num下的实验结果。

## 模型列表
- \`qwen1_5_0_5b/\` - Qwen1.5-0.5B
- \`qwen2_5_1_5b/\` - Qwen2.5-1.5B-Instruct

## 重现命令
要重现此实验，请使用以下命令:
\`\`\`bash
python xdpo/run_demo.py --config 配置文件路径 --wandb_project $WANDB_PROJECT
\`\`\`

## 实验时间
开始时间: $(date '+%Y-%m-%d %H:%M:%S')
EOF
        
        # 运行每个模型的实验
        echo "运行Qwen1.5-0.5B在配置$config_num下的实验..."
        run_model_for_config "$config_dir" "$config_num" "Qwen1.5-0.5B" "/root/models/Qwen1.5-0.5B" "qwen1_5_0_5b"
        
        echo "运行Qwen2.5-1.5B-Instruct在配置$config_num下的实验..."
        run_model_for_config "$config_dir" "$config_num" "Qwen2.5-1.5B-Instruct" "/root/models/Qwen2.5-1.5B-Instruct" "qwen2_5_1_5b"
        
        # 更新配置README
        echo "实验完成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$config_dir/README.md"
    done
    
    # 更新总README
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$COMPARE_DIR/README.md"
    
    echo "模型对比实验完成！"
    echo "对比实验结果目录: $COMPARE_DIR"
    echo "wandb项目: $WANDB_PROJECT"
    exit 0
fi

# 创建并更新所有配置文件
# 只为要运行的实验创建配置文件
declare -A CONFIG_FILES
for exp_num in "${experiments_to_run[@]}"; do
    CONFIG_FILES[$exp_num]=$(update_config "xdpo/config_${exp_num}.yaml" "${exp_num}")
done

# 显示已创建的配置文件
echo "======================================"
echo "已准备配置文件:"
for exp_num in "${experiments_to_run[@]}"; do
    echo "配置${exp_num}: ${CONFIG_FILES[$exp_num]}"
done
echo "======================================"

echo "======================================"
echo "开始运行选定的实验"
echo "======================================"

# 运行选定的实验
for exp_num in "${experiments_to_run[@]}"; do
    case $exp_num in
        1)
            echo "======================================"
            echo "运行实验1: use_dynamic_beta=false, disco_pref=false"
            echo "======================================"
            python xdpo/run_demo.py --config "${CONFIG_FILES[$exp_num]}" --wandb_project $WANDB_PROJECT
            echo "实验1完成"
            echo ""
            ;;
        2)
            echo "======================================"
            echo "运行实验2: use_dynamic_beta=false, disco_pref=true"
            echo "======================================"
            python xdpo/run_demo.py --config "${CONFIG_FILES[$exp_num]}" --wandb_project $WANDB_PROJECT
            echo "实验2完成"
            echo ""
            ;;
        3)
            echo "======================================"
            echo "运行实验3: use_dynamic_beta=true, disco_pref=false"
            echo "======================================"
            python xdpo/run_demo.py --config "${CONFIG_FILES[$exp_num]}" --wandb_project $WANDB_PROJECT
            echo "实验3完成"
            echo ""
            ;;
        4)
            echo "======================================"
            echo "运行实验4: use_dynamic_beta=true, disco_pref=true"
            echo "======================================"
            python xdpo/run_demo.py --config "${CONFIG_FILES[$exp_num]}" --wandb_project $WANDB_PROJECT
            echo "实验4完成"
            echo ""
            ;;
    esac
done

echo "所有实验已完成！"
echo "实验结果保存在: $EXPERIMENT_DIR"
echo "可通过 ${EXPERIMENT_DIR}/README.md 查看实验详情和重现命令。"

# 在统一实验目录创建说明文件
cat > "${EXPERIMENT_DIR}/README.md " << EOF
# DPO实验组合 - $(date '+%Y-%m-%d %H:%M:%S')

## 实验概述
这个目录包含一组使用相同参数设置的DPO实验，通过组合不同的dynamic_beta和disco_pref设置得到。

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
- **学习率**: $LEARNING_RATE
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
done)

## 原始运行命令
\`$0 $ORIGINAL_ARGS\`

## 实验时间
实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')
实验结束时间: $(date '+%Y-%m-%d %H:%M:%S')
EOF 