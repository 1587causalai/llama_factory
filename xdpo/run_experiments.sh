#!/bin/bash

# # 使用默认参数运行
# ./xdpo/run_experiments.sh

# # 设置样本数为200运行
# ./xdpo/run_experiments.sh --samples 200

# # 同时设置多个参数
# ./xdpo/run_experiments.sh -s 500 -d custom_dataset -e test_dataset -p 5.0

# 定义默认参数
MAX_SAMPLES=200
DATASET="dpo_en_demo"
EVAL_DATASET="dpo_zh_demo"
EPOCHS=1.0
MODEL_PATH="/root/models/Qwen1.5-0.5B"
WANDB_PROJECT="xdpo_demo2"

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
    echo "  -h, --help              显示帮助信息"
    exit 1
}

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
        -h|--help)
            show_help
            ;;
        *)
            echo "未知选项: $1"
            show_help
            ;;
    esac
done

echo "参数设置如下:"
echo "样本数量: $MAX_SAMPLES"
echo "训练数据集: $DATASET"
echo "评估数据集: $EVAL_DATASET"
echo "训练轮数: $EPOCHS"
echo "模型路径: $MODEL_PATH"
echo "WANDB项目: $WANDB_PROJECT"
echo ""

# 更新配置文件的函数
update_config() {
    local config_file=$1
    
    # 使用sed更新配置文件中的参数
    sed -i "s|max_samples:.*|max_samples: $MAX_SAMPLES|g" $config_file
    sed -i "s|dataset:.*|dataset: $DATASET|g" $config_file
    sed -i "s|eval_dataset:.*|eval_dataset: $EVAL_DATASET|g" $config_file
    sed -i "s|num_train_epochs:.*|num_train_epochs: $EPOCHS|g" $config_file
    sed -i "s|model_name_or_path:.*|model_name_or_path: $MODEL_PATH|g" $config_file
    
    echo "已更新配置文件: $config_file"
}

# 更新所有配置文件
update_config "xdpo/config_1.yaml"
update_config "xdpo/config_2.yaml"
update_config "xdpo/config_3.yaml"
update_config "xdpo/config_4.yaml"

echo "======================================"
echo "开始运行4种参数组合实验"
echo "======================================"

# 运行实验1: use_dynamic_beta=false, disco_pref=false
echo "======================================"
echo "运行实验1: use_dynamic_beta=false, disco_pref=false"
echo "======================================"
python xdpo/run_demo.py --config xdpo/config_1.yaml --wandb_project $WANDB_PROJECT
echo "实验1完成"
echo ""

# 运行实验2: use_dynamic_beta=false, disco_pref=true
echo "======================================"
echo "运行实验2: use_dynamic_beta=false, disco_pref=true"
echo "======================================"
python xdpo/run_demo.py --config xdpo/config_2.yaml --wandb_project $WANDB_PROJECT
echo "实验2完成"
echo ""

# 运行实验3: use_dynamic_beta=true, disco_pref=false
echo "======================================"
echo "运行实验3: use_dynamic_beta=true, disco_pref=false"
echo "======================================"
python xdpo/run_demo.py --config xdpo/config_3.yaml --wandb_project $WANDB_PROJECT
echo "实验3完成"
echo ""

# 运行实验4: use_dynamic_beta=true, disco_pref=true
echo "======================================"
echo "运行实验4: use_dynamic_beta=true, disco_pref=true"
echo "======================================"
python xdpo/run_demo.py --config xdpo/config_4.yaml --wandb_project $WANDB_PROJECT
echo "实验4完成"
echo ""

echo "所有实验已完成！" 