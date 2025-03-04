#!/bin/bash

# DPO训练和评估一键运行脚本

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印带颜色的信息
function log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
log_info "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    log_error "未找到python3，请安装Python 3.8+。"
    exit 1
fi

# 确保工作目录
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
log_info "项目根目录: $PROJECT_ROOT"

# 检查模型文件
MODEL_PATH=~/models/Qwen1.5-0.5B
if [ ! -d "$MODEL_PATH" ]; then
    log_error "未找到模型: $MODEL_PATH"
    exit 1
fi
log_success "找到模型: $MODEL_PATH"

# 检查数据文件
DATA_PATH=$PROJECT_ROOT/dpo_baseline/data/dpo_sample.json
if [ ! -f "$DATA_PATH" ]; then
    log_error "未找到数据文件: $DATA_PATH"
    exit 1
fi
log_success "找到数据文件: $DATA_PATH"

# 准备输出目录
OUTPUT_DIR=$PROJECT_ROOT/dpo_baseline/output
mkdir -p $OUTPUT_DIR
log_info "输出目录: $OUTPUT_DIR"

# 运行DPO训练
log_info "开始DPO训练..."
python src/train.py \
    --config_file dpo_baseline/dpo_config.yaml \
    --report_to none

if [ $? -eq 0 ]; then
    log_success "DPO训练完成！"
else
    log_error "DPO训练失败，请检查错误信息。"
    exit 1
fi

# 执行模型评估
log_info "开始评估模型..."
python src/evaluate.py \
    --model_name_or_path $OUTPUT_DIR \
    --template default \
    --max_length 1024 \
    --question "请解释下量子纠缠现象" \
    --answer_only

log_success "DPO训练和评估流程已完成！" 