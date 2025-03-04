#!/bin/bash

# DPO训练和评估一键运行脚本

# 设置变量
MODEL_PATH="/root/models/Qwen1.5-0.5B"  # 模型路径
DATA_PATH="Anthropic/hh-rlhf"           # 数据集路径
OUTPUT_DIR="dpo_output"                 # 输出目录
EVAL_DIR="dpo_eval"                     # 评估结果目录
MAX_SAMPLES=1000                        # 最大样本数（训练和评估）
BATCH_SIZE=4                            # 批次大小
EPOCHS=3                                # 训练轮数
BETA=0.1                                # DPO温度参数

# 颜色设置
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 无颜色

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $EVAL_DIR

# 输出配置信息
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}      DPO训练与评估一键脚本          ${NC}"
echo -e "${YELLOW}=====================================${NC}"
echo -e "${GREEN}模型路径:${NC} $MODEL_PATH"
echo -e "${GREEN}数据集:${NC} $DATA_PATH"
echo -e "${GREEN}最大样本数:${NC} $MAX_SAMPLES"
echo -e "${GREEN}批次大小:${NC} $BATCH_SIZE"
echo -e "${GREEN}训练轮数:${NC} $EPOCHS"
echo -e "${GREEN}DPO温度参数:${NC} $BETA"
echo -e "${YELLOW}=====================================${NC}"

# 确认运行
read -p "是否开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消运行"
    exit 1
fi

# 开始训练
echo -e "${YELLOW}=====================================${NC}"
echo -e "${GREEN}开始DPO训练...${NC}"
echo -e "${YELLOW}=====================================${NC}"

python train_dpo.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --max_samples $MAX_SAMPLES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --beta $BETA \
    --debug

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}训练过程发生错误，退出脚本${NC}"
    exit 1
fi

# 开始评估
echo -e "${YELLOW}=====================================${NC}"
echo -e "${GREEN}开始DPO评估...${NC}"
echo -e "${YELLOW}=====================================${NC}"

python evaluate_dpo.py \
    --base_model_path $MODEL_PATH \
    --dpo_model_path "$OUTPUT_DIR/final-model" \
    --eval_data_path $DATA_PATH \
    --output_dir $EVAL_DIR \
    --max_samples 100 \
    --batch_size $BATCH_SIZE \
    --beta $BETA

# 输出完成信息
if [ $? -eq 0 ]; then
    echo -e "${YELLOW}=====================================${NC}"
    echo -e "${GREEN}DPO训练和评估已完成!${NC}"
    echo -e "${GREEN}训练结果保存在:${NC} $OUTPUT_DIR"
    echo -e "${GREEN}评估结果保存在:${NC} $EVAL_DIR"
    echo -e "${YELLOW}=====================================${NC}"
else
    echo -e "${YELLOW}评估过程发生错误${NC}"
fi 