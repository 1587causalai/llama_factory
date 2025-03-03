#!/bin/bash

# 训练结果分析脚本 - 帮助分析fooDPO训练过程和结果
# 此脚本在训练完成后运行，显示关键指标

# 基本配置参数
OUTPUT_DIR="output/qwen_foodpo_test"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # 无颜色

echo -e "${GREEN}===== Qwen fooDPO 训练结果分析 =====${NC}"

# 显示训练信息摘要
echo -e "${BLUE}【训练信息摘要】${NC}"
echo "输出目录: $OUTPUT_DIR"

# 检查并显示最终损失
if [ -f "$OUTPUT_DIR/trainer_log.jsonl" ]; then
    echo -e "${BLUE}【训练损失】${NC}"
    # 提取最后10个loss值
    echo "最近的训练损失值:"
    tail -n 10 "$OUTPUT_DIR/trainer_log.jsonl" | grep -o '"loss":[0-9.]*' | cut -d':' -f2
    
    # 计算平均损失
    echo "最终平均损失:"
    tail -n 10 "$OUTPUT_DIR/trainer_log.jsonl" | grep -o '"loss":[0-9.]*' | cut -d':' -f2 | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; else print "没有损失数据"; }'
else
    echo -e "${YELLOW}未找到训练日志文件 trainer_log.jsonl${NC}"
fi

# 检查是否有wandb相关文件
if [ -f "$OUTPUT_DIR/wandb/latest-run/run-*.wandb" ] || [ -d "$OUTPUT_DIR/wandb" ]; then
    echo -e "${BLUE}【Weights & Biases 可视化】${NC}"
    echo "您的训练过程已记录到wandb，请访问以下网址查看详细训练过程和指标:"
    echo "https://wandb.ai/$(whoami)/qwen-foodpo"
    echo "或者运行以下命令打开wandb界面:"
    echo "wandb dashboard"
fi

# 检查模型保存情况
CHECKPOINT_DIRS=$(find "$OUTPUT_DIR" -type d -name "checkpoint-*" | sort)
if [ -n "$CHECKPOINT_DIRS" ]; then
    echo -e "${BLUE}【保存的检查点】${NC}"
    for dir in $CHECKPOINT_DIRS; do
        echo "- $dir"
    done
    
    # 显示最新检查点信息
    LATEST_CHECKPOINT=$(echo "$CHECKPOINT_DIRS" | tail -n 1)
    echo -e "${BLUE}【最新检查点】${NC} $LATEST_CHECKPOINT"
    
    if [ -f "$LATEST_CHECKPOINT/trainer_state.json" ]; then
        echo "训练步数: $(grep -o '"global_step":[0-9]*' "$LATEST_CHECKPOINT/trainer_state.json" | cut -d':' -f2)"
        echo "最佳指标: $(grep -o '"best_metric":[0-9.]*' "$LATEST_CHECKPOINT/trainer_state.json" | cut -d':' -f2 2>/dev/null || echo "无")"
    fi
else
    echo -e "${YELLOW}未找到任何模型检查点${NC}"
fi

# 显示使用指导
echo -e "${GREEN}===== 使用指导 =====${NC}"
echo "1. 查看详细训练曲线和指标: wandb dashboard"
echo "2. 使用训练好的模型进行推理: llamafactory-cli chat --model_name_or_path $OUTPUT_DIR"
echo "3. 合并LoRA权重: llamafactory-cli export --model_name_or_path ~/models/Qwen1.5-0.5B --adapter_name_or_path $OUTPUT_DIR --export_format pytorch" 