#!/bin/bash
# 运行LEDPO Beta分析脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_green() {
  echo -e "${GREEN}$1${NC}"
}

print_yellow() {
  echo -e "${YELLOW}$1${NC}"
}

print_red() {
  echo -e "${RED}$1${NC}"
}

# 默认输出目录
OUTPUT_DIR="./ledpo_analysis_results"
mkdir -p $OUTPUT_DIR

# 检查命令行参数
if [ $# -lt 1 ]; then
  print_red "错误: 请提供输出目录路径作为参数"
  echo "用法: $0 <输出目录路径> [模型名称前缀]"
  exit 1
fi

MODEL_OUTPUT_DIR=$1
PREFIX=${2:-""}

print_green "====== LEDPO Beta 分析工具 ======"
print_yellow "分析目标目录: $MODEL_OUTPUT_DIR"
if [ ! -z "$PREFIX" ]; then
  print_yellow "结果前缀: $PREFIX"
fi

# 检查目录是否存在
if [ ! -d "$MODEL_OUTPUT_DIR" ]; then
  print_red "错误: 输出目录 '$MODEL_OUTPUT_DIR' 不存在!"
  exit 1
fi

# 检查beta_analysis目录
BETA_DIR="$MODEL_OUTPUT_DIR/beta_analysis"
if [ ! -d "$BETA_DIR" ]; then
  print_red "错误: beta_analysis目录不存在于 '$MODEL_OUTPUT_DIR'!"
  print_yellow "提示: 请确保使用了增强版LEDPO Trainer进行训练"
  exit 1
fi

# 检查beta_history.npy文件
BETA_HISTORY="$BETA_DIR/beta_history.npy"
if [ ! -f "$BETA_HISTORY" ]; then
  print_red "错误: beta_history.npy文件不存在于 '$BETA_DIR'!"
  exit 1
fi

# 运行分析脚本
print_green "开始分析LEDPO Beta数据..."
python scripts/analyze_ledpo_beta.py --data "$BETA_HISTORY" --output "$OUTPUT_DIR" --prefix "$PREFIX"

# 检查是否成功
if [ $? -ne 0 ]; then
  print_red "分析过程中出现错误!"
  exit 1
fi

# 显示结果
print_green "\n===== 分析完成 ====="
print_yellow "分析结果保存在: $OUTPUT_DIR"

# 如果生成了结果文件，显示摘要
STATS_FILE="$OUTPUT_DIR/ledpo_beta_stats${PREFIX:+_$PREFIX}.json"
if [ -f "$STATS_FILE" ]; then
  print_green "\n===== 结果摘要 ====="
  # 使用python解析和显示JSON
  python -c "
import json
import sys
try:
    with open('$STATS_FILE', 'r') as f:
        data = json.load(f)
    
    print(f\"初始 beta_scale: {data.get('initial_beta_scale', 0):.4f}\")
    print(f\"最终 beta_scale: {data.get('final_beta_scale', 0):.4f}\")
    print(f\"最终正delta beta值: {data.get('final_pos_beta', 0):.4f}\")
    print(f\"最终负delta beta值: {data.get('final_neg_beta', 0):.4f}\")
    print(f\"pos_beta/neg_beta比值: {data.get('final_pos_neg_ratio', 0):.4f}\")
    print(f\"Beta趋零问题: {'存在' if data.get('has_zero_issue', True) else '不存在'}\")
    print(f\"Beta区分度: {'良好' if data.get('has_good_differentiation', False) else '不足'}\")
    print(f\"总体评价: {'健康' if data.get('success', False) else '存在问题'}\")
except Exception as e:
    print(f'读取结果摘要时出错: {e}')
    sys.exit(1)
  "
fi

# 列出生成的图表文件
print_green "\n生成的图表文件:"
ls -l "$OUTPUT_DIR" | grep -E ".png$|.json$" | awk '{print "  " $9}'

print_green "\n分析完成 🎉" 