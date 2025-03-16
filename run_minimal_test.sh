#!/bin/bash

# 极简化LEDPO测试运行脚本
# 这个脚本运行两个极简化版本的LEDPO实现进行测试和对比

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== LEDPO最小化实现测试 =====${NC}"

# 创建结果目录
mkdir -p results

echo -e "${YELLOW}1. 运行基本版最小化LEDPO实现${NC}"
# 运行基本版
python minimal_ledpo.py

echo -e "${YELLOW}2. 运行多变体对比实验${NC}"
# 运行变体对比
python minimal_ledpo_variants.py

echo -e "${GREEN}测试完成! 结果保存在results目录${NC}"

# 打印结果摘要
if [ -f "results/minimal_ledpo_results.txt" ]; then
    echo -e "${YELLOW}基本版测试结果:${NC}"
    cat results/minimal_ledpo_results.txt
fi

if [ -f "results/ledpo_variants_results.txt" ]; then
    echo -e "${YELLOW}多变体测试结果:${NC}"
    cat results/ledpo_variants_results.txt
fi

echo -e "${GREEN}图表已保存到results目录，可以查看以下文件:${NC}"
echo "  - results/minimal_ledpo_normal.png"
echo "  - results/minimal_ledpo_freeze_policy_model.png"
echo "  - results/ledpo不同变体对比.png" 