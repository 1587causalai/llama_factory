#!/bin/bash

# LEDPO训练修复脚本
# 这个脚本会执行以下操作：
# 1. 备份原始trainer.py文件
# 2. 应用修复patch
# 3. 运行修复后的LEDPO训练
# 4. 可选：恢复原始trainer.py文件

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== LEDPO动态beta修复脚本 =====${NC}"

# 设置路径
TRAINER_PATH="src/llamafactory/train/ledpo/trainer.py"
BACKUP_PATH="src/llamafactory/train/ledpo/trainer.py.bak"
CONFIG_PATH="experiments/ledpo_fixed.yaml"

# 1. 备份原始文件
echo -e "${YELLOW}备份原始trainer.py文件...${NC}"
if [ ! -f "$BACKUP_PATH" ]; then
    cp "$TRAINER_PATH" "$BACKUP_PATH"
    echo "备份完成: $BACKUP_PATH"
else
    echo "备份文件已存在，跳过备份步骤"
fi

# 2. 检查修复脚本是否存在
if [ ! -f "fix_ledpo_beta.py" ]; then
    echo -e "${RED}修复脚本fix_ledpo_beta.py不存在！${NC}"
    exit 1
fi

# 3. 应用修复
echo -e "${YELLOW}应用LEDPO beta修复...${NC}"
echo "修复后的代码已经创建，现在需要手动替换，请按照以下步骤操作:"
echo "1. 打开fix_ledpo_beta.py文件查看修复方案"
echo "2. 按照修复实施步骤部分的指导修改trainer.py"
echo -e "${YELLOW}是否已完成手动修复？(y/n)${NC}"
read confirm
if [ "$confirm" != "y" ]; then
    echo "请完成手动修复后再继续"
    exit 1
fi

# 4. 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}配置文件$CONFIG_PATH不存在！${NC}"
    exit 1
fi

# 5. 运行修复后的训练
echo -e "${GREEN}开始运行修复后的LEDPO训练...${NC}"
llamafactory-cli train "$CONFIG_PATH"

echo -e "${GREEN}训练完成！${NC}"

# 6. 询问是否恢复原始trainer.py
echo -e "${YELLOW}是否恢复原始trainer.py文件？(y/n)${NC}"
read restore
if [ "$restore" = "y" ]; then
    cp "$BACKUP_PATH" "$TRAINER_PATH"
    echo "已恢复原始trainer.py文件"
fi

echo -e "${GREEN}脚本执行完毕！${NC}" 