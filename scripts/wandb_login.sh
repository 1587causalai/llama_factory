#!/bin/bash

# Weights & Biases (wandb) 登录脚本
# 此脚本用于设置wandb的API密钥并登录

echo "===== Weights & Biases (wandb) 登录 ====="
echo "请先确保您已经注册wandb账户: https://wandb.ai"
echo ""

# 检查是否已经安装wandb
if ! command -v wandb &> /dev/null; then
    echo "未找到wandb命令，正在安装..."
    pip install wandb
fi

# 检查是否已经设置API密钥
if [ -f ~/.netrc ] && grep -q "wandb.ai" ~/.netrc; then
    echo "您已经登录wandb。"
    echo "当前登录的账户信息:"
    wandb login --relogin
else
    echo "请输入您的wandb API密钥 (从 https://wandb.ai/settings 获取):"
    read -p "API密钥: " api_key
    
    if [ -z "$api_key" ]; then
        echo "API密钥不能为空，退出登录过程。"
        exit 1
    fi
    
    # 登录wandb
    wandb login "$api_key"
fi

echo ""
echo "wandb登录完成，您现在可以运行训练脚本了。"
echo "例如: bash scripts/test_qwen_foodpo.sh" 