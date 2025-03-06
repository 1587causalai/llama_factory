#!/bin/bash

# 设置环境
# 适应InternStudio环境
conda activate llama

# 实验参数数组 - 初始beta_scale值
BETA_SCALES=(0.01 0.1 0.5 1.0)

# 原始配置文件
CONFIG_FILE="examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml"

# 循环运行每个实验
for scale in "${BETA_SCALES[@]}"; do
    echo "==================================================="
    echo "开始运行LEDPO实验: 初始beta_scale = $scale"
    echo "==================================================="
    
    # 创建临时配置文件
    TMP_CONFIG="examples/train_lora/tmp_ledpo_scale_${scale}.yaml"
    cp "$CONFIG_FILE" "$TMP_CONFIG"
    
    # 修改原始beta值
    sed -i "s/pref_beta:.*/pref_beta: 0.1  # 基础beta值/" "$TMP_CONFIG"
    
    # 修改输出目录
    OUTPUT_DIR="saves/qwen1.5-0.5b/lora/ledpo_scale${scale}"
    sed -i "s|output_dir:.*|output_dir: $OUTPUT_DIR|" "$TMP_CONFIG"
    
    # 修改运行名称
    RUN_NAME="ledpo_scale_${scale}"
    sed -i "s|run_name:.*|run_name: $RUN_NAME|" "$TMP_CONFIG"
    
    echo "配置文件已更新，输出目录: $OUTPUT_DIR"
    
    # 设置WANDB项目
    export WANDB_PROJECT="ledpo_experiments"
    
    # 运行训练
    echo "开始训练LEDPO..."
    python dpo_baseline/run_ledpo_detailed.py "$TMP_CONFIG"
    
    # 清理临时配置文件
    rm "$TMP_CONFIG"
    
    echo "实验完成: 初始beta_scale = $scale"
    echo ""
done

echo "所有LEDPO实验已完成!" 