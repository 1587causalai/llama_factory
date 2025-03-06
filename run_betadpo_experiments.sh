#!/bin/bash

# 设置环境
# 适应InternStudio环境
conda activate llama

# 实验参数数组
BETA_SCALES=(0 0.1 0.5)

# 原始配置文件
CONFIG_FILE="examples/train_lora/qwen1_5_0_5b_lora_betadpo.yaml"

# 循环运行每个实验
for scale in "${BETA_SCALES[@]}"; do
    echo "==================================================="
    echo "开始运行实验: pref_beta_scale = $scale"
    echo "==================================================="
    
    # 创建临时配置文件
    TMP_CONFIG="examples/train_lora/tmp_config_scale_${scale}.yaml"
    cp "$CONFIG_FILE" "$TMP_CONFIG"
    
    # 修改临时配置文件中的pref_beta_scale值
    sed -i "s/pref_beta_scale:.*/pref_beta_scale: $scale  # 动态beta缩放因子/" "$TMP_CONFIG"
    
    # 修改输出目录
    OUTPUT_DIR="saves/qwen1.5-0.5b/lora/scale${scale}"
    sed -i "s|output_dir:.*|output_dir: $OUTPUT_DIR|" "$TMP_CONFIG"
    
    echo "配置文件已更新，输出目录: $OUTPUT_DIR"
    
    # 设置WANDB项目
    export WANDB_PROJECT="betadpo_scale_experiments"
    
    # 运行训练
    echo "开始训练..."
    python dpo_baseline/run_betadpo_detailed.py "$TMP_CONFIG"
    
    # 清理临时配置文件
    rm "$TMP_CONFIG"
    
    echo "实验完成: pref_beta_scale = $scale"
    echo ""
done

echo "所有实验已完成!" 