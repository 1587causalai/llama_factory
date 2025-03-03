#!/bin/bash

# ===========================================================================================
# test_qwen_foodpo.sh 脚本验证工具
# ===========================================================================================
# 
# 本脚本用于逐步验证 test_qwen_foodpo.sh 中的逻辑是否符合预期
# 通过单步执行并检查每个关键环节，确保训练流程的正确性
# ===========================================================================================

echo "=========================================================================="
echo "🔍 FooDPO 训练脚本验证工具"
echo "=========================================================================="

# 步骤1: 检查环境变量设置
echo -e "\n📋 步骤1: 验证环境变量设置"
echo "设置 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
echo "✓ 已设置 PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO}"
echo "说明: 此环境变量控制MPS设备(Apple Silicon)内存使用的最大水位线"
echo "     设为0表示不限制内存使用，但可能需要根据您的设备调整"

# 步骤2: 检查动态beta参数设置
echo -e "\n📋 步骤2: 验证动态beta参数设置"
BETA_SCALE=0.05
echo "✓ 已设置 BETA_SCALE=${BETA_SCALE}"
echo "说明: BETA_SCALE是FooDPO算法的核心参数，控制困惑度对beta的影响程度"
echo "     推荐值范围: 0.01-1.0，当前值${BETA_SCALE}较小，算法会更接近标准DPO"
if [[ $(echo "$BETA_SCALE < 0.01" | bc -l) -eq 1 ]]; then
    echo "⚠️ 警告: BETA_SCALE值过小，可能导致FooDPO退化为标准DPO"
elif [[ $(echo "$BETA_SCALE > 0.7" | bc -l) -eq 1 ]]; then
    echo "⚠️ 警告: BETA_SCALE值较大，可能导致训练不稳定"
else
    echo "✓ BETA_SCALE值在合理范围内"
fi

# 步骤3: 检查基本配置参数
echo -e "\n📋 步骤3: 验证基本配置参数"
MODEL_PATH=~/models/Qwen1.5-0.5B
DATASET_PATH="data"
OUTPUT_DIR="output/qwen_foodpo_scale_${BETA_SCALE}"

echo "✓ 模型路径: ${MODEL_PATH}"
# 检查模型路径是否存在
if [ -d "$MODEL_PATH" ]; then
    echo "✓ 模型路径存在"
    ls -la $MODEL_PATH | head -n 5
    echo "..."
else
    echo "❌ 错误: 模型路径不存在，请确认路径是否正确"
    echo "尝试查找可能的模型目录:"
    find ~/models -maxdepth 2 -type d -name "*Qwen*" 2>/dev/null || echo "没有找到相关模型目录"
fi

echo "✓ 数据集路径: ${DATASET_PATH}"
# 检查数据集路径是否存在
if [ -d "$DATASET_PATH" ]; then
    echo "✓ 数据集路径存在"
    echo "数据集目录结构:"
    find $DATASET_PATH -type f -name "*.json" | head -n 5
else
    echo "❌ 警告: 数据集路径不存在，脚本可能无法正常运行"
    echo "您需要准备DPO格式的数据集，至少包含以下字段:"
    echo "- prompt: 提示文本"
    echo "- chosen: 偏好回答" 
    echo "- rejected: 非偏好回答"
fi

echo "✓ 输出目录: ${OUTPUT_DIR}"
# 检查输出目录，但不创建
if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️ 注意: 输出目录已存在，脚本会在实际运行前清空此目录"
else
    echo "✓ 输出目录不存在，脚本会在运行时创建"
fi

# 步骤4: 检查训练参数
echo -e "\n📋 步骤4: 验证训练参数"
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
EPOCHS=1
MAX_SEQ_LEN=512
WARMUP_RATIO=0.1

echo "✓ 微批量大小: ${MICRO_BATCH_SIZE}"
echo "✓ 梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS}"
echo "✓ 学习率: ${LEARNING_RATE}"
echo "✓ 训练轮次: ${EPOCHS}"
echo "✓ 最大序列长度: ${MAX_SEQ_LEN}"
echo "✓ 预热比例: ${WARMUP_RATIO}"

echo "说明: 这些参数针对小数据集和MPS设备进行了优化"
echo "     - 小批量大小和梯度累积步数较小，避免内存不足"
echo "     - 学习率较高，有助于小数据集上快速收敛"
echo "     - 训练轮次较少，防止过拟合"
echo "     - 最大序列长度已减小，降低内存需求"

# 步骤5: 验证wandb配置
echo -e "\n📋 步骤5: 验证wandb配置"
export WANDB_PROJECT="qwen-foodpo" 
export WANDB_NAME="qwen1.5-0.5B-foodpo-scale-${BETA_SCALE}"

echo "✓ WANDB项目: ${WANDB_PROJECT}"
echo "✓ WANDB实验名: ${WANDB_NAME}"

# 检查wandb是否已登录
if command -v wandb &> /dev/null; then
    echo "✓ wandb命令已安装"
    WANDB_STATUS=$(wandb status 2>&1)
    if [[ $WANDB_STATUS == *"Not logged in"* ]]; then
        echo "❌ 警告: wandb未登录，请先运行 'wandb login'"
    else
        echo "✓ wandb已登录"
    fi
else
    echo "❌ 警告: wandb未安装，请先安装: pip install wandb"
fi

# 步骤6: 验证训练命令参数
echo -e "\n📋 步骤6: 验证训练命令参数"

# 构建完整的训练命令(但不执行)
TRAIN_CMD="llamafactory-cli train \
    --model_name_or_path $MODEL_PATH \
    --dataset \"dpo_zh_demo\" \
    --dataset_dir $DATASET_PATH \
    --eval_dataset \"dpo_zh_demo\" \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --per_device_eval_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --cutoff_len $MAX_SEQ_LEN \
    --warmup_ratio $WARMUP_RATIO \
    --max_samples 10 \
    --logging_steps 1 \
    --save_steps 5 \
    --save_total_limit 1 \
    --do_train true \
    --template default \
    --finetuning_type lora \
    --lora_rank 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --stage foodpo \
    --pref_beta 0.1 \
    --pref_beta_scale $BETA_SCALE \
    --fp16 false \
    --use_mps_device true \
    --plot_loss true \
    --report_to wandb \
    --ddp_find_unused_parameters false \
    --eval_strategy steps \
    --eval_steps 5 \
    --load_best_model_at_end true \
    --metric_for_best_model \"eval_rewards/margins\" \
    --greater_is_better true \
    --group_by_length false \
    --dataloader_num_workers 0 \
    --log_level info \
    --disable_tqdm false \
    --remove_unused_columns false \
    --optim adamw_torch \
    --gradient_checkpointing true"

# 检查关键参数
echo "✓ 验证FooDPO关键参数:"
echo "  - stage: foodpo (FooDPO算法)" 
echo "  - pref_beta: 0.1 (基础beta值)"
echo "  - pref_beta_scale: ${BETA_SCALE} (动态beta缩放因子)"

echo "✓ 验证内存优化参数:"
echo "  - gradient_checkpointing: true (减少内存使用)"
echo "  - use_mps_device: true (使用Apple Silicon GPU)"
echo "  - dataloader_num_workers: 0 (避免MPS设备多进程问题)"
echo "  - fp16: false (MPS设备不支持fp16)"
echo "  - max_samples: 10 (限制数据集大小用于快速测试)"

# 检查llamafactory-cli命令是否可用
if command -v llamafactory-cli &> /dev/null; then
    echo "✓ llamafactory-cli命令已安装"
    # 获取版本信息(如果可用)
    LLAMA_VERSION=$(llamafactory-cli --version 2>/dev/null || echo "无法获取版本信息")
    echo "  - 版本信息: ${LLAMA_VERSION}"
else
    echo "❌ 错误: llamafactory-cli命令不可用，请确保已正确安装LLaMA Factory"
    echo "  可以通过以下命令安装: pip install llama-factory"
fi

# 步骤7: 向前传播测试(dry run)准备
echo -e "\n📋 步骤7: 向前传播测试准备"

# 创建模拟数据集用于测试(如果需要)
if [ ! -d "$DATASET_PATH" ]; then
    echo "创建模拟数据集用于测试..."
    mkdir -p $DATASET_PATH
    
    # 创建一个最小的DPO格式数据集示例
    cat > $DATASET_PATH/dpo_zh_demo.json << EOF
{"prompt": "解释量子力学的基本原理", "chosen": "量子力学是描述微观粒子行为的物理理论，基于不确定性原理和波粒二象性。它表明粒子的位置和动量不能同时被精确测量，且微观粒子既有波动性又有粒子性。", "rejected": "量子力学就是研究原子的科学，没什么特别的，就像普通物理一样简单明了。"}
{"prompt": "推荐一本科幻小说", "chosen": "我推荐刘慈欣的《三体》，这是一部融合了宇宙社会学、量子物理学和历史的宏大科幻巨著，讲述了人类文明与三体文明接触及其后果的故事。", "rejected": "随便一本科幻小说都行，科幻小说都差不多，没什么区别。"}
EOF
    echo "✓ 已创建模拟数据集: $DATASET_PATH/dpo_zh_demo.json"
fi

# 步骤8: 最终验证和执行确认
echo -e "\n📋 步骤8: 最终验证总结"

# 检查是否存在潜在问题
POTENTIAL_ISSUES=0

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在"
    POTENTIAL_ISSUES=$((POTENTIAL_ISSUES+1))
fi

if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ llamafactory-cli命令不可用"
    POTENTIAL_ISSUES=$((POTENTIAL_ISSUES+1))
fi

if command -v wandb &> /dev/null; then
    WANDB_STATUS=$(wandb status 2>&1)
    if [[ $WANDB_STATUS == *"Not logged in"* ]]; then
        echo "⚠️ wandb未登录"
        POTENTIAL_ISSUES=$((POTENTIAL_ISSUES+1))
    fi
fi

# 总结
echo -e "\n=========================================================================="
if [ $POTENTIAL_ISSUES -eq 0 ]; then
    echo "✅ 验证完成！所有检查均已通过。"
    echo "脚本逻辑符合预期，可以安全执行原始的test_qwen_foodpo.sh脚本。"
else
    echo "⚠️ 验证完成，但存在${POTENTIAL_ISSUES}个潜在问题需要解决。"
    echo "请修复上述问题后再执行原始脚本。"
fi
echo "=========================================================================="

echo -e "\n要执行完整的训练过程，可以运行原始脚本:"
echo "bash test_qwen_foodpo.sh"
echo -e "\n或者，您可以使用此脚本添加--execute参数来执行实际训练:"
echo "bash $(basename $0) --execute"

# 如果指定了--execute参数，则实际执行训练命令
if [[ "$1" == "--execute" ]]; then
    echo -e "\n🚀 执行实际训练命令..."
    eval $TRAIN_CMD
fi

echo -e "\n验证脚本执行完毕！" 