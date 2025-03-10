#!/bin/bash

# 设置tmux会话名称
SESSION_NAME="valuehead_training_experiment"

# 创建日志目录
mkdir -p /root/LLaMA-Factory/experiments/valuehead_only_training/logs

# 创建tmux脚本文件
TMUX_SCRIPT="/root/LLaMA-Factory/experiments/valuehead_only_training/tmux_commands.sh"

cat > $TMUX_SCRIPT << 'EOF'
#!/bin/bash

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llama

# 切换到项目目录
cd /root/LLaMA-Factory

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
LOG_DIR="experiments/valuehead_only_training/logs"
mkdir -p $LOG_DIR

# 1. 只训练 ValueHead 的实验
echo "========= 开始实验 1: 只训练 ValueHead ========="
python src/train_bash.py \
    --config experiments/valuehead_only_training/qwen1_5_0_5b_valuehead_only.yaml \
    --deepspeed deepspeed/zero2.json \
    2>&1 | tee $LOG_DIR/valuehead_only.log

# 2. 只训练 Policy 的实验
echo "========= 开始实验 2: 只训练 Policy ========="
python src/train_bash.py \
    --config experiments/valuehead_only_training/qwen1_5_0_5b_policy_only.yaml \
    --deepspeed deepspeed/zero2.json \
    2>&1 | tee $LOG_DIR/policy_only.log

# 3. 正常训练两者的实验
echo "========= 开始实验 3: 正常训练 Policy 和 ValueHead ========="
python src/train_bash.py \
    --config experiments/valuehead_only_training/qwen1_5_0_5b_normal_training.yaml \
    --deepspeed deepspeed/zero2.json \
    2>&1 | tee $LOG_DIR/normal_training.log

# 实验完成后，运行分析脚本生成图表
echo "所有实验完成！开始分析结果..."
python experiments/valuehead_only_training/analyze_results.py

echo "实验结束，可以查看 $LOG_DIR 目录下的日志文件和分析结果。"
EOF

chmod +x $TMUX_SCRIPT

# 检查tmux是否已安装
if ! command -v tmux &> /dev/null; then
    echo "tmux未安装，请先安装tmux: apt-get install -y tmux"
    exit 1
fi

# 检查会话是否已存在，如果存在则附加
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "会话 $SESSION_NAME 已存在，将附加到该会话"
    tmux attach -t $SESSION_NAME
    exit 0
fi

# 创建新的tmux会话并在后台运行
echo "创建新的tmux会话: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "$TMUX_SCRIPT" C-m

echo "实验已在tmux会话中启动"
echo "可以使用以下命令查看实验进度："
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "要分离会话，按 Ctrl+b 然后按 d"
echo "实验将在后台继续运行"
echo "日志文件将保存在: /root/LLaMA-Factory/experiments/valuehead_only_training/logs/" 