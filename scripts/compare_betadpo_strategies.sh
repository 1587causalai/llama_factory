#!/bin/bash
#==================================================================================
# BetaDPO算法演示 - PPL量纲系数对比与固定beta基线比较 (轻量级演示版)
#==================================================================================
# 
# 演示目的:
# 该脚本演示BetaDPO算法中不同PPL量纲系数的效果对比，使用轻量级配置适合快速测试。
# BetaDPO的核心思想是根据困惑度(PPL)动态调整beta值: β(x) = c · log(PPL(x)) · β_base
# 
# 实验模型与数据:
# - 模型: Qwen1.5-0.5B (轻量级中文大语言模型)
# - 数据集: dpo_zh_demo.json (中文偏好对齐数据集，仅使用少量样本)
# - 训练方式: LoRA微调 (仅训练少量参数)
#
# 快速对比:
# 1. 固定beta策略 (baseline): 常量beta=0.3
# 2. 动态beta策略 (BetaDPO): 量纲系数c=0.1
#
#==================================================================================

# 错误处理函数
handle_error() {
    local exit_code=$?
    local scale=$1
    echo "❌ 训练过程中出现错误 (退出代码: $exit_code)" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
    # 继续执行后续实验
    return 0
}

# 基础配置
MODEL_PATH=~/models/Qwen1.5-0.5B
DATASET_PATH="data"
BASE_OUTPUT_DIR="output/betadpo_demo"

# 确保输出目录存在
mkdir -p $BASE_OUTPUT_DIR

# 记录开始时间
echo "开始BetaDPO演示实验: $(date)" | tee $BASE_OUTPUT_DIR/comparison_log.txt
echo "使用模型: $MODEL_PATH" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
echo "数据集目录: $DATASET_PATH" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt

# 基本训练参数 - 轻量级demo设置
MICRO_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-5
EPOCHS=0.5    # 只训练半个epoch
MAX_SEQ_LEN=512  # 减少序列长度
WARMUP_RATIO=0.03
MAX_SAMPLES=20   # 限制样本数量加速训练

# 只测试两个量纲系数：固定beta(0.0)和动态beta(0.1)
PPL_SCALES=("0.0" "0.1")

# 记录开始的实验数量
TOTAL_EXPERIMENTS=${#PPL_SCALES[@]}
SUCCESSFUL_EXPERIMENTS=0

for scale in "${PPL_SCALES[@]}"; do
    OUTPUT_DIR="$BASE_OUTPUT_DIR/scale_${scale}"
    mkdir -p $OUTPUT_DIR
    
    echo "====================================================" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
    echo "开始测试 PPL量纲系数 c=${scale}: $(date)" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
    
    # 设置beta scale参数
    if [ "$scale" == "0.0" ]; then
        # 对于c=0，相当于固定beta策略(基线)
        BETA_SCALE_ARG="--pref_beta_scale 0.0"
        echo "策略: 固定beta (baseline)" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
    else
        # 对于其他c值，使用PPL动态调整beta
        BETA_SCALE_ARG="--pref_beta_scale $scale"
        echo "策略: 动态beta (c=${scale})" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
    fi
    
    # 执行训练 - 使用llamafactory-cli命令
    # 添加错误处理
    set +e  # 不终止脚本，即使命令失败
    
    llamafactory-cli train \
        --model_name_or_path $MODEL_PATH \
        --dataset "dpo_zh_demo" \
        --dataset_dir $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size $MICRO_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $EPOCHS \
        --cutoff_len $MAX_SEQ_LEN \
        --warmup_ratio $WARMUP_RATIO \
        --logging_steps 1 \
        --save_steps 20 \
        --save_total_limit 1 \
        --do_train true \
        --lr_scheduler_type cosine \
        --template default \
        --finetuning_type lora \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --stage betadpo \
        --pref_beta 0.3 \
        --max_samples $MAX_SAMPLES \
        $BETA_SCALE_ARG \
        --fp16 false \
        --use_mps_device true \
        --max_grad_norm 1.0 \
        --seed 42 \
        --plot_loss true \
        2>&1 | tee -a $OUTPUT_DIR/training_log.txt
    
    # 收集beta值和PPL数据
    python -c "
import re
import json
import os

output_dir = '${OUTPUT_DIR}'
log_file = os.path.join(output_dir, 'training_log.txt')
beta_ppl_data = []

# 从日志中提取beta和PPL信息
with open(log_file, 'r') as f:
    for line in f:
        # 查找包含beta/current_value的行
        beta_match = re.search(r'beta/current_value.: (\d+\.\d+)', line)
        ppl_match = re.search(r'ppl.: (\d+\.\d+)', line)
        
        if beta_match:
            beta_value = float(beta_match.group(1))
            # 可能在同一行找到PPL，如果没有则为None
            ppl_value = float(ppl_match.group(1)) if ppl_match else None
            beta_ppl_data.append({'beta': beta_value, 'ppl': ppl_value})
        # 如果beta和ppl在不同行，需要做额外处理...

# 保存提取的数据
with open(os.path.join(output_dir, 'beta_ppl_data.json'), 'w') as f:
    json.dump(beta_ppl_data, f)

print(f'已收集并保存beta和PPL数据到 {output_dir}/beta_ppl_data.json')
" 2>&1 | tee -a $OUTPUT_DIR/beta_analysis.log || echo "Beta值数据收集未成功，可能无此信息"
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 成功完成 PPL量纲系数 c=${scale} 测试: $(date)" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS+1))
    else
        handle_error $scale
    fi
    
    # 重新启用错误终止
    set -e
    
    echo "====================================================" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
done

echo "所有BetaDPO演示实验完成: $(date)" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
echo "成功完成 $SUCCESSFUL_EXPERIMENTS / $TOTAL_EXPERIMENTS 个实验" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt

# 收集结果和比较
echo "生成结果比较报告..." | tee -a $BASE_OUTPUT_DIR/comparison_log.txt

# 使用try-catch风格来处理可能的Python错误
{
python -c '
import os
import sys
import json
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    base_dir = "output/betadpo_demo"
    scales = ["0.0", "0.1"]
    results = {}
    available_scales = []
    
    # 收集训练指标
    print("开始收集训练数据...")
    for scale in scales:
        scale_dir = os.path.join(base_dir, f"scale_{scale}")
        train_file = os.path.join(scale_dir, "trainer_state.json")
        
        print(f"检查文件: {train_file}")
        if os.path.exists(train_file):
            print(f"  ✓ 文件存在")
            try:
                with open(train_file, "r") as f:
                    data = json.load(f)
                
                print(f"  ✓ 成功读取JSON数据")
                
                # 提取loss和rewards/accuracies
                log_history = data.get("log_history", [])
                print(f"  - 日志条目数: {len(log_history)}")
                
                if log_history:
                    # 从第一条日志中提取loss和beta (关键修复)
                    first_log = log_history[0] if len(log_history) > 0 else {}
                    last_log = log_history[-1] if len(log_history) > 0 else {}
                    
                    # 现在正确地提取数据
                    loss_val = first_log.get("loss", 0)
                    beta_val = first_log.get("beta/current_value", None)
                    step_val = last_log.get("step", 0)
                    
                    # 创建结果字典
                    results[scale] = {
                        "step": step_val,
                        "loss": loss_val,
                        "beta_value": beta_val
                    }
                    available_scales.append(scale)
                    print(f"  ✓ 成功加载 scale_{scale} 的训练数据: loss={loss_val}, beta={beta_val}")
                else:
                    print(f"  ✗ 未找到日志数据")
            except Exception as e:
                print(f"  ✗ 处理 scale_{scale} 数据时出错: {str(e)}")
                traceback.print_exc()
        else:
            # 尝试找到checkpoint目录中的trainer_state.json
            checkpoints = [d for d in os.listdir(scale_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(scale_dir, d))]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                checkpoint_file = os.path.join(scale_dir, latest_checkpoint, "trainer_state.json")
                print(f"尝试从checkpoint读取: {checkpoint_file}")
                
                if os.path.exists(checkpoint_file):
                    print(f"  ✓ 找到checkpoint文件")
                    try:
                        with open(checkpoint_file, "r") as f:
                            data = json.load(f)
                        
                        print(f"  ✓ 成功读取checkpoint JSON数据")
                        
                        # 提取loss和rewards/accuracies (与上面相同的逻辑)
                        log_history = data.get("log_history", [])
                        print(f"  - 日志条目数: {len(log_history)}")
                        
                        if log_history:
                            # 从第一条日志中提取loss和beta (关键修复)
                            first_log = log_history[0] if len(log_history) > 0 else {}
                            last_log = log_history[-1] if len(log_history) > 0 else {}
                            
                            # 现在正确地提取数据
                            loss_val = first_log.get("loss", 0)
                            beta_val = first_log.get("beta/current_value", None)
                            step_val = last_log.get("step", 0)
                            
                            # 创建结果字典
                            results[scale] = {
                                "step": step_val,
                                "loss": loss_val,
                                "beta_value": beta_val
                            }
                            available_scales.append(scale)
                            print(f"  ✓ 成功从checkpoint加载 scale_{scale} 的训练数据")
                        else:
                            print(f"  ✗ checkpoint中未找到日志数据")
                    except Exception as e:
                        print(f"  ✗ 处理checkpoint数据时出错: {str(e)}")
                        traceback.print_exc()
                else:
                    print(f"  ✗ checkpoint文件不存在")
            else:
                print(f"  ✗ 文件不存在且未找到checkpoint目录")
    
    # 添加beta-ppl分析
    for scale in scales:
        if scale != "0.0":  # 只对动态beta进行分析
            scale_dir = os.path.join(base_dir, f"scale_{scale}")
            beta_ppl_file = os.path.join(scale_dir, "beta_ppl_data.json")
            
            if os.path.exists(beta_ppl_file):
                try:
                    with open(beta_ppl_file, "r") as f:
                        beta_ppl_data = json.load(f)
                    
                    if scale in results:
                        results[scale]["beta_ppl_data"] = beta_ppl_data
                        print(f"  ✓ 成功加载 scale_{scale} 的beta-PPL数据")
                except Exception as e:
                    print(f"  ✗ 加载beta-PPL数据出错: {str(e)}")
            else:
                print(f"  ✗ 未找到beta-PPL数据文件: {beta_ppl_file}")

    if not available_scales:
        print("\n❌ 错误: 没有找到任何可用的训练数据。")
        # 创建一个空白图像以显示错误
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No training data found", 
                ha="center", va="center", fontsize=20)
        plt.title("Data Loading Error")
        plt.axis("off")
        plt.savefig(os.path.join(base_dir, "demo_results.png"))
        sys.exit(1)

    print(f"\n✓ 成功加载 {len(available_scales)} 个实验的数据")
    
    # 创建一个多面板图表进行全面比较
    plt.figure(figsize=(15, 15))
    
    # 1. 损失值对比 (柱状图)
    plt.subplot(3, 2, 1)
    labels = []
    loss_values = []
    for scale in available_scales:
        labels.append("Fixed Beta" if scale == "0.0" else f"Dynamic Beta (c={scale})")
        loss_value = results[scale]["loss"]
        loss_values.append(loss_value)
    
    x_pos = np.arange(len(labels))
    bars = plt.bar(x_pos, loss_values)
    plt.xticks(x_pos, labels)
    plt.ylabel("Loss Value")
    plt.title("Final Loss Comparison")
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{height:.4f}", ha="center", va="bottom")
    
    # 2. 损失随时间变化曲线 - 我们只有一个点，所以暂时留空或使用一个简单的图表
    plt.subplot(3, 2, 2)
    plt.text(0.5, 0.5, "Only one data point available,\nno trend to show", 
            ha="center", va="center", fontsize=12)
    plt.axis("off")
    plt.title("Loss Change (Not Available)")
    
    # 3. Beta值对比
    plt.subplot(3, 2, 3)
    labels = []
    beta_values = []
    for scale in available_scales:
        beta_val = results[scale]["beta_value"]
        if beta_val is not None:  # 确保有beta值
            labels.append("Fixed Beta" if scale == "0.0" else f"Dynamic Beta (c={scale})")
            beta_values.append(beta_val)
    
    if beta_values:  # 确保有数据
        x_pos = np.arange(len(labels))
        bars = plt.bar(x_pos, beta_values)
        plt.xticks(x_pos, labels)
        plt.ylabel("Beta Value")
        plt.title("Beta Value Comparison")
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f"{height:.3f}", ha="center", va="bottom")
    
    # 4. Beta与PPL关系分析 (对动态beta)
    plt.subplot(3, 2, 4)
    for scale in available_scales:
        if scale != "0.0" and "beta_ppl_data" in results[scale]:
            beta_values = [item["beta"] for item in results[scale]["beta_ppl_data"] if item["beta"] is not None]
            ppl_values = [item["ppl"] for item in results[scale]["beta_ppl_data"] if item["ppl"] is not None and item["beta"] is not None]
            
            if beta_values and ppl_values and len(beta_values) == len(ppl_values):
                plt.scatter(ppl_values, beta_values, alpha=0.7, label=f"c={scale}")
                
                # 添加趋势线
                if len(beta_values) > 1:
                    try:
                        z = np.polyfit(ppl_values, beta_values, 1)
                        p = np.poly1d(z)
                        plt.plot(ppl_values, p(ppl_values), "r--", alpha=0.5)
                    except:
                        pass
                
                plt.xlabel("Perplexity (PPL)")
                plt.ylabel("Beta Value")
                plt.title("Beta vs PPL Relationship")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.5)
            else:
                plt.text(0.5, 0.5, "No Beta-PPL data available", 
                        ha="center", va="center", fontsize=12)
                plt.axis("off")
    
    # 5. 训练速度分析 - 留空或简单显示
    plt.subplot(3, 2, 5)
    plt.text(0.5, 0.5, "Training speed analysis requires\nmultiple data points", 
            ha="center", va="center", fontsize=12)
    plt.axis("off")
    plt.title("Training Speed (Not Available)")
    
    # 6. 算法比较摘要表格
    plt.subplot(3, 2, 6)
    plt.axis("off")  # 关闭坐标轴
    
    # 创建表格数据
    table_data = []
    table_data.append(["Metric", "Fixed Beta", "Dynamic Beta", "Improvement"])
    
    # 损失值行
    fixed_loss = loss_values[0] if len(loss_values) > 0 else None
    dynamic_loss = loss_values[1] if len(loss_values) > 1 else None
    
    if fixed_loss is not None and dynamic_loss is not None:
        # 添加除零保护
        if fixed_loss != 0:
            loss_improvement = (fixed_loss - dynamic_loss) / fixed_loss * 100
            loss_row = ["Loss", f"{fixed_loss:.4f}", f"{dynamic_loss:.4f}", f"{loss_improvement:.2f}%"]
        else:
            loss_row = ["Loss", f"{fixed_loss:.4f}", f"{dynamic_loss:.4f}", "N/A (base=0)"]
    else:
        loss_row = ["Loss", "N/A", "N/A", "N/A"]
    
    table_data.append(loss_row)
    
    # Beta值行
    fixed_beta = beta_values[0] if len(beta_values) > 0 else None
    dynamic_beta = beta_values[1] if len(beta_values) > 1 else None
    
    if fixed_beta is not None and dynamic_beta is not None:
        beta_row = ["Beta", f"{fixed_beta:.3f}", f"{dynamic_beta:.3f}", "N/A"]
    else:
        beta_row = ["Beta", "N/A", "N/A", "N/A"]
    
    table_data.append(beta_row)
    
    # 绘制表格
    plt.table(cellText=table_data, cellLoc="center", loc="center", bbox=[0.1, 0.1, 0.8, 0.8])
    plt.title("BetaDPO Algorithm Comparison")
    
    plt.tight_layout()
    output_path = os.path.join(base_dir, "demo_results.png")
    plt.savefig(output_path, dpi=300)
    print(f"✓ 结果已保存到: {output_path}")
    
    # 生成指标比较表
    comparison_summary = {
        "Metric": ["Final Loss", "Beta Value", "Convergence Steps", "Beta-PPL Correlation"],
        "Fixed Beta": [],
        "Dynamic Beta(c=0.1)": [],
        "Relative Improvement": []
    }
    
    # 填充比较表
    for scale in available_scales:
        if scale == "0.0":
            # 固定beta
            comparison_summary["Fixed Beta"].append(f"{results[scale]['loss']:.4f}")
            comparison_summary["Fixed Beta"].append(f"{results[scale]['beta_value']:.3f}")
            
            # 收敛速度
            comparison_summary["Fixed Beta"].append("N/A (single point)")
            comparison_summary["Fixed Beta"].append("Fixed Value")
        elif scale == "0.1":
            # 动态beta
            comparison_summary["Dynamic Beta(c=0.1)"].append(f"{results[scale]['loss']:.4f}")
            comparison_summary["Dynamic Beta(c=0.1)"].append(f"{results[scale]['beta_value']:.3f}")
            
            # 收敛速度
            comparison_summary["Dynamic Beta(c=0.1)"].append("N/A (single point)")
            
            # 相关性
            if "beta_ppl_data" in results[scale]:
                beta_values = [item["beta"] for item in results[scale]["beta_ppl_data"] if item["beta"] is not None]
                ppl_values = [item["ppl"] for item in results[scale]["beta_ppl_data"] if item["ppl"] is not None and item["beta"] is not None]
                
                if beta_values and ppl_values and len(beta_values) == len(ppl_values) and len(beta_values) > 1:
                    try:
                        correlation = np.corrcoef(ppl_values, beta_values)[0, 1]
                        comparison_summary["Dynamic Beta(c=0.1)"].append(f"{correlation:.3f}")
                    except:
                        comparison_summary["Dynamic Beta(c=0.1)"].append("Calculation Failed")
                else:
                    comparison_summary["Dynamic Beta(c=0.1)"].append("Insufficient Data")
            else:
                comparison_summary["Dynamic Beta(c=0.1)"].append("No Data")
    
    # 计算相对改进
    if len(comparison_summary["Fixed Beta"]) == 4 and len(comparison_summary["Dynamic Beta(c=0.1)"]) == 4:
        # 损失值改进
        try:
            fixed_loss = float(comparison_summary["Fixed Beta"][0])
            dynamic_loss = float(comparison_summary["Dynamic Beta(c=0.1)"][0])
            
            # 添加除零保护
            if fixed_loss != 0:
                loss_improvement = (fixed_loss - dynamic_loss) / fixed_loss * 100
                comparison_summary["Relative Improvement"].append(f"{loss_improvement:.2f}%")
            else:
                comparison_summary["Relative Improvement"].append("N/A (base=0)")
        except:
            comparison_summary["Relative Improvement"].append("N/A")
            
        # beta值
        comparison_summary["Relative Improvement"].append("N/A")
        
        # 收敛速度
        comparison_summary["Relative Improvement"].append("N/A")
        
        # 相关性
        comparison_summary["Relative Improvement"].append("N/A")
    
    # 保存比较表
    with open(os.path.join(base_dir, "comparison_summary.txt"), "w") as f:
        # 表头
        f.write("BetaDPO Algorithm Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # 表格
        max_lengths = {}
        for key in comparison_summary:
            max_lengths[key] = max(len(key), max([len(str(item)) for item in comparison_summary[key]]))
        
        # 写入表头
        header = "| "
        for key in comparison_summary:
            header += f"{key:{max_lengths[key]}} | "
        f.write(header + "\n")
        
        # 分隔线
        separator = "| "
        for key in comparison_summary:
            separator += "-" * max_lengths[key] + " | "
        f.write(separator + "\n")
        
        # 写入数据行
        for i in range(len(comparison_summary["Metric"])):
            row = "| "
            for key in comparison_summary:
                if i < len(comparison_summary[key]):
                    row += f"{comparison_summary[key][i]:{max_lengths[key]}} | "
                else:
                    row += " " * max_lengths[key] + " | "
            f.write(row + "\n")
        
        # 总结
        f.write("\nAnalysis Conclusion:\n")
        try:
            fixed_loss = float(comparison_summary["Fixed Beta"][0])
            dynamic_loss = float(comparison_summary["Dynamic Beta(c=0.1)"][0])
            
            if dynamic_loss < fixed_loss:
                f.write("✓ Dynamic beta strategy has lower loss than fixed beta, performs better\n")
            elif dynamic_loss > fixed_loss:
                f.write("✗ Dynamic beta strategy has higher loss than fixed beta, performs worse\n")
            else:
                f.write("= Dynamic beta strategy has the same loss as fixed beta\n")
        except:
            f.write("Insufficient data, cannot draw clear conclusions\n")
    
    print(f"✓ 比较摘要已保存到: {os.path.join(base_dir, 'comparison_summary.txt')}")
    
except Exception as e:
    print(f"\n❌ 生成比较图表时出错: {str(e)}")
    traceback.print_exc()
    
    # 即使出错也创建一个错误图像
    try:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error generating chart:\n{str(e)}", 
                ha="center", va="center", fontsize=16)
        plt.title("Chart Generation Error")
        plt.axis("off")
        plt.savefig(os.path.join(base_dir, "demo_results_error.png"))
        print(f"已生成错误信息图: {os.path.join(base_dir, 'demo_results_error.png')}")
    except:
        pass
' 2>&1 | tee -a $BASE_OUTPUT_DIR/plot_log.txt
} || {
    echo "❌ 生成比较图表时出错，请检查 $BASE_OUTPUT_DIR/plot_log.txt 获取详细信息" | tee -a $BASE_OUTPUT_DIR/comparison_log.txt
}

echo "请检查结果图: $BASE_OUTPUT_DIR/demo_results.png"
echo "比较摘要: $BASE_OUTPUT_DIR/comparison_summary.txt"
echo "如果未找到训练数据，也可能在 $BASE_OUTPUT_DIR/demo_results_error.png"
echo "详细日志保存在: $BASE_OUTPUT_DIR/comparison_log.txt" 
echo "执行完成时间: $(date)" 