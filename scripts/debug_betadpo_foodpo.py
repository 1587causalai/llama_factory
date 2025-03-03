#!/usr/bin/env python

# 调试脚本：用于测试fooDPO数据处理和训练过程的分析

import os
import sys
import json
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import torch
import glob
from datetime import datetime

# 调试日志
def log(message):
    """打印带时间戳的调试信息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[DEBUG {timestamp}] {message}")

# 设置基础目录
base_dir = "output/qwen_foodpo_test"
log_dir = os.path.join(base_dir, "trainer_state.json")

def debug_data_loading():
    """检查数据集加载情况"""
    log("分析数据集加载情况...")
    
    try:
        # 检查数据集文件
        data_file = "data/dpo_zh_demo.json"
        if os.path.exists(data_file):
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            log(f"数据集加载成功，包含 {len(data)} 个样本")
            
            # 统计基本信息
            prompt_lengths = []
            chosen_lengths = []
            rejected_lengths = []
            
            for item in data:
                if isinstance(item, dict):
                    if "messages" in item:
                        prompt = "\n".join([m.get("content", "") for m in item.get("messages", [])])
                        prompt_lengths.append(len(prompt))
                    
                    chosen = item.get("chosen", "")
                    rejected = item.get("rejected", "")
                    
                    chosen_lengths.append(len(chosen) if chosen else 0)
                    rejected_lengths.append(len(rejected) if rejected else 0)
            
            log(f"提示平均长度: {np.mean(prompt_lengths):.1f} 字符")
            log(f"chosen回复平均长度: {np.mean(chosen_lengths):.1f} 字符")
            log(f"rejected回复平均长度: {np.mean(rejected_lengths):.1f} 字符")
            
            # 绘制长度分布
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 3, 1)
            plt.hist(prompt_lengths, bins=10, alpha=0.7)
            plt.title("提示长度分布")
            
            plt.subplot(1, 3, 2)
            plt.hist(chosen_lengths, bins=10, alpha=0.7)
            plt.title("chosen回复长度分布")
            
            plt.subplot(1, 3, 3)
            plt.hist(rejected_lengths, bins=10, alpha=0.7)
            plt.title("rejected回复长度分布")
            
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, "data_analysis.png"))
            log(f"数据分析图表已保存至 {os.path.join(base_dir, 'data_analysis.png')}")
        else:
            log(f"错误：找不到数据文件 {data_file}")
    except Exception as e:
        log(f"数据加载分析出错: {str(e)}")
        traceback.print_exc()

def debug_training_process():
    """分析训练过程"""
    log("分析训练过程...")
    
    try:
        if os.path.exists(log_dir):
            with open(log_dir, "r") as f:
                training_log = json.load(f)
            
            log(f"训练日志加载成功")
            
            # 提取训练指标
            logs = training_log.get("log_history", [])
            epochs = [entry.get("epoch", 0) for entry in logs if "epoch" in entry]
            loss = [entry.get("loss", 0) for entry in logs if "loss" in entry]
            rewards_chosen = [entry.get("rewards/chosen", 0) for entry in logs if "rewards/chosen" in entry]
            rewards_rejected = [entry.get("rewards/rejected", 0) for entry in logs if "rewards/rejected" in entry]
            margins = [entry.get("rewards/margins", 0) for entry in logs if "rewards/margins" in entry]
            accuracies = [entry.get("rewards/accuracies", 0) for entry in logs if "rewards/accuracies" in entry]
            
            # 绘制训练过程图表
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(epochs, loss, marker='o', linestyle='-')
            plt.title("训练损失")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(epochs, rewards_chosen, marker='o', label="Chosen奖励", color="green")
            plt.plot(epochs, rewards_rejected, marker='x', label="Rejected奖励", color="red")
            plt.title("奖励值变化")
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(epochs, margins, marker='o', color="purple")
            plt.title("奖励边际差值")
            plt.xlabel("Epoch")
            plt.ylabel("Margin")
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.plot(epochs, accuracies, marker='o', color="blue")
            plt.title("选择准确率")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, "training_analysis.png"))
            log(f"训练分析图表已保存至 {os.path.join(base_dir, 'training_analysis.png')}")
            
            # 打印训练统计信息
            log(f"训练步数: {len(logs)}")
            if loss:
                log(f"最终损失: {loss[-1]:.4f}")
            if margins:
                log(f"最终奖励边际: {margins[-1]:.4f}")
            if accuracies:
                log(f"最终选择准确率: {accuracies[-1]:.4f}")
        else:
            log(f"错误：找不到训练日志文件 {log_dir}")
    except Exception as e:
        log(f"训练过程分析出错: {str(e)}")
        traceback.print_exc()

def debug_model_outputs():
    """分析模型输出"""
    log("分析模型保存的检查点和权重...")
    
    try:
        # 检查检查点文件
        checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
        if checkpoints:
            log(f"找到 {len(checkpoints)} 个检查点")
            for cp in checkpoints:
                log(f"检查点: {os.path.basename(cp)}")
                
                # 检查适配器配置
                adapter_config = os.path.join(cp, "adapter_config.json")
                if os.path.exists(adapter_config):
                    with open(adapter_config, "r") as f:
                        config = json.load(f)
                    log(f"适配器配置:")
                    pprint(config)
                
                # 检查模型文件大小
                adapter_model = os.path.join(cp, "adapter_model.bin")
                if os.path.exists(adapter_model):
                    size_mb = os.path.getsize(adapter_model) / (1024 * 1024)
                    log(f"适配器模型大小: {size_mb:.2f} MB")
                
                # 如果有safetensors文件
                safetensors = os.path.join(cp, "adapter_model.safetensors")
                if os.path.exists(safetensors):
                    size_mb = os.path.getsize(safetensors) / (1024 * 1024)
                    log(f"适配器safetensors大小: {size_mb:.2f} MB")
        else:
            log("未找到任何检查点")
            
        # 分析最终模型
        final_adapter = os.path.join(base_dir, "adapter_model.bin")
        if os.path.exists(final_adapter):
            size_mb = os.path.getsize(final_adapter) / (1024 * 1024)
            log(f"最终适配器模型大小: {size_mb:.2f} MB")
            
            # 尝试加载模型并分析参数
            try:
                state_dict = torch.load(final_adapter, map_location="cpu")
                param_count = sum(p.numel() for p in state_dict.values())
                log(f"模型参数总数: {param_count:,}")
                
                # 分析参数分布
                param_norms = [torch.norm(p).item() for p in state_dict.values()]
                plt.figure(figsize=(8, 6))
                plt.hist(param_norms, bins=30)
                plt.title("参数范数分布")
                plt.xlabel("参数范数")
                plt.ylabel("频率")
                plt.savefig(os.path.join(base_dir, "param_analysis.png"))
                log(f"参数分析图表已保存至 {os.path.join(base_dir, 'param_analysis.png')}")
            except Exception as e:
                log(f"加载模型权重失败: {str(e)}")
    except Exception as e:
        log(f"模型输出分析出错: {str(e)}")
        traceback.print_exc()

def main():
    """主函数"""
    log("开始fooDPO训练分析...")
    
    # 确保输出目录存在
    if not os.path.exists(base_dir):
        log(f"错误: 输出目录 {base_dir} 不存在，请先运行训练脚本!")
        return
    
    # 运行各个调试函数
    debug_data_loading()
    debug_training_process()
    debug_model_outputs()
    
    log("fooDPO训练分析完成!")

if __name__ == "__main__":
    main() 