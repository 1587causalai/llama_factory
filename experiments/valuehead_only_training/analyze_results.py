#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析三个实验的训练结果，对比损失和准确率变化
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 结果路径
results_paths = {
    "valuehead_only": "experiments/valuehead_only_training/outputs/trainer_state.json",
    "policy_only": "experiments/valuehead_only_training/outputs_policy_only/trainer_state.json",
    "normal": "experiments/valuehead_only_training/outputs_normal/trainer_state.json"
}

# 结果标签
labels = {
    "valuehead_only": "只训练 ValueHead",
    "policy_only": "只训练 Policy",
    "normal": "正常训练"
}

# 颜色设置
colors = {
    "valuehead_only": "blue",
    "policy_only": "red",
    "normal": "green"
}

def load_results(path):
    """加载训练结果"""
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data

def extract_metrics(data):
    """提取关键指标"""
    metrics = defaultdict(list)
    
    for log in data["log_history"]:
        step = log.get("step", None)
        if step is None:
            continue
            
        # 训练损失
        loss = log.get("loss", None)
        if loss is not None:
            metrics["train_loss"].append((step, loss))
            
        # 评估损失
        eval_loss = log.get("eval_loss", None)
        if eval_loss is not None:
            metrics["eval_loss"].append((step, eval_loss))
            
        # 精度 (accuracy)
        accuracy = log.get("eval_accuracy", None)
        if accuracy is not None:
            metrics["accuracy"].append((step, accuracy))
            
    return metrics

def plot_comparison(metrics_dict):
    """绘制对比图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 训练损失对比
    ax = axes[0]
    for exp_name, metrics in metrics_dict.items():
        if not metrics or "train_loss" not in metrics or not metrics["train_loss"]:
            continue
        steps, values = zip(*metrics["train_loss"])
        ax.plot(steps, values, label=labels[exp_name], color=colors[exp_name])
    
    ax.set_title("训练损失对比")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("损失值")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 评估损失对比
    ax = axes[1]
    for exp_name, metrics in metrics_dict.items():
        if not metrics or "eval_loss" not in metrics or not metrics["eval_loss"]:
            continue
        steps, values = zip(*metrics["eval_loss"])
        ax.plot(steps, values, label=labels[exp_name], color=colors[exp_name])
    
    ax.set_title("评估损失对比")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("损失值")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 准确率对比
    ax = axes[2]
    for exp_name, metrics in metrics_dict.items():
        if not metrics or "accuracy" not in metrics or not metrics["accuracy"]:
            continue
        steps, values = zip(*metrics["accuracy"])
        ax.plot(steps, values, label=labels[exp_name], color=colors[exp_name])
    
    ax.set_title("准确率对比")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("准确率")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("experiments/valuehead_only_training/comparison_results.png", dpi=300)
    plt.savefig("experiments/valuehead_only_training/comparison_results.pdf")
    print(f"图表已保存到: experiments/valuehead_only_training/comparison_results.png")

def main():
    # 加载所有实验结果
    all_results = {}
    for exp_name, path in results_paths.items():
        data = load_results(path)
        if data:
            all_results[exp_name] = extract_metrics(data)
        else:
            print(f"无法加载 {exp_name} 的结果")
    
    # 打印结果摘要
    for exp_name, metrics in all_results.items():
        print(f"\n===== {labels[exp_name]} =====")
        if "train_loss" in metrics and metrics["train_loss"]:
            initial_loss = metrics["train_loss"][0][1]
            final_loss = metrics["train_loss"][-1][1]
            print(f"训练损失: 初始 {initial_loss:.4f} -> 最终 {final_loss:.4f}, 变化 {final_loss-initial_loss:.4f}")
        
        if "eval_loss" in metrics and metrics["eval_loss"]:
            initial_loss = metrics["eval_loss"][0][1]
            final_loss = metrics["eval_loss"][-1][1]
            print(f"评估损失: 初始 {initial_loss:.4f} -> 最终 {final_loss:.4f}, 变化 {final_loss-initial_loss:.4f}")
        
        if "accuracy" in metrics and metrics["accuracy"]:
            initial_acc = metrics["accuracy"][0][1]
            final_acc = metrics["accuracy"][-1][1]
            print(f"准确率: 初始 {initial_acc:.4f} -> 最终 {final_acc:.4f}, 变化 {final_acc-initial_acc:.4f}")
    
    # 绘制对比图表
    plot_comparison(all_results)

if __name__ == "__main__":
    main() 