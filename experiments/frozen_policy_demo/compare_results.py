#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
冻结与非冻结策略模型实验结果对比脚本
比较两个实验的训练指标，生成对比图表
"""

import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置字体和样式
plt.style.use('seaborn-v0_8-darkgrid')
# 使用标准字体配置，适合学术论文
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

# 当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 实验结果路径
FROZEN_DIR = os.path.join(SCRIPT_DIR, 'results/frozen')
UNFROZEN_DIR = os.path.join(SCRIPT_DIR, 'results/unfrozen')
COMPARISON_DIR = os.path.join(SCRIPT_DIR, 'comparison')

# 关注的指标及其显示顺序
PRIORITY_METRICS = [
    "delta",                # delta值 (关键指标)
    "beta",                 # beta值 (关键指标)
    "pos_beta",             # 正样本beta值
    "neg_beta",             # 负样本beta值
    "rewards/accuracies",   # accuracy
    "loss",                 # loss
    "rewards/margins",      # reward/margin
    "rewards/chosen",       # reward/chosen
    "rewards/rejected",     # reward/rejected
]

# 指标的英文名称，用于图表标题
METRIC_NAMES = {
    "rewards/accuracies": "Accuracy",
    "loss": "Loss",
    "rewards/margins": "Reward Margin",
    "delta": "Delta = (π_θ(y_w|x) - π_ref(y_w|x)) - (π_θ(y_l|x) - π_ref(y_l|x))",
    "rewards/chosen": "Reward (Chosen)",
    "rewards/rejected": "Reward (Rejected)",
    "beta": "Beta Parameter",
    "pos_beta": "Positive Beta",
    "neg_beta": "Negative Beta"
}

def read_trainer_state(state_path):
    """读取trainer_state.json文件"""
    with open(state_path, 'r', encoding='utf-8') as f:
        state_data = json.load(f)
    return state_data

def extract_metrics_from_state(state_data):
    """从trainer_state中提取训练指标数据"""
    train_data = []
    
    # 遍历日志历史
    for entry in state_data["log_history"]:
        # 只关注训练指标 (忽略eval数据)
        if "step" in entry and "train_loss" not in entry and "eval_" not in str(entry.keys()):
            # 获取动态beta值，如果不存在则使用默认值0.1
            beta_value = entry.get("beta/mean", 0.1)
            
            # 直接从日志中获取delta值
            delta_value = entry.get("delta", None)
            
            train_entry = {
                "step": entry["step"],
                "rewards/accuracies": entry.get("rewards/accuracies", None),
                "loss": entry.get("loss", None),
                "rewards/margins": entry.get("rewards/margins", None),
                "delta": delta_value,  # 直接使用记录的delta值
                "rewards/chosen": entry.get("rewards/chosen", None),
                "rewards/rejected": entry.get("rewards/rejected", None),
                "beta": beta_value,  # 使用日志中的动态beta值
                "pos_beta": entry.get("pos_beta", None),  # 正样本beta
                "neg_beta": entry.get("neg_beta", None)   # 负样本beta
            }
            train_data.append(train_entry)
    
    # 创建DataFrame
    train_df = pd.DataFrame(train_data)
    return train_df

def plot_comparison(frozen_df, unfrozen_df, output_dir):
    """对比冻结与非冻结模型的训练指标"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if frozen_df.empty and unfrozen_df.empty:
        print("警告: 没有找到任何数据，无法生成图表")
        return
    
    # 创建一个包含所有指标的大图
    fig, axes = plt.subplots(len(PRIORITY_METRICS), 1, figsize=(12, 4*len(PRIORITY_METRICS)), 
                            sharex=True, constrained_layout=True)
    
    # 如果只有一个指标，axes不会是数组
    if len(PRIORITY_METRICS) == 1:
        axes = [axes]
    
    # 设置颜色和样式
    frozen_color = 'blue'
    unfrozen_color = 'red'
    frozen_style = '-'
    unfrozen_style = '-'
    
    # 按顺序绘制每个指标
    for i, metric in enumerate(PRIORITY_METRICS):
        ax = axes[i]
        has_data = False
        
        # 绘制冻结模型数据
        if not frozen_df.empty and metric in frozen_df.columns and not frozen_df[metric].isnull().all():
            frozen_x = frozen_df['step']
            frozen_y = frozen_df[metric]
            ax.plot(frozen_x, frozen_y, marker='o', linestyle=frozen_style, color=frozen_color, 
                    markersize=5, linewidth=2, alpha=0.8, label='Frozen Policy Model')
            has_data = True
        
        # 绘制非冻结模型数据
        if not unfrozen_df.empty and metric in unfrozen_df.columns and not unfrozen_df[metric].isnull().all():
            unfrozen_x = unfrozen_df['step']
            unfrozen_y = unfrozen_df[metric]
            ax.plot(unfrozen_x, unfrozen_y, marker='s', linestyle=unfrozen_style, color=unfrozen_color, 
                    markersize=5, linewidth=2, alpha=0.8, label='Unfrozen Policy Model')
            has_data = True
        
        if has_data:
            # 添加标题和标签
            ax.set_title(f"{METRIC_NAMES.get(metric, metric)}", fontsize=14)
            ax.set_ylabel(f"{METRIC_NAMES.get(metric, metric)}")
            
            # 添加图例
            ax.legend(loc='best', fontsize=12)
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 美化图表
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            # 如果没有数据，隐藏该子图
            ax.set_visible(False)
            print(f"警告: 未找到指标 '{metric}' 的任何数据")
    
    # 为最后一个可见的子图添加x轴标签
    for ax in reversed(axes):
        if ax.get_visible():
            ax.set_xlabel('Training Steps', fontsize=12)
            break
    
    # 添加总标题
    fig.suptitle('Comparison of Frozen and Unfrozen Policy Models', fontsize=16)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'frozen_vs_unfrozen_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"已生成对比图表: {os.path.join(output_dir, 'frozen_vs_unfrozen_comparison.png')}")
    
    # 保存摘要
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    
    # 只绘制delta对比
    if not frozen_df.empty and 'delta' in frozen_df.columns:
        plt.plot(frozen_df['step'], frozen_df['delta'], marker='o', linestyle=frozen_style, color=frozen_color, 
                markersize=5, linewidth=2, alpha=0.8, label='Frozen Policy Model')
    if not unfrozen_df.empty and 'delta' in unfrozen_df.columns:
        plt.plot(unfrozen_df['step'], unfrozen_df['delta'], marker='s', linestyle=unfrozen_style, color=unfrozen_color, 
                markersize=5, linewidth=2, alpha=0.8, label='Unfrozen Policy Model')
    
    plt.title('Delta Value Comparison', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Delta Value', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    
    # 只绘制beta对比
    if not frozen_df.empty and 'beta' in frozen_df.columns:
        plt.plot(frozen_df['step'], frozen_df['beta'], marker='o', linestyle=frozen_style, color=frozen_color, 
                markersize=5, linewidth=2, alpha=0.8, label='Frozen Policy Model')
    if not unfrozen_df.empty and 'beta' in unfrozen_df.columns:
        plt.plot(unfrozen_df['step'], unfrozen_df['beta'], marker='s', linestyle=unfrozen_style, color=unfrozen_color, 
                markersize=5, linewidth=2, alpha=0.8, label='Unfrozen Policy Model')
    
    plt.title('Beta Value Comparison', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Beta Value', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_beta_summary.png'), dpi=150, bbox_inches='tight')
    print(f"已生成Delta和Beta摘要图表: {os.path.join(output_dir, 'delta_beta_summary.png')}")

def main():
    """主函数"""
    # 创建比较目录
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    
    # 读取冻结模型训练状态
    frozen_state_path = os.path.join(FROZEN_DIR, 'trainer_state.json')
    if not os.path.exists(frozen_state_path):
        print(f"错误: 无法找到冻结模型的trainer_state.json文件: {frozen_state_path}")
        print("请先运行冻结模型实验")
        return
    
    # 读取非冻结模型训练状态
    unfrozen_state_path = os.path.join(UNFROZEN_DIR, 'trainer_state.json')
    if not os.path.exists(unfrozen_state_path):
        print(f"错误: 无法找到非冻结模型的trainer_state.json文件: {unfrozen_state_path}")
        print("请先运行非冻结模型实验")
        return
    
    # 提取指标数据
    frozen_state_data = read_trainer_state(frozen_state_path)
    unfrozen_state_data = read_trainer_state(unfrozen_state_path)
    
    frozen_df = extract_metrics_from_state(frozen_state_data)
    unfrozen_df = extract_metrics_from_state(unfrozen_state_data)
    
    # 绘制对比图表
    plot_comparison(frozen_df, unfrozen_df, COMPARISON_DIR)
    
    print(f"对比分析完成，结果保存在: {COMPARISON_DIR}")
    print("主要关注点:")
    print("1. 冻结模型的delta值是否稳定")
    print("2. 动态beta值的学习趋势")
    print("3. 正样本和负样本beta值的差异")

if __name__ == "__main__":
    main() 