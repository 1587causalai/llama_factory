#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制LEDPO训练指标图表，按照指定优先级顺序展示
将train和eval指标绘制在同一张图上以便比较

用法:
    python plot_ledpo_metrics.py --result_dir results/qwen15-0.5b/lora/foodpo
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置字体和样式
plt.style.use('seaborn-v0_8-darkgrid')

# 指定关心的指标及其显示顺序
PRIORITY_METRICS = [
    "rewards/accuracies",  # accuracy
    "loss",                # loss
    "rewards/margins",     # reward/margin
    "rewards/chosen",      # reward/chosen
    "rewards/rejected",    # reward/rejected
    "beta",                # beta值
    "pos_beta",            # 正样本beta值
    "neg_beta"             # 负样本beta值
]

# 指标的英文名称，用于图表标题
METRIC_NAMES = {
    "rewards/accuracies": "Accuracy",
    "loss": "Loss",
    "rewards/margins": "Reward Margin",
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
    """从trainer_state中提取指标数据"""
    train_data = []
    eval_data = []
    
    # 遍历日志历史
    for entry in state_data["log_history"]:
        # 检查是训练还是评估指标
        if "eval_" in str(entry.keys()):
            # 这是评估指标
            if "step" in entry:
                # 获取动态beta值，如果不存在则使用默认值0.1
                beta_value = entry.get("eval_beta/mean", 0.1)
                
                eval_entry = {
                    "step": entry["step"],
                    "eval_rewards/accuracies": entry.get("eval_rewards/accuracies", None),
                    "eval_loss": entry.get("eval_dpo_zh_demo_loss", None),  # 注意特殊的损失名称
                    "eval_rewards/margins": entry.get("eval_rewards/margins", None),
                    "eval_rewards/chosen": entry.get("eval_rewards/chosen", None),
                    "eval_rewards/rejected": entry.get("eval_rewards/rejected", None),
                    "eval_beta": beta_value,  # 使用日志中的动态beta值
                    "eval_pos_beta": entry.get("eval_pos_beta", None),  # 新增：正样本beta
                    "eval_neg_beta": entry.get("eval_neg_beta", None)   # 新增：负样本beta
                }
                eval_data.append(eval_entry)
        elif "step" in entry and "train_loss" not in entry:
            # 这是训练指标
            # 获取动态beta值，如果不存在则使用默认值0.1
            beta_value = entry.get("beta/mean", 0.1)
            
            train_entry = {
                "step": entry["step"],
                "rewards/accuracies": entry.get("rewards/accuracies", None),
                "loss": entry.get("loss", None),
                "rewards/margins": entry.get("rewards/margins", None),
                "rewards/chosen": entry.get("rewards/chosen", None),
                "rewards/rejected": entry.get("rewards/rejected", None),
                "beta": beta_value,  # 使用日志中的动态beta值
                "pos_beta": entry.get("pos_beta", None),  # 新增：正样本beta
                "neg_beta": entry.get("neg_beta", None)   # 新增：负样本beta
            }
            train_data.append(train_entry)
    
    # 创建DataFrame
    train_df = pd.DataFrame(train_data)
    eval_df = pd.DataFrame(eval_data)
    
    return {"train": train_df, "eval": eval_df}

def plot_combined_metrics(metrics_dict, output_dir):
    """将train和eval数据绘制在同一张图上"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据
    train_df = metrics_dict.get("train", pd.DataFrame())
    eval_df = metrics_dict.get("eval", pd.DataFrame())
    
    if train_df.empty and eval_df.empty:
        print("警告: 没有找到任何数据，无法生成图表")
        return
    
    # 创建一个包含所有指标的大图
    fig, axes = plt.subplots(len(PRIORITY_METRICS), 1, figsize=(12, 3*len(PRIORITY_METRICS)), 
                            sharex=True, constrained_layout=True)
    
    # 如果只有一个指标，axes不会是数组
    if len(PRIORITY_METRICS) == 1:
        axes = [axes]
    
    # 设置颜色和样式
    train_color = 'blue'
    eval_color = 'red'
    train_style = '-'
    eval_style = '--'
    
    # 按顺序绘制每个指标
    for i, metric in enumerate(PRIORITY_METRICS):
        ax = axes[i]
        has_data = False
        
        # 绘制训练数据
        if not train_df.empty and metric in train_df.columns and not train_df[metric].isnull().all():
            train_x = train_df['step']
            train_y = train_df[metric]
            ax.plot(train_x, train_y, marker='o', linestyle=train_style, color=train_color, 
                    markersize=5, linewidth=2, alpha=0.8, label='Train')
            has_data = True
        
        # 绘制评估数据
        eval_metric = f"eval_{metric}"
        if not eval_df.empty and eval_metric in eval_df.columns and not eval_df[eval_metric].isnull().all():
            eval_x = eval_df['step']
            eval_y = eval_df[eval_metric]
            ax.plot(eval_x, eval_y, marker='s', linestyle=eval_style, color=eval_color, 
                    markersize=5, linewidth=2, alpha=0.8, label='Eval')
            has_data = True
        
        if has_data:
            # 添加标题和标签
            ax.set_title(f"{METRIC_NAMES.get(metric, metric)}", fontsize=14)
            ax.set_ylabel(f"{METRIC_NAMES.get(metric, metric)}")
            
            # 添加图例（只在首次有数据时添加一次）
            if i == 0 or not axes[i-1].get_legend():
                ax.legend(loc='best')
            
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
    fig.suptitle('LEDPO Training Metrics (Train vs Eval)', fontsize=16)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'ledpo_combined_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"已生成合并图表: {os.path.join(output_dir, 'ledpo_combined_metrics.png')}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制LEDPO训练指标图表')
    parser.add_argument('--result_dir', type=str, required=True, 
                        help='训练结果目录，包含trainer_state.json文件')
    args = parser.parse_args()
    
    # 构建trainer_state.json路径
    state_path = os.path.join(args.result_dir, 'trainer_state.json')
    if not os.path.exists(state_path):
        print(f"错误: 无法找到trainer_state.json文件: {state_path}")
        return
    
    # 读取trainer_state数据
    state_data = read_trainer_state(state_path)
    
    # 提取指标数据
    metrics_dict = extract_metrics_from_state(state_data)
    
    # 输出目录
    output_dir = os.path.join(args.result_dir, 'ledpo_plots')
    
    # 打印可用指标信息
    print("找到的指标数据:")
    for data_type, df in metrics_dict.items():
        print(f"  {data_type}类型的数据列:")
        for col in df.columns:
            print(f"    - {col}")
    
    # 绘制合并图表
    plot_combined_metrics(metrics_dict, output_dir)
    print(f"图表已生成到目录: {output_dir}")

if __name__ == "__main__":
    main() 