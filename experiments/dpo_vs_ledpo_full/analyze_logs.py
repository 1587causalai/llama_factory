#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析标准DPO与LEDPO的完整实验日志
比较训练效果、奖励变化和beta值分布特性
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings
from scipy import stats
import re

# 图表设置 - 避免中文字体问题
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 颜色方案
COLOR_DPO = "#1f77b4"  # 蓝色
COLOR_LEDPO = "#ff7f0e"  # 橙色

def find_field_by_pattern(entry: Dict, patterns: List[str]) -> Optional[str]:
    """根据正则模式在日志条目中查找匹配的字段"""
    for key in entry.keys():
        for pattern in patterns:
            if re.search(pattern, key):
                return key
    return None

def load_trainer_state(log_dir: str) -> Optional[Dict]:
    """加载trainer_state.json文件"""
    trainer_state_path = os.path.join(log_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        print(f"Warning: Cannot find trainer_state.json file: {trainer_state_path}")
        return None
    
    with open(trainer_state_path, "r") as f:
        return json.load(f)

def extract_metrics(trainer_state: Dict) -> Optional[pd.DataFrame]:
    """从trainer_state中提取指标数据，并转换为标准化的DataFrame"""
    if trainer_state is None:
        return None
        
    log_history = trainer_state.get("log_history", [])
    if not log_history:
        print("Warning: Log history is empty")
        return None
    
    # 字段模式映射到标准字段名
    field_patterns = {
        r'loss$': 'loss',
        r'eval_loss$': 'eval_loss',
        r'learning_rate$': 'learning_rate',
        r'(reward|rewards)/(margin|margins)$': 'reward_margin',
        r'(reward|rewards)/chosen$': 'reward_chosen',
        r'(reward|rewards)/rejected$': 'reward_rejected',
        r'(reward|rewards)/accuracies$': 'accuracy',
        r'eval_accuracy$': 'eval_accuracy',
        r'beta$': 'beta',
        r'pos_beta$': 'pos_beta',
        r'neg_beta$': 'neg_beta',
    }
    
    # 扫描第一个有loss的日志条目来确定字段映射
    field_mapping = {}
    for entry in log_history:
        if 'loss' in entry:
            for pattern, standard_name in field_patterns.items():
                field = find_field_by_pattern(entry, [pattern])
                if field:
                    field_mapping[standard_name] = field
            break
    
    print(f"Detected field mapping: {field_mapping}")
    
    # 提取训练和评估指标
    train_data = []
    eval_data = []
    
    for entry in log_history:
        # 基本信息
        record = {"step": entry.get("step", 0)}
        
        # 确定是训练还是评估记录
        is_eval = any(k.startswith("eval_") for k in entry.keys())
        
        # 提取标准化的指标
        for standard_name, original_field in field_mapping.items():
            if original_field in entry:
                record[standard_name] = entry[original_field]
        
        # 添加到对应的列表
        if is_eval:
            eval_data.append(record)
        elif "loss" in record:  # 确保是训练记录
            train_data.append(record)
    
    train_df = pd.DataFrame(train_data) if train_data else None
    eval_df = pd.DataFrame(eval_data) if eval_data else None
    
    return {
        "train": train_df,
        "eval": eval_df
    }

def plot_training_curves(dpo_metrics: Dict[str, pd.DataFrame], 
                        ledpo_metrics: Dict[str, pd.DataFrame], 
                        output_dir: str):
    """绘制训练曲线对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练指标列表
    metrics_to_plot = [
        ("loss", "Training Loss", "Loss"),
        ("reward_margin", "Reward Margin", "Margin"),
        ("reward_chosen", "Chosen Reward", "Reward"),
        ("reward_rejected", "Rejected Reward", "Reward"),
        ("accuracy", "Training Accuracy", "Accuracy")
    ]
    
    for metric_name, title, ylabel in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制DPO曲线
        if dpo_metrics and "train" in dpo_metrics and dpo_metrics["train"] is not None:
            dpo_df = dpo_metrics["train"]
            if metric_name in dpo_df.columns:
                ax.plot(dpo_df["step"], dpo_df[metric_name], color=COLOR_DPO, 
                        label="Standard DPO", linewidth=2)
        
        # 绘制LEDPO曲线
        if ledpo_metrics and "train" in ledpo_metrics and ledpo_metrics["train"] is not None:
            ledpo_df = ledpo_metrics["train"]
            if metric_name in ledpo_df.columns:
                ax.plot(ledpo_df["step"], ledpo_df[metric_name], color=COLOR_LEDPO, 
                        label="LEDPO (Dynamic Beta)", linewidth=2)
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name}_comparison.png"))
        plt.close()

def plot_evaluation_curves(dpo_metrics: Dict[str, pd.DataFrame], 
                          ledpo_metrics: Dict[str, pd.DataFrame], 
                          output_dir: str):
    """绘制评估曲线对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估指标列表
    metrics_to_plot = [
        ("eval_loss", "Evaluation Loss", "Loss"),
        ("eval_accuracy", "Evaluation Accuracy", "Accuracy")
    ]
    
    for metric_name, title, ylabel in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制DPO曲线
        if dpo_metrics and "eval" in dpo_metrics and dpo_metrics["eval"] is not None:
            dpo_df = dpo_metrics["eval"]
            if metric_name in dpo_df.columns:
                ax.plot(dpo_df["step"], dpo_df[metric_name], color=COLOR_DPO, 
                        label="Standard DPO", linewidth=2, marker='o')
        
        # 绘制LEDPO曲线
        if ledpo_metrics and "eval" in ledpo_metrics and ledpo_metrics["eval"] is not None:
            ledpo_df = ledpo_metrics["eval"]
            if metric_name in ledpo_df.columns:
                ax.plot(ledpo_df["step"], ledpo_df[metric_name], color=COLOR_LEDPO, 
                        label="LEDPO (Dynamic Beta)", linewidth=2, marker='o')
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name}_comparison.png"))
        plt.close()

def analyze_beta_values(ledpo_metrics: Dict[str, pd.DataFrame], output_dir: str):
    """分析LEDPO中beta值的变化特性"""
    if not ledpo_metrics or "train" not in ledpo_metrics or ledpo_metrics["train"] is None:
        print("Warning: No LEDPO training data available for beta analysis")
        return
    
    train_df = ledpo_metrics["train"]
    beta_fields = []
    if "beta" in train_df.columns:
        beta_fields.append(("beta", "Overall Beta"))
    if "pos_beta" in train_df.columns:
        beta_fields.append(("pos_beta", "Positive Beta"))
    if "neg_beta" in train_df.columns:
        beta_fields.append(("neg_beta", "Negative Beta"))
    
    if not beta_fields:
        print("Warning: No beta fields found in LEDPO logs")
        return
    
    # 绘制beta变化趋势图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for field, label in beta_fields:
        ax.plot(train_df["step"], train_df[field], label=label, linewidth=2)
    
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Beta Value")
    ax.set_title("Beta Values During LEDPO Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加初始beta值参考线
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label="Initial Beta (0.1)")
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beta_values_trend.png"))
    plt.close()
    
    # 绘制beta值分布图 (使用训练后期的数据)
    if len(train_df) > 100:  # 确保有足够的数据点
        last_third = train_df.iloc[len(train_df)*2//3:]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for field, label in beta_fields:
            sns.histplot(last_third[field], kde=True, label=label, ax=ax, alpha=0.6)
        
        ax.set_xlabel("Beta Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Beta Values (Last Third of Training)")
        ax.legend()
        
        # 添加初始beta值参考线
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.7, label="Initial Beta (0.1)")
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "beta_values_distribution.png"))
        plt.close()

def create_performance_summary(dpo_metrics: Dict[str, pd.DataFrame], 
                              ledpo_metrics: Dict[str, pd.DataFrame], 
                              output_dir: str):
    """创建性能摘要报告"""
    summary = []
    
    # 添加标题
    summary.append("# DPO vs LEDPO Performance Summary")
    summary.append("\n## Final Training Metrics\n")
    
    # 提取最后的训练指标
    training_metrics = ["loss", "reward_margin", "accuracy"]
    summary.append("| Metric | Standard DPO | LEDPO | Difference |")
    summary.append("|--------|-------------|-------|------------|")
    
    for metric in training_metrics:
        dpo_value = "N/A"
        ledpo_value = "N/A"
        diff = "N/A"
        
        if (dpo_metrics and "train" in dpo_metrics and dpo_metrics["train"] is not None and 
            metric in dpo_metrics["train"].columns):
            dpo_df = dpo_metrics["train"]
            dpo_value = f"{dpo_df[metric].iloc[-1]:.4f}"
        
        if (ledpo_metrics and "train" in ledpo_metrics and ledpo_metrics["train"] is not None and
            metric in ledpo_metrics["train"].columns):
            ledpo_df = ledpo_metrics["train"]
            ledpo_value = f"{ledpo_df[metric].iloc[-1]:.4f}"
        
        if dpo_value != "N/A" and ledpo_value != "N/A":
            try:
                dpo_val = float(dpo_value)
                ledpo_val = float(ledpo_value)
                diff = f"{ledpo_val - dpo_val:.4f}"
                if ledpo_val < dpo_val and metric == "loss":
                    diff += " 🔽"  # 损失降低是好的
                elif ledpo_val > dpo_val and metric != "loss":
                    diff += " 🔼"  # 其他指标升高是好的
            except ValueError:
                diff = "N/A"
        
        summary.append(f"| {metric} | {dpo_value} | {ledpo_value} | {diff} |")
    
    # 添加评估指标摘要
    summary.append("\n## Final Evaluation Metrics\n")
    
    eval_metrics = ["eval_loss", "eval_accuracy"]
    summary.append("| Metric | Standard DPO | LEDPO | Difference |")
    summary.append("|--------|-------------|-------|------------|")
    
    for metric in eval_metrics:
        dpo_value = "N/A"
        ledpo_value = "N/A"
        diff = "N/A"
        
        if (dpo_metrics and "eval" in dpo_metrics and dpo_metrics["eval"] is not None and 
            metric in dpo_metrics["eval"].columns):
            dpo_df = dpo_metrics["eval"]
            dpo_value = f"{dpo_df[metric].iloc[-1]:.4f}"
        
        if (ledpo_metrics and "eval" in ledpo_metrics and ledpo_metrics["eval"] is not None and
            metric in ledpo_metrics["eval"].columns):
            ledpo_df = ledpo_metrics["eval"]
            ledpo_value = f"{ledpo_df[metric].iloc[-1]:.4f}"
        
        if dpo_value != "N/A" and ledpo_value != "N/A":
            try:
                dpo_val = float(dpo_value)
                ledpo_val = float(ledpo_value)
                diff = f"{ledpo_val - dpo_val:.4f}"
                if ledpo_val < dpo_val and metric == "eval_loss":
                    diff += " 🔽"  # 损失降低是好的
                elif ledpo_val > dpo_val and metric != "eval_loss":
                    diff += " 🔼"  # 其他指标升高是好的
            except ValueError:
                diff = "N/A"
        
        summary.append(f"| {metric} | {dpo_value} | {ledpo_value} | {diff} |")
    
    # 添加Beta值分析 (仅LEDPO)
    if (ledpo_metrics and "train" in ledpo_metrics and ledpo_metrics["train"] is not None):
        ledpo_df = ledpo_metrics["train"]
        beta_fields = []
        
        if "beta" in ledpo_df.columns:
            beta_fields.append(("beta", "Overall Beta"))
        if "pos_beta" in ledpo_df.columns:
            beta_fields.append(("pos_beta", "Positive Beta"))
        if "neg_beta" in ledpo_df.columns:
            beta_fields.append(("neg_beta", "Negative Beta"))
        
        if beta_fields:
            summary.append("\n## Beta Value Analysis (LEDPO)\n")
            summary.append("| Beta Type | Initial | Final | Mean | Std Dev | Min | Max |")
            summary.append("|-----------|---------|-------|------|---------|-----|-----|")
            
            for field, label in beta_fields:
                initial = 0.1  # 初始beta值
                final = ledpo_df[field].iloc[-1]
                mean = ledpo_df[field].mean()
                std = ledpo_df[field].std()
                min_val = ledpo_df[field].min()
                max_val = ledpo_df[field].max()
                
                summary.append(f"| {label} | {initial:.4f} | {final:.4f} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |")
    
    # 保存摘要报告
    with open(os.path.join(output_dir, "performance_summary.md"), "w") as f:
        f.write("\n".join(summary))

def perform_statistical_tests(dpo_metrics: Dict[str, pd.DataFrame], 
                             ledpo_metrics: Dict[str, pd.DataFrame], 
                             output_dir: str):
    """执行统计显著性检验"""
    results = []
    
    # 添加标题
    results.append("# Statistical Significance Tests")
    results.append("\nThis report shows the results of statistical tests comparing DPO and LEDPO metrics.\n")
    
    # 训练指标统计检验
    if (dpo_metrics and "train" in dpo_metrics and dpo_metrics["train"] is not None and
        ledpo_metrics and "train" in ledpo_metrics and ledpo_metrics["train"] is not None):
        
        dpo_train = dpo_metrics["train"]
        ledpo_train = ledpo_metrics["train"]
        
        # 只考虑两者都有的指标
        common_metrics = set(dpo_train.columns) & set(ledpo_train.columns)
        common_metrics = common_metrics - {"step", "learning_rate"}  # 排除非性能指标
        
        if common_metrics:
            results.append("\n## Training Metrics\n")
            results.append("| Metric | p-value | Significant at α=0.05 | Better Model |")
            results.append("|--------|---------|------------------------|-------------|")
            
            for metric in common_metrics:
                # 使用训练的后半部分数据进行比较
                dpo_values = dpo_train[metric].iloc[len(dpo_train)//2:].dropna()
                ledpo_values = ledpo_train[metric].iloc[len(ledpo_train)//2:].dropna()
                
                if len(dpo_values) > 0 and len(ledpo_values) > 0:
                    try:
                        # 曼-惠特尼U检验 (非参数检验，适用于非正态分布)
                        stat, p_value = stats.mannwhitneyu(dpo_values, ledpo_values)
                        significant = "Yes" if p_value < 0.05 else "No"
                        
                        # 确定哪个模型更好
                        if metric == "loss":
                            # 对于损失，更低更好
                            better = "LEDPO" if ledpo_values.mean() < dpo_values.mean() else "DPO"
                        else:
                            # 对于其他指标（如准确率、奖励），更高更好
                            better = "LEDPO" if ledpo_values.mean() > dpo_values.mean() else "DPO"
                        
                        results.append(f"| {metric} | {p_value:.6f} | {significant} | {better} |")
                    except Exception as e:
                        results.append(f"| {metric} | Error: {str(e)} | - | - |")
    
    # 评估指标统计检验
    if (dpo_metrics and "eval" in dpo_metrics and dpo_metrics["eval"] is not None and
        ledpo_metrics and "eval" in ledpo_metrics and ledpo_metrics["eval"] is not None):
        
        dpo_eval = dpo_metrics["eval"]
        ledpo_eval = ledpo_metrics["eval"]
        
        # 只考虑两者都有的指标
        common_metrics = set(dpo_eval.columns) & set(ledpo_eval.columns)
        common_metrics = common_metrics - {"step"}  # 排除非性能指标
        
        if common_metrics:
            results.append("\n## Evaluation Metrics\n")
            results.append("| Metric | p-value | Significant at α=0.05 | Better Model |")
            results.append("|--------|---------|------------------------|-------------|")
            
            for metric in common_metrics:
                # 使用所有评估数据
                dpo_values = dpo_eval[metric].dropna()
                ledpo_values = ledpo_eval[metric].dropna()
                
                if len(dpo_values) > 0 and len(ledpo_values) > 0:
                    try:
                        # 曼-惠特尼U检验
                        stat, p_value = stats.mannwhitneyu(dpo_values, ledpo_values)
                        significant = "Yes" if p_value < 0.05 else "No"
                        
                        # 确定哪个模型更好
                        if "loss" in metric:
                            # 对于损失，更低更好
                            better = "LEDPO" if ledpo_values.mean() < dpo_values.mean() else "DPO"
                        else:
                            # 对于其他指标，更高更好
                            better = "LEDPO" if ledpo_values.mean() > dpo_values.mean() else "DPO"
                        
                        results.append(f"| {metric} | {p_value:.6f} | {significant} | {better} |")
                    except Exception as e:
                        results.append(f"| {metric} | Error: {str(e)} | - | - |")
    
    # 保存统计检验结果
    with open(os.path.join(output_dir, "statistical_tests.md"), "w") as f:
        f.write("\n".join(results))

def main():
    """主函数: 分析实验日志并生成报告"""
    parser = argparse.ArgumentParser(description="Analyze DPO vs LEDPO experiment logs")
    parser.add_argument("--dpo_dir", type=str, required=True, help="Standard DPO results directory")
    parser.add_argument("--ledpo_dir", type=str, required=True, help="LEDPO results directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Starting Experiment Log Analysis ===")
    
    # 加载日志数据
    print(f"Loading DPO logs from: {args.dpo_dir}")
    dpo_trainer_state = load_trainer_state(args.dpo_dir)
    
    print(f"Loading LEDPO logs from: {args.ledpo_dir}")
    ledpo_trainer_state = load_trainer_state(args.ledpo_dir)
    
    if dpo_trainer_state is None and ledpo_trainer_state is None:
        print("Error: Could not load any log data")
        return
    
    # 提取指标
    dpo_metrics = extract_metrics(dpo_trainer_state)
    ledpo_metrics = extract_metrics(ledpo_trainer_state)
    
    # 绘制训练曲线对比图
    print("Generating training curve comparisons...")
    plot_training_curves(dpo_metrics, ledpo_metrics, args.output_dir)
    
    # 绘制评估曲线对比图
    print("Generating evaluation curve comparisons...")
    plot_evaluation_curves(dpo_metrics, ledpo_metrics, args.output_dir)
    
    # 分析LEDPO中的beta值
    print("Analyzing LEDPO beta values...")
    analyze_beta_values(ledpo_metrics, args.output_dir)
    
    # 创建性能摘要报告
    print("Creating performance summary report...")
    create_performance_summary(dpo_metrics, ledpo_metrics, args.output_dir)
    
    # 执行统计显著性检验
    print("Performing statistical significance tests...")
    perform_statistical_tests(dpo_metrics, ledpo_metrics, args.output_dir)
    
    print(f"Analysis complete. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 