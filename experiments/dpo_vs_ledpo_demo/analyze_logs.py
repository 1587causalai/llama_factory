#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析标准DPO与LEDPO的实验日志
比较训练效果、奖励变化和beta值特性
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Set
from pathlib import Path
import warnings
import re

# 图表字体设置
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100

def load_trainer_state(log_dir: str) -> Dict:
    """加载trainer_state.json文件"""
    trainer_state_path = os.path.join(log_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        print(f"警告: 找不到trainer_state.json文件: {trainer_state_path}")
        return None
    
    with open(trainer_state_path, "r") as f:
        return json.load(f)

def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """安全获取字典中的值，避免KeyError"""
    try:
        value = dictionary.get(key, default)
        # 检查是否是nan
        if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
            return default
        return value
    except:
        return default

def find_field_by_pattern(dictionary: Dict, patterns: List[str]) -> Optional[str]:
    """根据模式查找字典中的字段"""
    for pattern in patterns:
        # 创建正则表达式
        regex = re.compile(pattern, re.IGNORECASE)
        # 查找匹配的键
        matches = [key for key in dictionary.keys() if regex.search(key)]
        if matches:
            return matches[0]  # 返回第一个匹配
    return None

def extract_metrics(trainer_state: Dict) -> pd.DataFrame:
    """从trainer_state中提取指标数据，并转换为DataFrame"""
    if trainer_state is None:
        return None
        
    log_history = trainer_state.get("log_history", [])
    if not log_history:
        print("警告: 日志历史记录为空")
        return None
    
    # 首先检查日志中包含哪些字段
    field_patterns = {
        # 字段模式映射到标准字段名
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
    
    print(f"检测到的字段映射: {field_mapping}")
    
    # 提取训练指标
    metrics_data = []
    
    # 分离训练和评估指标
    for log_entry in log_history:
        entry_data = {}
        
        # 提取通用字段
        entry_data["step"] = safe_get(log_entry, "step", 0)
        entry_data["epoch"] = safe_get(log_entry, "epoch", 0)
        
        # 根据entry类型分类处理
        if "loss" in log_entry:
            # 训练指标
            entry_data["loss"] = safe_get(log_entry, field_mapping.get('loss', 'loss'))
            entry_data["learning_rate"] = safe_get(log_entry, field_mapping.get('learning_rate', 'learning_rate'))
            
            # 奖励相关字段
            if 'reward_margin' in field_mapping:
                entry_data["reward_margin"] = safe_get(log_entry, field_mapping['reward_margin'])
            if 'reward_chosen' in field_mapping:
                entry_data["reward_chosen"] = safe_get(log_entry, field_mapping['reward_chosen'])
            if 'reward_rejected' in field_mapping:
                entry_data["reward_rejected"] = safe_get(log_entry, field_mapping['reward_rejected'])
            if 'accuracy' in field_mapping:
                entry_data["accuracy"] = safe_get(log_entry, field_mapping['accuracy'])
            
            # Beta相关字段（如果存在）
            if 'beta' in field_mapping:
                entry_data["beta"] = safe_get(log_entry, field_mapping['beta'])
            if 'pos_beta' in field_mapping:
                entry_data["pos_beta"] = safe_get(log_entry, field_mapping['pos_beta'])
            if 'neg_beta' in field_mapping:
                entry_data["neg_beta"] = safe_get(log_entry, field_mapping['neg_beta'])
                
            entry_data["entry_type"] = "train"
            
        elif "eval_loss" in log_entry:
            # 评估指标
            entry_data["eval_loss"] = safe_get(log_entry, field_mapping.get('eval_loss', 'eval_loss'))
            if 'eval_accuracy' in field_mapping:
                entry_data["eval_accuracy"] = safe_get(log_entry, field_mapping['eval_accuracy'])
            
            entry_data["entry_type"] = "eval"
        
        if entry_data:
            metrics_data.append(entry_data)
    
    # 转换为DataFrame
    if not metrics_data:
        return None
        
    df = pd.DataFrame(metrics_data)
    
    # 确保数值型列的数据类型正确
    numeric_columns = ["loss", "eval_loss", "eval_accuracy", "learning_rate", 
                      "reward_margin", "reward_chosen", "reward_rejected",
                      "accuracy", "beta", "pos_beta", "neg_beta"]
    
    for col in numeric_columns:
        if col in df.columns:
            # 转换非数值为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 输出检测到的字段
    print(f"提取到的DataFrame列: {df.columns.tolist()}")
    
    return df

def analyze_beta_values(ledpo_df: pd.DataFrame) -> Dict:
    """分析LEDPO的beta值分布和变化趋势"""
    if ledpo_df is None or 'beta' not in ledpo_df.columns:
        print("警告: DataFrame中没有beta字段，无法分析beta值")
        return None
    
    # 过滤掉NaN值
    beta_df = ledpo_df[ledpo_df['beta'].notna()].copy()
    if beta_df.empty:
        print("警告: beta值全为NaN，无法分析")
        return None
    
    beta_stats = {}
    
    # 基本统计量
    beta_stats['mean'] = beta_df['beta'].mean()
    beta_stats['median'] = beta_df['beta'].median()
    beta_stats['min'] = beta_df['beta'].min()
    beta_stats['max'] = beta_df['beta'].max()
    beta_stats['std'] = beta_df['beta'].std()
    
    # 开始和结束时的beta值
    beta_stats['start'] = beta_df['beta'].iloc[0] if len(beta_df) > 0 else None
    beta_stats['end'] = beta_df['beta'].iloc[-1] if len(beta_df) > 0 else None
    
    # 如果存在pos_beta和neg_beta，也进行分析
    if 'pos_beta' in beta_df.columns and 'neg_beta' in beta_df.columns:
        pos_beta_df = beta_df[beta_df['pos_beta'].notna()]
        neg_beta_df = beta_df[beta_df['neg_beta'].notna()]
        
        if not pos_beta_df.empty and not neg_beta_df.empty:
            beta_stats['pos_mean'] = pos_beta_df['pos_beta'].mean()
            beta_stats['neg_mean'] = neg_beta_df['neg_beta'].mean()
            beta_stats['pos_end'] = pos_beta_df['pos_beta'].iloc[-1] if len(pos_beta_df) > 0 else None
            beta_stats['neg_end'] = neg_beta_df['neg_beta'].iloc[-1] if len(neg_beta_df) > 0 else None
    
    return beta_stats

def plot_metrics_comparison(dpo_df: pd.DataFrame, ledpo_df: pd.DataFrame, output_dir: str) -> None:
    """绘制DPO和LEDPO指标对比图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 图表1: 训练损失对比
    plt.figure(figsize=(12, 6))
    legends = []
    
    if dpo_df is not None and 'loss' in dpo_df.columns and 'step' in dpo_df.columns:
        # 筛选只包含loss的行并且损失值有效
        dpo_loss_df = dpo_df[(dpo_df['loss'].notna()) & (dpo_df['entry_type'] == 'train')]
        if not dpo_loss_df.empty:
            plt.plot(dpo_loss_df['step'], dpo_loss_df['loss'], 
                    label='Standard DPO', color='blue', marker='o', markersize=4)
            legends.append('Standard DPO')
    
    if ledpo_df is not None and 'loss' in ledpo_df.columns and 'step' in ledpo_df.columns:
        # 筛选只包含loss的行并且损失值有效
        ledpo_loss_df = ledpo_df[(ledpo_df['loss'].notna()) & (ledpo_df['entry_type'] == 'train')]
        if not ledpo_loss_df.empty:
            plt.plot(ledpo_loss_df['step'], ledpo_loss_df['loss'], 
                    label='LEDPO', color='red', marker='x', markersize=4)
            legends.append('LEDPO')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    if legends:
        plt.legend(legends)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图表2: 评估准确率对比
    plt.figure(figsize=(12, 6))
    legends = []
    
    if dpo_df is not None and 'eval_accuracy' in dpo_df.columns:
        # 筛选只包含eval_accuracy的行并且准确率值有效
        dpo_eval_df = dpo_df[(dpo_df['eval_accuracy'].notna()) & (dpo_df['entry_type'] == 'eval')]
        if not dpo_eval_df.empty:
            plt.plot(dpo_eval_df['step'], dpo_eval_df['eval_accuracy'], 
                    label='Standard DPO', color='blue', marker='o', markersize=6)
            legends.append('Standard DPO')
    
    if ledpo_df is not None and 'eval_accuracy' in ledpo_df.columns:
        # 筛选只包含eval_accuracy的行并且准确率值有效
        ledpo_eval_df = ledpo_df[(ledpo_df['eval_accuracy'].notna()) & (ledpo_df['entry_type'] == 'eval')]
        if not ledpo_eval_df.empty:
            plt.plot(ledpo_eval_df['step'], ledpo_eval_df['eval_accuracy'], 
                    label='LEDPO', color='red', marker='x', markersize=6)
            legends.append('LEDPO')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy Comparison')
    if legends:
        plt.legend(legends)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图表3: 奖励差值对比
    plt.figure(figsize=(12, 6))
    legends = []
    
    if dpo_df is not None and 'reward_margin' in dpo_df.columns:
        # 筛选只包含reward_margin的行并且值有效
        dpo_reward_df = dpo_df[dpo_df['reward_margin'].notna()]
        if not dpo_reward_df.empty:
            plt.plot(dpo_reward_df['step'], dpo_reward_df['reward_margin'], 
                    label='Standard DPO', color='blue', marker='o', markersize=4)
            legends.append('Standard DPO')
    
    if ledpo_df is not None and 'reward_margin' in ledpo_df.columns:
        # 筛选只包含reward_margin的行并且值有效
        ledpo_reward_df = ledpo_df[ledpo_df['reward_margin'].notna()]
        if not ledpo_reward_df.empty:
            plt.plot(ledpo_reward_df['step'], ledpo_reward_df['reward_margin'], 
                    label='LEDPO', color='red', marker='x', markersize=4)
            legends.append('LEDPO')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Reward Margin')
    plt.title('Reward Margin Comparison')
    if legends:
        plt.legend(legends)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path / 'reward_margin_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图表4: 准确率对比
    plt.figure(figsize=(12, 6))
    legends = []
    
    if dpo_df is not None and 'accuracy' in dpo_df.columns:
        # 筛选只包含accuracy的行并且值有效
        dpo_acc_df = dpo_df[dpo_df['accuracy'].notna()]
        if not dpo_acc_df.empty:
            plt.plot(dpo_acc_df['step'], dpo_acc_df['accuracy'], 
                    label='Standard DPO', color='blue', marker='o', markersize=4)
            legends.append('Standard DPO')
    
    if ledpo_df is not None and 'accuracy' in ledpo_df.columns:
        # 筛选只包含accuracy的行并且值有效
        ledpo_acc_df = ledpo_df[ledpo_df['accuracy'].notna()]
        if not ledpo_acc_df.empty:
            plt.plot(ledpo_acc_df['step'], ledpo_acc_df['accuracy'], 
                    label='LEDPO', color='red', marker='x', markersize=4)
            legends.append('LEDPO')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison')
    if legends:
        plt.legend(legends)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path / 'training_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图表5: Beta值变化 (仅LEDPO)
    if ledpo_df is not None and 'beta' in ledpo_df.columns:
        plt.figure(figsize=(12, 6))
        legends = []
        
        # 筛选只包含beta的行并且值有效
        beta_df = ledpo_df[ledpo_df['beta'].notna()]
        if not beta_df.empty:
            plt.plot(beta_df['step'], beta_df['beta'], 
                    label='Beta', color='green', marker='o', markersize=4)
            legends.append('Beta')
            
            # 如果有pos_beta和neg_beta并且值有效
            if 'pos_beta' in beta_df.columns and 'neg_beta' in beta_df.columns:
                pos_beta_df = beta_df[beta_df['pos_beta'].notna()]
                neg_beta_df = beta_df[beta_df['neg_beta'].notna()]
                
                if not pos_beta_df.empty:
                    plt.plot(pos_beta_df['step'], pos_beta_df['pos_beta'], 
                            label='Positive Beta', color='blue', marker='x', markersize=4, linestyle='--')
                    legends.append('Positive Beta')
                
                if not neg_beta_df.empty:
                    plt.plot(neg_beta_df['step'], neg_beta_df['neg_beta'], 
                            label='Negative Beta', color='red', marker='+', markersize=4, linestyle='--')
                    legends.append('Negative Beta')
        
            plt.xlabel('Training Steps')
            plt.ylabel('Beta Value')
            plt.title('LEDPO Beta Value Evolution')
            if legends:
                plt.legend(legends)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(output_path / 'beta_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 图表6: 奖励对比 (chosen vs rejected)
    plt.figure(figsize=(12, 6))
    legends = []
    
    # DPO的奖励对比
    if dpo_df is not None and 'reward_chosen' in dpo_df.columns and 'reward_rejected' in dpo_df.columns:
        dpo_reward_df = dpo_df[(dpo_df['reward_chosen'].notna()) & (dpo_df['reward_rejected'].notna())]
        if not dpo_reward_df.empty:
            plt.plot(dpo_reward_df['step'], dpo_reward_df['reward_chosen'], 
                    label='DPO Chosen', color='blue', marker='o', markersize=4, linestyle='-')
            plt.plot(dpo_reward_df['step'], dpo_reward_df['reward_rejected'], 
                    label='DPO Rejected', color='blue', marker='x', markersize=4, linestyle='--')
            legends.extend(['DPO Chosen', 'DPO Rejected'])
    
    # LEDPO的奖励对比
    if ledpo_df is not None and 'reward_chosen' in ledpo_df.columns and 'reward_rejected' in ledpo_df.columns:
        ledpo_reward_df = ledpo_df[(ledpo_df['reward_chosen'].notna()) & (ledpo_df['reward_rejected'].notna())]
        if not ledpo_reward_df.empty:
            plt.plot(ledpo_reward_df['step'], ledpo_reward_df['reward_chosen'], 
                    label='LEDPO Chosen', color='red', marker='o', markersize=4, linestyle='-')
            plt.plot(ledpo_reward_df['step'], ledpo_reward_df['reward_rejected'], 
                    label='LEDPO Rejected', color='red', marker='x', markersize=4, linestyle='--')
            legends.extend(['LEDPO Chosen', 'LEDPO Rejected'])
    
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('Chosen vs Rejected Rewards')
    if legends:
        plt.legend(legends)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path / 'rewards_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_last_valid_value(df: pd.DataFrame, column: str) -> Any:
    """获取DataFrame中某列的最后一个有效值"""
    if df is None or column not in df.columns:
        return "N/A"
    
    valid_values = df[df[column].notna()][column]
    if valid_values.empty:
        return "N/A"
    
    last_value = valid_values.iloc[-1]
    if isinstance(last_value, float):
        return f"{last_value:.4f}"
    return last_value

def get_max_valid_value(df: pd.DataFrame, column: str) -> Any:
    """获取DataFrame中某列的最大有效值"""
    if df is None or column not in df.columns:
        return "N/A"
    
    valid_values = df[df[column].notna()][column]
    if valid_values.empty:
        return "N/A"
    
    max_value = valid_values.max()
    if isinstance(max_value, float):
        return f"{max_value:.4f}"
    return max_value

def generate_summary_report(dpo_df: pd.DataFrame, ledpo_df: pd.DataFrame, 
                          ledpo_beta_stats: Dict, output_dir: str) -> None:
    """生成实验结果摘要报告"""
    output_path = Path(output_dir) / 'summary_report.md'
    
    with open(output_path, 'w') as f:
        f.write("# DPO vs LEDPO 实验结果摘要\n\n")
        
        # 训练信息
        f.write("## 训练信息\n\n")
        
        if dpo_df is not None:
            # 分离训练和评估数据
            dpo_train_df = dpo_df[dpo_df['entry_type'] == 'train'] if 'entry_type' in dpo_df.columns else dpo_df
            dpo_eval_df = dpo_df[dpo_df['entry_type'] == 'eval'] if 'entry_type' in dpo_df.columns else None
            
            dpo_last_step = dpo_train_df['step'].max() if 'step' in dpo_train_df.columns else 'N/A'
            dpo_last_loss = get_last_valid_value(dpo_train_df, 'loss')
            dpo_best_acc = get_max_valid_value(dpo_eval_df, 'eval_accuracy')
            dpo_train_acc = get_max_valid_value(dpo_train_df, 'accuracy')
            dpo_last_reward_margin = get_last_valid_value(dpo_train_df, 'reward_margin')
            
            f.write(f"### 标准DPO\n")
            f.write(f"- 训练步数: {dpo_last_step}\n")
            f.write(f"- 最终损失: {dpo_last_loss}\n")
            f.write(f"- 训练准确率: {dpo_train_acc}\n")
            f.write(f"- 最佳评估准确率: {dpo_best_acc}\n")
            f.write(f"- 最终奖励差值: {dpo_last_reward_margin}\n\n")
        else:
            f.write("### 标准DPO\n")
            f.write("- 无数据\n\n")
        
        if ledpo_df is not None:
            # 分离训练和评估数据
            ledpo_train_df = ledpo_df[ledpo_df['entry_type'] == 'train'] if 'entry_type' in ledpo_df.columns else ledpo_df
            ledpo_eval_df = ledpo_df[ledpo_df['entry_type'] == 'eval'] if 'entry_type' in ledpo_df.columns else None
            
            ledpo_last_step = ledpo_train_df['step'].max() if 'step' in ledpo_train_df.columns else 'N/A'
            ledpo_last_loss = get_last_valid_value(ledpo_train_df, 'loss')
            ledpo_best_acc = get_max_valid_value(ledpo_eval_df, 'eval_accuracy')
            ledpo_train_acc = get_max_valid_value(ledpo_train_df, 'accuracy')
            ledpo_last_reward_margin = get_last_valid_value(ledpo_train_df, 'reward_margin')
            
            f.write(f"### LEDPO\n")
            f.write(f"- 训练步数: {ledpo_last_step}\n")
            f.write(f"- 最终损失: {ledpo_last_loss}\n")
            f.write(f"- 训练准确率: {ledpo_train_acc}\n")
            f.write(f"- 最佳评估准确率: {ledpo_best_acc}\n")
            f.write(f"- 最终奖励差值: {ledpo_last_reward_margin}\n\n")
        else:
            f.write("### LEDPO\n")
            f.write("- 无数据\n\n")
        
        # Beta统计（如果存在）
        if ledpo_beta_stats:
            f.write("## Beta值分析\n\n")
            f.write("### 统计信息\n")
            f.write(f"- 平均值: {ledpo_beta_stats['mean']:.4f}\n")
            f.write(f"- 中位数: {ledpo_beta_stats['median']:.4f}\n")
            f.write(f"- 最小值: {ledpo_beta_stats['min']:.4f}\n")
            f.write(f"- 最大值: {ledpo_beta_stats['max']:.4f}\n")
            f.write(f"- 标准差: {ledpo_beta_stats['std']:.4f}\n\n")
            
            f.write("### 变化趋势\n")
            f.write(f"- 初始值: {ledpo_beta_stats['start']:.4f}\n")
            f.write(f"- 最终值: {ledpo_beta_stats['end']:.4f}\n")
            
            if 'pos_mean' in ledpo_beta_stats:
                f.write("\n### 正负样本Beta\n")
                f.write(f"- 平均正样本Beta: {ledpo_beta_stats['pos_mean']:.4f}\n")
                f.write(f"- 平均负样本Beta: {ledpo_beta_stats['neg_mean']:.4f}\n")
                f.write(f"- 最终正样本Beta: {ledpo_beta_stats['pos_end']:.4f}\n")
                f.write(f"- 最终负样本Beta: {ledpo_beta_stats['neg_end']:.4f}\n")
        
        # 添加一个注释，说明Beta值的情况
        else:
            f.write("## Beta值情况\n\n")
            f.write("在训练日志中没有找到Beta相关的记录。这可能有以下几种原因：\n\n")
            f.write("1. LEDPO模型没有正确地将Beta值记录到训练日志中\n")
            f.write("2. Beta值使用了不同的字段名称\n")
            f.write("3. 当前的LEDPO实现不输出动态Beta值\n\n")
            f.write("建议检查LEDPO训练器的实现，确保Beta值被正确计算并记录到训练日志中。\n\n")
        
        # 结论
        f.write("\n## 对比结论\n\n")
        
        # 自动生成一些基本分析
        if dpo_df is not None and ledpo_df is not None:
            dpo_loss = float(get_last_valid_value(dpo_df, 'loss').replace('N/A', '0'))
            ledpo_loss = float(get_last_valid_value(ledpo_df, 'loss').replace('N/A', '0'))
            
            f.write("### 性能对比\n")
            if ledpo_loss < dpo_loss:
                f.write(f"- LEDPO的最终损失({ledpo_loss:.4f})低于标准DPO({dpo_loss:.4f})，表明LEDPO可能在训练效果上有一定优势\n")
            elif ledpo_loss > dpo_loss:
                f.write(f"- 标准DPO的最终损失({dpo_loss:.4f})低于LEDPO({ledpo_loss:.4f})，表明在这个实验中标准DPO表现更好\n")
            else:
                f.write(f"- LEDPO和标准DPO的最终损失相近，两者性能差异不明显\n")
                
            # 分析奖励差值
            if 'reward_margin' in dpo_df.columns and 'reward_margin' in ledpo_df.columns:
                dpo_margin = get_last_valid_value(dpo_df, 'reward_margin')
                ledpo_margin = get_last_valid_value(ledpo_df, 'reward_margin')
                
                if dpo_margin != 'N/A' and ledpo_margin != 'N/A':
                    dpo_margin_float = float(dpo_margin)
                    ledpo_margin_float = float(ledpo_margin)
                    
                    if ledpo_margin_float > dpo_margin_float:
                        f.write(f"- LEDPO的奖励差值({ledpo_margin})大于标准DPO({dpo_margin})，这表明LEDPO在区分正负样本方面更有效\n")
                    elif ledpo_margin_float < dpo_margin_float:
                        f.write(f"- 标准DPO的奖励差值({dpo_margin})大于LEDPO({ledpo_margin})，这表明在此实验中标准DPO能更好地区分正负样本\n")
                    else:
                        f.write(f"- LEDPO和标准DPO的奖励差值相近，两者在区分正负样本的能力上差异不明显\n")
                        
            # 分析准确率
            if 'accuracy' in dpo_df.columns and 'accuracy' in ledpo_df.columns:
                dpo_acc = get_max_valid_value(dpo_df, 'accuracy')
                ledpo_acc = get_max_valid_value(ledpo_df, 'accuracy')
                
                if dpo_acc != 'N/A' and ledpo_acc != 'N/A':
                    f.write(f"- LEDPO的训练准确率为{ledpo_acc}，标准DPO为{dpo_acc}\n")
                
            f.write("\n### 分析与建议\n")
            f.write("- 目前的实验规模较小，需要更多训练步骤和更全面的评估以得出确定结论\n")
            f.write("- 由于没有记录Beta值相关数据，无法验证LEDPO的核心优势\n")
            f.write("- 建议修改LEDPO实现，确保Beta值及其相关统计被正确记录\n")
            
            f.write("\n### 下一步实验建议\n")
            f.write("1. 修改LEDPO训练器，确保Beta值被正确记录\n")
            f.write("2. 增加训练步数，观察长期趋势\n")
            f.write("3. 使用更大的数据集进行测试\n")
            f.write("4. 添加更多评估指标，全面比较模型性能\n")
        else:
            f.write("### 性能对比\n")
            f.write("- 由于缺少数据，无法进行详细比较\n\n")
            
            f.write("### Beta值特性\n")
            f.write("- 没有检测到Beta值数据\n\n")
            
            f.write("### 建议\n")
            f.write("- 确保两个模型都正确运行并记录训练数据\n")
    
    print(f"摘要报告已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='分析DPO与LEDPO实验结果')
    parser.add_argument('--dpo_dir', type=str, default='experiments/dpo_vs_ledpo_demo/results/dpo',
                        help='标准DPO结果目录')
    parser.add_argument('--ledpo_dir', type=str, default='experiments/dpo_vs_ledpo_demo/results/ledpo',
                        help='LEDPO结果目录')
    parser.add_argument('--output_dir', type=str, default='experiments/dpo_vs_ledpo_demo/analysis',
                        help='分析结果输出目录')
    parser.add_argument('--verbose', action='store_true',
                        help='是否输出详细信息')
    
    args = parser.parse_args()
    
    print(f"分析标准DPO结果：{args.dpo_dir}")
    print(f"分析LEDPO结果：{args.ledpo_dir}")
    
    # 加载trainer_state
    dpo_state = load_trainer_state(args.dpo_dir)
    ledpo_state = load_trainer_state(args.ledpo_dir)
    
    # 提取指标
    print("提取标准DPO指标...")
    dpo_df = extract_metrics(dpo_state)
    print("提取LEDPO指标...")
    ledpo_df = extract_metrics(ledpo_state)
    
    # 分析beta值
    ledpo_beta_stats = analyze_beta_values(ledpo_df) if ledpo_df is not None else None
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制指标对比图
    print("生成对比图表...")
    plot_metrics_comparison(dpo_df, ledpo_df, args.output_dir)
    
    # 生成摘要报告
    print("生成摘要报告...")
    generate_summary_report(dpo_df, ledpo_df, ledpo_beta_stats, args.output_dir)
    
    print(f"分析完成，结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 