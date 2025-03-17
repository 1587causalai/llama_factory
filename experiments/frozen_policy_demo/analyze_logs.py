#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深入分析实验日志脚本
检查beta值重合现象的原因，提取更详细的统计信息
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# 设置字体和样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.figsize'] = (14, 10)

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(SCRIPT_DIR, 'results/frozen')
UNFROZEN_DIR = os.path.join(SCRIPT_DIR, 'results/unfrozen')
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, 'analysis')

# 确保分析目录存在
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def load_json_log(filepath):
    """加载JSON格式的训练日志"""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"JSON解析错误: {filepath}")
            return None

def extract_all_metrics(log_data, include_eval=True):
    """提取所有指标，包括训练和评估指标"""
    train_entries = []
    eval_entries = []
    
    if log_data is None or "log_history" not in log_data:
        return pd.DataFrame(), pd.DataFrame()
    
    for entry in log_data["log_history"]:
        # 提取训练指标
        if "step" in entry and "train_loss" not in entry and "eval_" not in str(entry.keys()):
            # 标准指标
            metrics = {
                "step": entry.get("step"),
                "loss": entry.get("loss"),
                "rewards/accuracies": entry.get("rewards/accuracies"),
                "rewards/margins": entry.get("rewards/margins"),
                "rewards/chosen": entry.get("rewards/chosen"),
                "rewards/rejected": entry.get("rewards/rejected"),
            }
            
            # beta和delta相关指标
            metrics["delta"] = entry.get("delta")
            metrics["beta"] = entry.get("beta/mean", entry.get("beta", 0.1))
            metrics["pos_beta"] = entry.get("pos_beta")
            metrics["neg_beta"] = entry.get("neg_beta")
            
            # 提取所有额外beta统计数据
            for key, value in entry.items():
                if "beta" in key.lower() and key not in ["beta", "beta/mean", "pos_beta", "neg_beta"]:
                    metrics[key] = value
            
            train_entries.append(metrics)
        
        # 提取评估指标
        elif include_eval and "eval_" in str(entry.keys()) and "step" in entry:
            eval_metrics = {"step": entry.get("step")}
            
            # 提取所有eval_前缀的指标
            for key, value in entry.items():
                if key.startswith("eval_"):
                    # 移除eval_前缀以便于比较
                    clean_key = key[5:]  # 移除"eval_"
                    eval_metrics[clean_key] = value
            
            eval_entries.append(eval_metrics)
    
    train_df = pd.DataFrame(train_entries)
    eval_df = pd.DataFrame(eval_entries)
    
    return train_df, eval_df

def analyze_beta_values(frozen_train, unfrozen_train, frozen_eval=None, unfrozen_eval=None):
    """分析beta值的统计特性"""
    print("\n=== Beta值统计分析 ===")
    
    # 训练集上的beta值分析
    if not frozen_train.empty and "beta" in frozen_train.columns:
        frozen_beta = frozen_train["beta"].dropna()
        print(f"冻结模型训练集beta值统计:")
        print(f"  平均值: {frozen_beta.mean():.4f}")
        print(f"  标准差: {frozen_beta.std():.4f}")
        print(f"  变异系数: {frozen_beta.std() / frozen_beta.mean():.4f}")
        print(f"  最小值: {frozen_beta.min():.4f}")
        print(f"  最大值: {frozen_beta.max():.4f}")
        print(f"  变化范围: {frozen_beta.max() - frozen_beta.min():.4f}")
        # 检查beta值是否有变化
        if abs(frozen_beta.max() - frozen_beta.min()) < 1e-6:
            print("  警告: 冻结模型beta值几乎没有变化!")
    
    if not unfrozen_train.empty and "beta" in unfrozen_train.columns:
        unfrozen_beta = unfrozen_train["beta"].dropna()
        print(f"\n非冻结模型训练集beta值统计:")
        print(f"  平均值: {unfrozen_beta.mean():.4f}")
        print(f"  标准差: {unfrozen_beta.std():.4f}")
        print(f"  变异系数: {unfrozen_beta.std() / unfrozen_beta.mean():.4f}")
        print(f"  最小值: {unfrozen_beta.min():.4f}")
        print(f"  最大值: {unfrozen_beta.max():.4f}")
        print(f"  变化范围: {unfrozen_beta.max() - unfrozen_beta.min():.4f}")
        if abs(unfrozen_beta.max() - unfrozen_beta.min()) < 1e-6:
            print("  警告: 非冻结模型beta值几乎没有变化!")
    
    # 检查是否有pos_beta和neg_beta数据
    for df, model_type in [(frozen_train, "冻结模型"), (unfrozen_train, "非冻结模型")]:
        if not df.empty:
            has_pos_beta = "pos_beta" in df.columns and not df["pos_beta"].isna().all()
            has_neg_beta = "neg_beta" in df.columns and not df["neg_beta"].isna().all()
            
            if has_pos_beta and has_neg_beta:
                pos_beta = df["pos_beta"].dropna()
                neg_beta = df["neg_beta"].dropna()
                
                print(f"\n{model_type} pos_beta/neg_beta对比:")
                print(f"  pos_beta平均值: {pos_beta.mean():.4f}")
                print(f"  neg_beta平均值: {neg_beta.mean():.4f}")
                print(f"  差异: {pos_beta.mean() - neg_beta.mean():.4f}")
    
    # 如果有评估数据，进行类似分析...
    
    # 比较冻结和非冻结模型
    if (not frozen_train.empty and "beta" in frozen_train.columns and 
        not unfrozen_train.empty and "beta" in unfrozen_train.columns):
        # 获取相同步数的beta值
        common_steps = set(frozen_train["step"]) & set(unfrozen_train["step"])
        if common_steps:
            print("\n冻结与非冻结模型beta值对比 (相同训练步):")
            frozen_filtered = frozen_train[frozen_train["step"].isin(common_steps)]
            unfrozen_filtered = unfrozen_train[unfrozen_train["step"].isin(common_steps)]
            
            # 计算每步的差异
            diffs = []
            for step in sorted(common_steps):
                f_beta = frozen_filtered[frozen_filtered["step"] == step]["beta"].values[0]
                u_beta = unfrozen_filtered[unfrozen_filtered["step"] == step]["beta"].values[0]
                diffs.append(abs(f_beta - u_beta))
            
            print(f"  平均绝对差异: {np.mean(diffs):.6f}")
            print(f"  最大绝对差异: {np.max(diffs):.6f}")
            print(f"  最小绝对差异: {np.min(diffs):.6f}")
            
            if np.mean(diffs) < 1e-5:
                print("  结论: 两个模型的beta值确实非常接近!")
                print("  可能原因: beta_head参数未被充分更新，或模型初始化时使用了相同的种子")

def analyze_delta_values(frozen_train, unfrozen_train, frozen_eval=None, unfrozen_eval=None):
    """分析delta值的统计特性以及与beta值的关系"""
    print("\n=== Delta值统计分析 ===")
    
    # 检查delta值
    for df, model_type in [(frozen_train, "冻结模型"), (unfrozen_train, "非冻结模型")]:
        if not df.empty and "delta" in df.columns and not df["delta"].isna().all():
            delta = df["delta"].dropna()
            print(f"\n{model_type} delta值统计:")
            print(f"  平均值: {delta.mean():.4f}")
            print(f"  标准差: {delta.std():.4f}")
            print(f"  最小值: {delta.min():.4f}")
            print(f"  最大值: {delta.max():.4f}")
            print(f"  正值比例: {(delta > 0).mean():.2%}")
            
            # 检查delta是否全为常数
            if len(delta) > 1:
                if (delta == delta.iloc[0]).all():
                    print(f"  注意: 所有delta值都相同 ({delta.iloc[0]:.4f})")
                    # 当delta值全相同时跳过变化率计算
                    if model_type == "冻结模型":
                        print("  冻结模型delta值完全不变，符合冻结策略模型的预期")
                else:
                    # 只有在delta不是常数时才计算变化率
                    delta_changes = np.abs(np.diff(delta))
                    print(f"  变化率(平均绝对差): {np.mean(delta_changes):.4f}")
                    
                    # 检查delta是否随时间有系统性变化
                    steps = df["step"].dropna().values[:len(delta_changes)]
                    if len(steps) > 1:
                        try:
                            delta_trend = np.polyfit(steps, delta_changes, 1)[0]
                            print(f"  趋势斜率: {delta_trend:.6f}")
                            if model_type == "冻结模型" and abs(delta_trend) < 1e-4:
                                print("  冻结模型delta值变化趋于稳定，符合预期")
                        except Exception as e:
                            print(f"  无法计算趋势: {e}")
    
    # 分析delta和beta的关系
    for df, model_type in [(frozen_train, "冻结模型"), (unfrozen_train, "非冻结模型")]:
        if (not df.empty and "delta" in df.columns and not df["delta"].isna().all() and
            "beta" in df.columns and not df["beta"].isna().all()):
            
            # 准备数据
            data = df[["step", "delta", "beta"]].dropna()
            if len(data) > 1:
                # 检查delta值是否有变化
                if (data["delta"] == data["delta"].iloc[0]).all():
                    print(f"\n{model_type} delta值全相同，无法计算与beta的相关性")
                else:
                    # 计算相关性
                    try:
                        corr = data["delta"].corr(data["beta"])
                        print(f"\n{model_type} delta与beta的相关性: {corr:.4f}")
                        
                        if abs(corr) < 0.1:
                            print("  delta和beta几乎没有相关性")
                        elif corr > 0:
                            print("  delta和beta呈正相关")
                        else:
                            print("  delta和beta呈负相关")
                    except Exception as e:
                        print(f"  无法计算相关性: {e}")

def plot_beta_distribution(frozen_train, unfrozen_train, output_dir):
    """绘制beta值的分布对比"""
    plt.figure(figsize=(12, 6))
    
    # 绘制beta分布
    if not frozen_train.empty and "beta" in frozen_train.columns:
        frozen_beta = frozen_train["beta"].dropna()
        if len(frozen_beta) > 0:
            sns.kdeplot(frozen_beta, label="Frozen Model", color="blue")
    
    if not unfrozen_train.empty and "beta" in unfrozen_train.columns:
        unfrozen_beta = unfrozen_train["beta"].dropna()
        if len(unfrozen_beta) > 0:
            sns.kdeplot(unfrozen_beta, label="Unfrozen Model", color="red")
    
    plt.title("Beta Value Distribution", fontsize=16)
    plt.xlabel("Beta Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "beta_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Beta分布图已保存: {output_path}")
    plt.close()

def plot_detailed_metrics(frozen_train, unfrozen_train, output_dir):
    """绘制更详细的指标变化图表"""
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Delta随时间变化
    ax = axes[0, 0]
    if not frozen_train.empty and "delta" in frozen_train.columns:
        ax.plot(frozen_train["step"], frozen_train["delta"], 
                marker='o', linestyle='-', color='blue', alpha=0.8, label="Frozen")
    if not unfrozen_train.empty and "delta" in unfrozen_train.columns:
        ax.plot(unfrozen_train["step"], unfrozen_train["delta"], 
                marker='s', linestyle='-', color='red', alpha=0.8, label="Unfrozen")
    ax.set_title("Delta Value Over Training Steps", fontsize=14)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Delta Value", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Beta随时间变化
    ax = axes[0, 1]
    if not frozen_train.empty and "beta" in frozen_train.columns:
        ax.plot(frozen_train["step"], frozen_train["beta"], 
                marker='o', linestyle='-', color='blue', alpha=0.8, label="Frozen")
    if not unfrozen_train.empty and "beta" in unfrozen_train.columns:
        ax.plot(unfrozen_train["step"], unfrozen_train["beta"], 
                marker='s', linestyle='-', color='red', alpha=0.8, label="Unfrozen")
    ax.set_title("Beta Value Over Training Steps", fontsize=14)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Beta Value", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Delta值的变化率
    ax = axes[1, 0]
    for df, label, color in [(frozen_train, "Frozen", "blue"), (unfrozen_train, "Unfrozen", "red")]:
        if not df.empty and "delta" in df.columns and len(df) > 1:
            delta = df["delta"].dropna()
            if len(delta) > 1 and not (delta == delta.iloc[0]).all():  # 确保delta值不全相同
                steps = df["step"].dropna().iloc[:-1].values
                delta_changes = np.abs(np.diff(delta))
                ax.plot(steps, delta_changes, marker='o', linestyle='-', color=color, 
                        alpha=0.8, label=f"{label} Delta Change")
    ax.set_title("Delta Value Change Rate", fontsize=14)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Absolute Change in Delta", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Delta-Beta散点图
    ax = axes[1, 1]
    for df, label, color in [(frozen_train, "Frozen", "blue"), (unfrozen_train, "Unfrozen", "red")]:
        if (not df.empty and "delta" in df.columns and not df["delta"].isna().all() and
            "beta" in df.columns and not df["beta"].isna().all()):
            
            data = df[["delta", "beta"]].dropna()
            
            # 只有数据点足够且delta有变化时才绘制散点图和趋势线
            if len(data) > 1 and not (data["delta"] == data["delta"].iloc[0]).all():
                # 绘制散点图
                ax.scatter(data["delta"], data["beta"], alpha=0.8, c=color, label=label)
                
                try:
                    # 安全地尝试添加趋势线
                    z = np.polyfit(data["delta"], data["beta"], 1)
                    p = np.poly1d(z)
                    delta_range = np.linspace(data["delta"].min(), data["delta"].max(), 100)
                    ax.plot(delta_range, p(delta_range), '--', color=color)
                except Exception as e:
                    print(f"警告: 无法为{label}模型绘制趋势线: {e}")
            elif len(data) > 0:
                # 如果delta值全相同但有数据，仍然绘制散点
                ax.scatter(data["delta"], data["beta"], alpha=0.8, c=color, label=label)
                print(f"警告: {label}模型的delta值没有变化, 无法绘制趋势线")
    
    ax.set_title("Beta vs Delta Relationship", fontsize=14)
    ax.set_xlabel("Delta Value", fontsize=12)
    ax.set_ylabel("Beta Value", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "detailed_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"详细指标图已保存: {output_path}")
    plt.close(fig)

def check_beta_initialization():
    """检查beta_head初始化情况"""
    print("\n=== Beta初始化分析 ===")
    beta_base = 0.1  # 配置文件中的值
    
    # 从trainer.py中分析初始化方式
    print(f"配置中beta_base值: {beta_base}")
    print("初始化策略:")
    print(f"  1. beta_head使用beta_base={beta_base}初始化")
    print(f"  2. 使用Softplus激活函数确保beta值非负")
    print(f"  3. 最后一层偏置初始化为: log(exp(beta_base) - 1)")
    
    # 计算初始偏置值
    bias_init = np.log(np.exp(beta_base) - 1.0)
    print(f"  4. 计算得到的初始偏置值: {bias_init:.6f}")
    
    # 分析偏置初始化的影响
    print("\n初始化对beta值的影响:")
    print(f"  - 相同的beta_base值会导致相同的初始偏置")
    print(f"  - 如果训练不充分，两个模型可能保持接近初始值")
    print(f"  - 标准的随机初始化会导致投影层权重相似")

def check_learning_dynamics():
    """分析学习动态和冻结影响"""
    print("\n=== 学习动态分析 ===")
    print("冻结策略模型影响:")
    print("  1. 冻结策略模型使hidden_states保持不变")
    print("  2. beta_head的输入分布稳定，可能导致学习缓慢")
    print("  3. 对于相同的输入，beta_head会产生相同或相似的输出")
    
    print("\nbeta_head学习率影响:")
    print("  1. beta_head使用10倍基础学习率")
    print("  2. 若训练轮次少(3轮)，可能不足以看到显著变化")
    print("  3. 即使学习率较高，若输入分布稳定，学习也会受限")
    
    print("\n日志记录方式的影响:")
    print("  1. 记录的是批次平均值(beta/mean)")
    print("  2. 即使个体样本beta值有差异，平均值可能相似")
    print("  3. 当样本量小时，随机性可能被平均掉")

def main():
    """主函数: 分析实验日志并生成报告"""
    print("=== 开始分析实验日志 ===")
    
    # 加载日志数据
    frozen_log = load_json_log(os.path.join(FROZEN_DIR, 'trainer_state.json'))
    unfrozen_log = load_json_log(os.path.join(UNFROZEN_DIR, 'trainer_state.json'))
    
    if frozen_log is None and unfrozen_log is None:
        print("错误: 无法加载任何日志数据")
        return
    
    # 提取所有指标
    frozen_train, frozen_eval = extract_all_metrics(frozen_log)
    unfrozen_train, unfrozen_eval = extract_all_metrics(unfrozen_log)
    
    # 打印基本信息
    print("\n基本信息:")
    print(f"冻结模型训练数据点: {len(frozen_train)}")
    print(f"非冻结模型训练数据点: {len(unfrozen_train)}")
    print(f"冻结模型评估数据点: {len(frozen_eval)}")
    print(f"非冻结模型评估数据点: {len(unfrozen_eval)}")
    
    # 分析beta和delta值
    analyze_beta_values(frozen_train, unfrozen_train, frozen_eval, unfrozen_eval)
    analyze_delta_values(frozen_train, unfrozen_train, frozen_eval, unfrozen_eval)
    
    # 检查beta初始化和学习动态
    check_beta_initialization()
    check_learning_dynamics()
    
    # 绘制图表
    plot_beta_distribution(frozen_train, unfrozen_train, ANALYSIS_DIR)
    plot_detailed_metrics(frozen_train, unfrozen_train, ANALYSIS_DIR)
    
    print(f"\n分析完成，结果保存在: {ANALYSIS_DIR}")

if __name__ == "__main__":
    main() 