#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析冻结策略模型与非冻结策略模型的实验日志
比较beta和delta值的分布与变化趋势
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional

# 删除中文字体设置，使用默认字体
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100

def load_trainer_state(log_dir: str) -> Dict:
    """加载trainer_state.json文件"""
    trainer_state_path = os.path.join(log_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"找不到trainer_state.json文件: {trainer_state_path}")
    
    with open(trainer_state_path, "r") as f:
        return json.load(f)

def extract_metrics(trainer_state: Dict) -> pd.DataFrame:
    """从trainer_state中提取指标数据，并转换为DataFrame"""
    log_history = trainer_state.get("log_history", [])
    if not log_history:
        raise ValueError("日志历史记录为空")
    
    # 提取训练指标
    metrics_data = []
    for log_entry in log_history:
        # 排除非训练指标记录
        if "loss" not in log_entry and "eval_loss" not in log_entry:
            continue
        
        # 构建指标记录
        entry_data = {"step": log_entry.get("step", 0)}
        
        # 提取所有指标
        for key, value in log_entry.items():
            if key not in ["step", "epoch"]:
                entry_data[key] = value
        
        metrics_data.append(entry_data)
    
    return pd.DataFrame(metrics_data)

def analyze_beta_delta(df: pd.DataFrame) -> Dict:
    """分析beta和delta值的统计特性"""
    results = {}
    
    # 获取beta值统计信息
    if "beta/mean" in df.columns:
        beta_values = df["beta/mean"].dropna().values
        if len(beta_values) > 0:
            results["beta_mean"] = np.mean(beta_values)
            results["beta_std"] = np.std(beta_values)
            results["beta_min"] = np.min(beta_values)
            results["beta_max"] = np.max(beta_values)
            results["beta_change"] = beta_values[-1] - beta_values[0] if len(beta_values) > 1 else 0
    
    # 获取delta值统计信息
    if "delta" in df.columns:
        delta_values = df["delta"].dropna().values
        if len(delta_values) > 0:
            results["delta_mean"] = np.mean(delta_values)
            results["delta_std"] = np.std(delta_values)
            results["delta_min"] = np.min(delta_values)
            results["delta_max"] = np.max(delta_values)
            
            # 计算delta符号分布
            positive_delta = np.sum(delta_values > 0)
            results["delta_positive_ratio"] = positive_delta / len(delta_values)
    
    # 计算beta和delta的相关性
    if "beta/mean" in df.columns and "delta" in df.columns:
        common_indexes = df[["beta/mean", "delta"]].dropna().index
        if len(common_indexes) > 0:
            beta_for_corr = df.loc[common_indexes, "beta/mean"].values
            delta_for_corr = df.loc[common_indexes, "delta"].values
            if len(beta_for_corr) > 1:  # 需要至少两个点才能计算相关性
                results["beta_delta_correlation"] = np.corrcoef(beta_for_corr, delta_for_corr)[0, 1]
    
    return results

def plot_metrics_comparison(models_data: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """绘制不同模型的指标对比图表
    
    Args:
        models_data: 字典，键为模型名称，值为该模型的DataFrame数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义模型颜色和标记映射
    # 保留中文模型名作为字典键，但图表中会使用英文标签
    colors = {
        "冻结模型": "blue", 
        "冻结模型-新ref": "green", 
        "非冻结模型": "red"
    }
    markers = {
        "冻结模型": "o", 
        "冻结模型-新ref": "s", 
        "非冻结模型": "x"
    }
    
    # 模型名称映射到英文
    model_name_map = {
        "冻结模型": "Frozen Model",
        "冻结模型-新ref": "Frozen Model-New Ref", 
        "非冻结模型": "Non-Frozen Model"
    }
    
    # 创建图表1：beta值对比
    plt.figure(figsize=(12, 6))
    has_beta_data = False
    
    for model_name, df in models_data.items():
        if "beta/mean" in df.columns:
            plt.plot(df["step"], df["beta/mean"], 
                    label=model_name_map.get(model_name, model_name), 
                    color=colors.get(model_name, "gray"), 
                    marker=markers.get(model_name, "."), 
                    alpha=0.7)
            has_beta_data = True
    
    if has_beta_data:
        plt.xlabel("Training Steps")
        plt.ylabel("Beta Value")
        plt.title("Beta Value Comparison Between Models")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "beta_comparison.png"))
    plt.close()
    
    # 创建图表2：delta值对比
    plt.figure(figsize=(12, 6))
    has_delta_data = False
    
    for model_name, df in models_data.items():
        if "delta" in df.columns:
            plt.plot(df["step"], df["delta"], 
                    label=model_name_map.get(model_name, model_name), 
                    color=colors.get(model_name, "gray"), 
                    marker=markers.get(model_name, "."), 
                    alpha=0.7)
            has_delta_data = True
    
    if has_delta_data:
        plt.xlabel("Training Steps")
        plt.ylabel("Delta Value")
        plt.title("Delta Value Comparison Between Models")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "delta_comparison.png"))
    plt.close()
    
    # 创建图表3：beta与delta的散点图 (每个模型一个子图)
    n_models = len(models_data)
    if n_models > 0:
        fig = plt.figure(figsize=(5 * n_models, 5))
        gs = GridSpec(1, n_models, figure=fig, wspace=0.3)
        
        for i, (model_name, df) in enumerate(models_data.items()):
            if "beta/mean" in df.columns and "delta" in df.columns:
                ax = fig.add_subplot(gs[0, i])
                
                common_indexes = df[["beta/mean", "delta"]].dropna().index
                if len(common_indexes) > 0:
                    beta_values = df.loc[common_indexes, "beta/mean"]
                    delta_values = df.loc[common_indexes, "delta"]
                    
                    ax.scatter(beta_values, delta_values, 
                              color=colors.get(model_name, "gray"), 
                              alpha=0.7)
                    ax.set_xlabel("Beta Value")
                    ax.set_ylabel("Delta Value")
                    ax.set_title(f"Beta vs Delta: {model_name_map.get(model_name, model_name)}")
                    ax.grid(True, alpha=0.3)
                    
                    # 添加趋势线
                    if len(beta_values) > 1:
                        z = np.polyfit(beta_values, delta_values, 1)
                        p = np.poly1d(z)
                        ax.plot(beta_values, p(beta_values), "r--", alpha=0.7)
                        
                        # 添加相关系数文本
                        corr = np.corrcoef(beta_values, delta_values)[0, 1]
                        ax.text(0.05, 0.95, f"Correlation: {corr:.4f}", transform=ax.transAxes, 
                               bbox=dict(facecolor='white', alpha=0.7))
        
        plt.savefig(os.path.join(output_dir, "beta_delta_scatter.png"))
    plt.close()
    
    # 创建图表4：统计指标汇总
    plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=plt.gcf(), wspace=0.3, hspace=0.4)
    
    # 子图1：beta值直方图对比
    ax1 = plt.subplot(gs[0, 0])
    has_beta_data = False
    
    for model_name, df in models_data.items():
        if "beta/mean" in df.columns:
            ax1.hist(df["beta/mean"].dropna(), bins=20, alpha=0.5, 
                    label=model_name_map.get(model_name, model_name), 
                    color=colors.get(model_name, "gray"))
            has_beta_data = True
    
    if has_beta_data:
        ax1.set_xlabel("Beta Value")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Beta Value Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 子图2：delta值直方图对比
    ax2 = plt.subplot(gs[0, 1])
    has_delta_data = False
    
    for model_name, df in models_data.items():
        if "delta" in df.columns:
            ax2.hist(df["delta"].dropna(), bins=20, alpha=0.5, 
                    label=model_name_map.get(model_name, model_name), 
                    color=colors.get(model_name, "gray"))
            has_delta_data = True
    
    if has_delta_data:
        ax2.set_xlabel("Delta Value")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Delta Value Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 子图3：正负delta比例对比
    ax3 = plt.subplot(gs[1, 0])
    has_delta_data = False
    
    model_names = []
    pos_ratios = []
    neg_ratios = []
    
    for model_name, df in models_data.items():
        if "delta" in df.columns:
            delta_values = df["delta"].dropna().values
            if len(delta_values) > 0:
                pos_ratio = np.sum(delta_values > 0) / len(delta_values)
                neg_ratio = 1 - pos_ratio
                
                model_names.append(model_name_map.get(model_name, model_name))
                pos_ratios.append(pos_ratio)
                neg_ratios.append(neg_ratio)
                has_delta_data = True
    
    if has_delta_data:
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, pos_ratios, width, label="Positive Delta", color="green", alpha=0.7)
        ax3.bar(x + width/2, neg_ratios, width, label="Negative Delta", color="orange", alpha=0.7)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.set_ylabel("Ratio")
        ax3.set_title("Positive/Negative Delta Ratio")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 添加百分比标签
        for i, v in enumerate(pos_ratios):
            ax3.text(i - width/2, v + 0.02, f"{v:.1%}", ha="center")
        for i, v in enumerate(neg_ratios):
            ax3.text(i + width/2, v + 0.02, f"{v:.1%}", ha="center")
    
    # 子图4：beta值变化趋势
    ax4 = plt.subplot(gs[1, 1])
    has_beta_data = False
    
    for model_name, df in models_data.items():
        if "beta/mean" in df.columns:
            steps = df["step"]
            beta = df["beta/mean"]
            
            # 计算移动平均以平滑曲线
            window = 5
            if len(beta) >= window:
                beta_smooth = beta.rolling(window=window, min_periods=1).mean()
            else:
                beta_smooth = beta
            
            ax4.plot(steps, beta_smooth, 
                    label=model_name_map.get(model_name, model_name), 
                    color=colors.get(model_name, "gray"), 
                    alpha=0.7)
            
            # 添加起点和终点标记
            ax4.scatter([steps.iloc[0], steps.iloc[-1]], 
                       [beta.iloc[0], beta.iloc[-1]], 
                       color=colors.get(model_name, "gray"), 
                       s=80, zorder=5, 
                       marker=markers.get(model_name, "."))
            
            has_beta_data = True
    
    if has_beta_data:
        ax4.set_xlabel("Training Steps")
        ax4.set_ylabel("Beta Value (Moving Average)")
        ax4.set_title("Beta Value Trend")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
    plt.close()

def generate_report(models_stats: Dict[str, Dict], output_dir: str) -> None:
    """生成分析报告
    
    Args:
        models_stats: 字典，键为模型名称，值为该模型的统计数据
        output_dir: 输出目录
    """
    # 模型名称映射到英文（仅用于报告标题和小节标题）
    model_name_map = {
        "冻结模型": "Frozen Model",
        "冻结模型-新ref": "Frozen Model-New Ref", 
        "非冻结模型": "Non-Frozen Model"
    }
    
    report_path = os.path.join(output_dir, "analysis_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Policy Model Experiment Analysis Report\n\n")
        
        # 1. Beta值统计
        f.write("## 1. Beta Value Statistics\n\n")
        f.write("| Metric | " + " | ".join(models_stats.keys()) + " |\n")
        f.write("|------|" + "|".join(["------" for _ in models_stats]) + "|\n")
        
        metrics = [
            ("Mean", "beta_mean", "%.4f"),
            ("Std Dev", "beta_std", "%.4f"),
            ("Min", "beta_min", "%.4f"),
            ("Max", "beta_max", "%.4f"),
            ("Change", "beta_change", "%.4f"),
        ]
        
        for name, key, fmt in metrics:
            values = []
            for model_stats in models_stats.values():
                val = fmt % model_stats.get(key, 0) if key in model_stats else "N/A"
                values.append(val)
            f.write(f"| {name} | " + " | ".join(values) + " |\n")
        
        # 2. Delta值统计
        f.write("\n## 2. Delta Value Statistics\n\n")
        f.write("| Metric | " + " | ".join(models_stats.keys()) + " |\n")
        f.write("|------|" + "|".join(["------" for _ in models_stats]) + "|\n")
        
        metrics = [
            ("Mean", "delta_mean", "%.4f"),
            ("Std Dev", "delta_std", "%.4f"),
            ("Min", "delta_min", "%.4f"),
            ("Max", "delta_max", "%.4f"),
            ("Positive Ratio", "delta_positive_ratio", "%.2f%%"),
        ]
        
        for name, key, fmt in metrics:
            values = []
            for model_stats in models_stats.values():
                if key == "delta_positive_ratio":
                    val = fmt % (model_stats.get(key, 0) * 100) if key in model_stats else "N/A"
                else:
                    val = fmt % model_stats.get(key, 0) if key in model_stats else "N/A"
                values.append(val)
            f.write(f"| {name} | " + " | ".join(values) + " |\n")
        
        # 3. Beta与Delta相关性
        f.write("\n## 3. Beta-Delta Correlation\n\n")
        f.write("| Model | Correlation |\n")
        f.write("|------|----------|\n")
        
        for model_name, model_stats in models_stats.items():
            corr = "%.4f" % model_stats.get("beta_delta_correlation", 0) if "beta_delta_correlation" in model_stats else "N/A"
            f.write(f"| {model_name} | {corr} |\n")
        
        # 4. 分析结论
        f.write("\n## 4. Analysis Conclusions\n\n")
        
        # 分别分析每个模型
        f.write("### Delta Value Analysis\n\n")
        
        for model_name, model_stats in models_stats.items():
            delta_mean = model_stats.get("delta_mean", 0)
            model_name_eng = model_name_map.get(model_name, model_name)
            
            f.write(f"**{model_name_eng}**:\n")
            
            if "冻结模型" in model_name and abs(delta_mean) < 1e-6:
                f.write(f"- Delta value is close to 0 ({delta_mean:.4f}), which meets expectations. Since the policy model is frozen, its output remains unchanged, resulting in constant delta values.\n")
            elif "冻结模型" in model_name:
                f.write(f"- Delta mean is {delta_mean:.4f}, which deviates from theoretical expectations. This might be due to numerical errors or differences in model initialization.\n")
            elif delta_mean > 0:
                f.write(f"- Delta mean is {delta_mean:.4f} (positive), indicating that the model successfully learned preference relationships during training, and tends to assign higher probabilities to preferred samples.\n")
            else:
                f.write(f"- Delta mean is {delta_mean:.4f} (negative), which is abnormal. This might indicate issues in the training process, where the model failed to learn preference relationships correctly.\n")
        
        f.write("\n### Beta Value Analysis\n\n")
        
        for model_name, model_stats in models_stats.items():
            beta_change = model_stats.get("beta_change", 0)
            model_name_eng = model_name_map.get(model_name, model_name)
            
            f.write(f"**{model_name_eng}**:\n")
            
            if abs(beta_change) > 0.01:
                f.write(f"- Beta value changed by {beta_change:.4f} from beginning to end, indicating that the beta_head is indeed learning.\n")
            else:
                f.write(f"- Beta value changed very little ({beta_change:.4f}), which may indicate that the beta_head learning is limited or already close to optimal value.\n")
            
        f.write("\n### Correlation Analysis\n\n")
        
        for model_name, model_stats in models_stats.items():
            corr = model_stats.get("beta_delta_correlation", 0)
            model_name_eng = model_name_map.get(model_name, model_name)
            
            f.write(f"**{model_name_eng}**:\n")
            
            if abs(corr) > 0.3:
                direction = "positive" if corr > 0 else "negative"
                f.write(f"- There is a {direction} correlation ({corr:.4f}) between beta and delta, indicating that the beta_head is adapting to the data distribution.\n")
            else:
                f.write(f"- There is almost no correlation ({corr:.4f}) between beta and delta. More training rounds may be needed to observe a clear relationship.\n")
        
        f.write("\n### Summary\n\n")
        
        # 分析冻结模型与非冻结模型的主要区别
        frozen_models = {k: v for k, v in models_stats.items() if "冻结模型" in k}
        unfrozen_models = {k: v for k, v in models_stats.items() if "非冻结模型" in k}
        
        if frozen_models and unfrozen_models:
            # 获取第一个模型的数据作为代表
            frozen_name = list(frozen_models.keys())[0]
            frozen_delta = frozen_models[frozen_name].get("delta_mean", 0)
            
            unfrozen_name = list(unfrozen_models.keys())[0]
            unfrozen_delta = unfrozen_models[unfrozen_name].get("delta_mean", 0)
            
            delta_diff = abs(unfrozen_delta - frozen_delta)
            
            if abs(frozen_delta) < 1e-6 and unfrozen_delta > 0:
                f.write("The experimental results confirm our hypothesis: the delta value remains constant when the policy model is frozen, while the non-frozen model successfully learns preference relationships. "
                       "This indicates that the implementation of the LEDPO algorithm is correct, and the introduction of beta_head indeed provides the model with the ability to dynamically adjust beta values.\n\n")
            elif abs(frozen_delta) < 1e-6 and unfrozen_delta <= 0:
                f.write("The experimental results partially meet expectations: the delta value remains constant when the policy model is frozen, but the non-frozen model does not learn preference relationships well. "
                       "It is recommended to increase training rounds or adjust the learning rate to improve learning outcomes.\n\n")
            elif abs(frozen_delta) >= 1e-6 and unfrozen_delta > 0:
                f.write("The experimental results show that the delta value of the frozen model is not completely zero, but the non-frozen model successfully learns preference relationships. "
                       "It is recommended to check if the freezing implementation is completely correct, while it can be considered that the LEDPO algorithm is generally effective.\n\n")
            else:
                f.write("The experimental results differ significantly from expectations. It is recommended to check the model freezing implementation, dataset quality, and training parameter settings, and consider increasing training rounds.\n\n")
        
        # 分析不同类型冻结模型之间的差异
        if len(frozen_models) > 1:
            f.write("\n### Comparison of Different Frozen Models\n\n")
            model_names = list(frozen_models.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    name1, name2 = model_names[i], model_names[j]
                    name1_eng = model_name_map.get(name1, name1)
                    name2_eng = model_name_map.get(name2, name2)
                    delta1 = frozen_models[name1].get("delta_mean", 0)
                    delta2 = frozen_models[name2].get("delta_mean", 0)
                    beta_change1 = frozen_models[name1].get("beta_change", 0)
                    beta_change2 = frozen_models[name2].get("beta_change", 0)
                    
                    f.write(f"**{name1_eng} vs {name2_eng}**:\n")
                    f.write(f"- Delta value difference: {abs(delta1 - delta2):.4f}\n")
                    f.write(f"- Beta change difference: {abs(beta_change1 - beta_change2):.4f}\n")
                    
                    if abs(beta_change1) > abs(beta_change2):
                        f.write(f"- {name1_eng} has a larger beta change, which may indicate more effective learning.\n")
                    elif abs(beta_change1) < abs(beta_change2):
                        f.write(f"- {name2_eng} has a larger beta change, which may indicate more effective learning.\n")
                    else:
                        f.write(f"- Both models have similar beta changes.\n")
        
        f.write("\n## 5. Recommendations\n\n")
        
        f.write("1. **Increase Training Rounds**: Consider increasing training rounds from the current 3 to 5-10, to observe beta and delta trends over a longer period.\n")
        f.write("2. **Adjust Learning Rate**: Try setting a higher learning rate for beta_head to facilitate faster adaptation to data distribution.\n")
        f.write("3. **Use Different Initializations**: Try initializing models with different random seeds to verify the stability of results.\n")
        f.write("4. **Expand Dataset**: Use larger datasets, or test algorithm performance on data from different domains.\n")
        f.write("5. **Monitor Gradient Flow**: Add gradient monitoring for beta_head parameters to verify gradient flow and confirm actual learning.\n")
        f.write("6. **Further Study of Reference Model Impact**: Continue researching the impact of different reference models on frozen policy experiments to explore best practices.\n")

def main():
    parser = argparse.ArgumentParser(description="分析冻结与非冻结策略模型的实验结果")
    parser.add_argument("--frozen_dir", type=str, required=True, help="冻结模型结果目录")
    parser.add_argument("--frozen_diff_ref_dir", type=str, help="使用不同ref模型的冻结模型结果目录")
    parser.add_argument("--unfrozen_dir", type=str, required=True, help="非冻结模型结果目录")
    parser.add_argument("--output_dir", type=str, required=True, help="分析结果输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集所有模型数据
    models_data = {}
    models_stats = {}
    
    # 加载冻结模型数据
    print(f"开始分析冻结模型数据: {args.frozen_dir}")
    frozen_trainer_state = load_trainer_state(args.frozen_dir)
    frozen_df = extract_metrics(frozen_trainer_state)
    frozen_stats = analyze_beta_delta(frozen_df)
    
    models_data["冻结模型"] = frozen_df
    models_stats["冻结模型"] = frozen_stats
    
    # 加载非冻结模型数据
    print(f"开始分析非冻结模型数据: {args.unfrozen_dir}")
    unfrozen_trainer_state = load_trainer_state(args.unfrozen_dir)
    unfrozen_df = extract_metrics(unfrozen_trainer_state)
    unfrozen_stats = analyze_beta_delta(unfrozen_df)
    
    models_data["非冻结模型"] = unfrozen_df
    models_stats["非冻结模型"] = unfrozen_stats
    
    # 可选：加载使用不同ref模型的冻结模型数据
    if args.frozen_diff_ref_dir:
        print(f"开始分析使用不同ref模型的冻结模型数据: {args.frozen_diff_ref_dir}")
        try:
            frozen_diff_ref_trainer_state = load_trainer_state(args.frozen_diff_ref_dir)
            frozen_diff_ref_df = extract_metrics(frozen_diff_ref_trainer_state)
            frozen_diff_ref_stats = analyze_beta_delta(frozen_diff_ref_df)
            
            models_data["冻结模型-新ref"] = frozen_diff_ref_df
            models_stats["冻结模型-新ref"] = frozen_diff_ref_stats
        except FileNotFoundError as e:
            print(f"警告: {e}")
            print("继续分析其他模型数据...")
    
    print("生成对比图表...")
    plot_metrics_comparison(models_data, args.output_dir)
    
    print("生成分析报告...")
    generate_report(models_stats, args.output_dir)
    
    print(f"分析完成! 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 