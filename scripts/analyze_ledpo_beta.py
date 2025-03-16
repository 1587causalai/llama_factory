#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEDPO Beta分析脚本

此脚本用于分析LEDPO训练过程中的beta值变化情况，
可以加载训练过程中保存的beta历史数据并生成可视化报告。
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


def load_beta_history(file_path):
    """加载保存的beta历史数据"""
    if not os.path.exists(file_path):
        print(f"错误: 找不到beta历史数据文件: {file_path}")
        sys.exit(1)
    
    try:
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"加载beta历史数据时出错: {e}")
        sys.exit(1)


def plot_comprehensive_beta_analysis(data, output_dir, prefix=""):
    """绘制全面的beta分析图表"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取训练数据
    train_data = data["train"]
    eval_data = data["eval"]
    
    # 检查是否有足够的数据
    if len(train_data["steps"]) < 2:
        print("警告: 训练数据不足，无法绘制分析图表")
        return
    
    # 设置图表风格
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 创建多面板图表
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Beta Scale趋势图 - 左上角
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_data["steps"], train_data["beta_scale"], 'b-', label='训练集')
    if eval_data["steps"]:
        ax1.plot(eval_data["steps"], eval_data["beta_scale"], 'r-', label='验证集')
    ax1.set_xlabel("训练步数")
    ax1.set_ylabel("Beta Scale")
    ax1.set_title("Beta Scale 变化趋势")
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 2. Pos/Neg Beta趋势图 - 右上角
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(train_data["steps"], train_data["pos_beta"], 'g-', linewidth=2, label='正delta beta (训练)')
    ax2.plot(train_data["steps"], train_data["neg_beta"], 'r-', linewidth=2, label='负delta beta (训练)')
    if eval_data["steps"]:
        ax2.plot(eval_data["steps"], eval_data["pos_beta"], 'g--', alpha=0.7, label='正delta beta (验证)')
        ax2.plot(eval_data["steps"], eval_data["neg_beta"], 'r--', alpha=0.7, label='负delta beta (验证)')
    
    # 添加beta_min参考线
    avg_beta_min = np.mean(train_data["pos_beta"][:5]) * 0.1  # 估计beta_min
    ax2.axhline(y=avg_beta_min, color='k', linestyle='--', alpha=0.5, label='估计beta_min')
    
    ax2.set_xlabel("训练步数")
    ax2.set_ylabel("Beta值")
    ax2.set_title("正/负Delta对应的Beta值变化趋势")
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 3. Beta比值趋势图 - 中左
    ax3 = fig.add_subplot(gs[1, 0])
    ratio_data = [p/n if n > 0 else 0 for p, n in zip(train_data["pos_beta"], train_data["neg_beta"])]
    ax3.plot(train_data["steps"], ratio_data, 'b-', label='训练集')
    if eval_data["steps"]:
        eval_ratio = [p/n if n > 0 else 0 for p, n in zip(eval_data["pos_beta"], eval_data["neg_beta"])]
        ax3.plot(eval_data["steps"], eval_ratio, 'r-', label='验证集')
    
    # 添加比值为1的参考线
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel("训练步数")
    ax3.set_ylabel("Pos Beta / Neg Beta")
    ax3.set_title("Beta正负比值趋势")
    ax3.legend()
    ax3.grid(True)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 4. 损失变化图 - 中中
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(train_data["steps"], train_data["loss"], 'b-', label='训练集')
    if eval_data["steps"]:
        ax4.plot(eval_data["steps"], eval_data["loss"], 'r-', label='验证集')
    ax4.set_xlabel("训练步数")
    ax4.set_ylabel("损失值")
    ax4.set_title("训练/验证损失变化")
    ax4.legend()
    ax4.grid(True)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 5. Beta分布热图 - 中右
    ax5 = fig.add_subplot(gs[1, 2])
    
    # 计算beta值的2D直方图 (pos_beta vs neg_beta)
    if len(train_data["pos_beta"]) > 10:
        # 确定显示范围 
        xmax = max(train_data["pos_beta"]) * 1.1
        ymax = max(train_data["neg_beta"]) * 1.1
        xmin = min(train_data["pos_beta"]) * 0.9
        ymin = min(train_data["neg_beta"]) * 0.9
        
        # 创建网格
        xbins = np.linspace(xmin, xmax, 20)
        ybins = np.linspace(ymin, ymax, 20)
        
        # 计算2D直方图
        h, xedges, yedges = np.histogram2d(train_data["pos_beta"], train_data["neg_beta"], 
                                           bins=[xbins, ybins])
        
        # 绘制热图
        im = ax5.imshow(h.T, interpolation='nearest', origin='lower', aspect='auto',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        cmap='plasma')
        
        # 添加对角线
        diag_min = min(xmin, ymin)
        diag_max = max(xmax, ymax)
        ax5.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', alpha=0.5)
        
        # 添加colorbar
        plt.colorbar(im, ax=ax5, label='频率')
        
    ax5.set_xlabel("正delta Beta值")
    ax5.set_ylabel("负delta Beta值")
    ax5.set_title("Beta值分布热图")
    ax5.grid(True)
    
    # 6. Beta Scale与Beta Ratio关系散点图 - 下左
    ax6 = fig.add_subplot(gs[2, 0])
    sc = ax6.scatter(train_data["beta_scale"], ratio_data, c=train_data["steps"], 
                     cmap='viridis', alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
    ax6.set_xlabel("Beta Scale")
    ax6.set_ylabel("Pos Beta / Neg Beta")
    ax6.set_title("Beta Scale与Beta Ratio关系")
    plt.colorbar(sc, ax=ax6, label='训练步数')
    ax6.grid(True)
    
    # 7. Beta Scale与训练损失关系图 - 下中
    ax7 = fig.add_subplot(gs[2, 1])
    sc2 = ax7.scatter(train_data["beta_scale"], train_data["loss"], c=train_data["steps"],
                      cmap='viridis', alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
    ax7.set_xlabel("Beta Scale")
    ax7.set_ylabel("训练损失")
    ax7.set_title("Beta Scale与训练损失关系")
    plt.colorbar(sc2, ax=ax7, label='训练步数')
    ax7.grid(True)
    
    # 8. 统计摘要 - 下右
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # 计算统计摘要
    if len(train_data["steps"]) >= 10:
        last_20_percent = max(1, len(train_data["steps"]) // 5)
        
        initial_beta_scale = train_data["beta_scale"][0]
        final_beta_scale = train_data["beta_scale"][-1]
        beta_scale_change = final_beta_scale - initial_beta_scale
        
        mean_pos_beta = np.mean(train_data["pos_beta"][-last_20_percent:])
        mean_neg_beta = np.mean(train_data["neg_beta"][-last_20_percent:])
        pos_neg_ratio = mean_pos_beta / mean_neg_beta if mean_neg_beta > 0 else 0
        
        # 确定是否有beta趋零问题
        beta_min_estimate = avg_beta_min
        has_zero_issue = mean_pos_beta < beta_min_estimate * 2 or mean_neg_beta < beta_min_estimate * 2
        
        # 确定beta区分度
        has_good_diff = pos_neg_ratio > 1.2
        
        # 添加统计摘要文本
        summary = [
            "LEDPO Beta 统计摘要",
            "=" * 25,
            f"分析步数: {len(train_data['steps'])}",
            f"初始 beta_scale: {initial_beta_scale:.4f}",
            f"最终 beta_scale: {final_beta_scale:.4f} ({'+' if beta_scale_change > 0 else ''}{beta_scale_change:.4f})",
            f"最终正delta beta平均值: {mean_pos_beta:.4f}",
            f"最终负delta beta平均值: {mean_neg_beta:.4f}",
            f"最终beta正负比值: {pos_neg_ratio:.4f}",
            "",
            "分析结论:",
            f"Beta趋零问题: {'存在 ❌' if has_zero_issue else '不存在 ✓'}",
            f"Beta区分度: {'良好 ✓' if has_good_diff else '不足 ❌'}",
            f"总体评价: {'健康 ✓' if not has_zero_issue and has_good_diff else '存在问题 ❌'}"
        ]
        
        y_pos = 0.95
        for line in summary:
            if line.startswith("=") or line == "":
                y_pos -= 0.05
            elif line.startswith("LEDPO") or line.startswith("分析结论"):
                ax8.text(0.05, y_pos, line, fontsize=12, fontweight='bold')
                y_pos -= 0.05
            else:
                ax8.text(0.05, y_pos, line, fontsize=10)
                y_pos -= 0.05
    
    # 调整总体布局
    plt.suptitle(f"LEDPO Beta 全面分析报告{' - ' + prefix if prefix else ''}", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图表
    output_path = os.path.join(output_dir, f"ledpo_beta_analysis{'_' + prefix if prefix else ''}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"分析图表已保存到: {output_path}")
    
    # 将关键统计数据保存为JSON
    stats = {
        "data_points": len(train_data["steps"]),
        "initial_beta_scale": float(train_data["beta_scale"][0]) if train_data["beta_scale"] else 0,
        "final_beta_scale": float(train_data["beta_scale"][-1]) if train_data["beta_scale"] else 0,
        "final_pos_beta": float(np.mean(train_data["pos_beta"][-last_20_percent:])) if train_data["pos_beta"] else 0,
        "final_neg_beta": float(np.mean(train_data["neg_beta"][-last_20_percent:])) if train_data["neg_beta"] else 0,
        "final_pos_neg_ratio": float(pos_neg_ratio),
        "has_zero_issue": bool(has_zero_issue),
        "has_good_differentiation": bool(has_good_diff),
        "success": bool(not has_zero_issue and has_good_diff)
    }
    
    json_path = os.path.join(output_dir, f"ledpo_beta_stats{'_' + prefix if prefix else ''}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"统计数据已保存到: {json_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LEDPO Beta分析工具")
    parser.add_argument("--data", "-d", type=str, required=True, 
                        help="beta_history.npy文件路径")
    parser.add_argument("--output", "-o", type=str, default="./ledpo_analysis_results",
                        help="分析结果输出目录")
    parser.add_argument("--prefix", "-p", type=str, default="",
                        help="输出文件的前缀")
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_beta_history(args.data)
    
    # 绘制分析图表
    plot_comprehensive_beta_analysis(data, args.output, args.prefix)
    
    print("LEDPO Beta分析完成!")


if __name__ == "__main__":
    main() 