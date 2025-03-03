#!/usr/bin/env python
# 简化的调试脚本：用于测试BetaDPO数据处理和绘图功能

import os
import sys
import json
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 调试日志
def log(message):
    print(f"DEBUG: {message}")

# 设置基础目录
base_dir = "output/betadpo_demo"
scales = ["0.0", "0.1"]
results = {}
available_scales = []

# 创建输出目录
os.makedirs(base_dir, exist_ok=True)

# 主要调试函数
def debug_data_loading():
    """测试数据加载部分"""
    log("开始测试数据加载...")
    
    for scale in scales:
        scale_dir = os.path.join(base_dir, f"scale_{scale}")
        train_file = os.path.join(scale_dir, "trainer_state.json")
        
        log(f"检查文件: {train_file}")
        if os.path.exists(train_file):
            log(f"  ✓ 文件存在")
            try:
                with open(train_file, "r") as f:
                    data = json.load(f)
                
                log(f"  ✓ 成功读取JSON数据")
                log(f"  - 文件内容的键: {list(data.keys())}")
                
                # 提取loss和rewards/accuracies
                if "log_history" in data:
                    log_history = data["log_history"]
                    log(f"  - 日志条目数: {len(log_history)}")
                    log(f"  - 第一条日志内容的键: {list(log_history[0].keys()) if log_history else '无'}")
                    
                    for i, entry in enumerate(log_history):
                        log(f"  - 日志条目 {i}: {entry}")
                    
                    if log_history:
                        # 现在我们知道log_history[0]包含loss和beta/current_value
                        # log_history[1]包含train_loss等其他统计信息
                        first_log = log_history[0] if len(log_history) > 0 else {}
                        last_log = log_history[-1] if len(log_history) > 0 else {}
                        
                        # 从第一条日志提取loss和beta
                        loss_val = first_log.get("loss", 0)
                        beta_val = first_log.get("beta/current_value", None)
                        
                        # 从最后一条日志提取step信息
                        step_val = last_log.get("step", 0)
                        
                        # 输出值用于验证
                        log(f"  - 提取的值: step={step_val}, loss={loss_val}, beta={beta_val}")
                        
                        # 保存数据到结果字典
                        results[scale] = {
                            "step": step_val,
                            "loss": loss_val,
                            "beta_value": beta_val
                        }
                        available_scales.append(scale)
                        
                else:
                    log(f"  ✗ 未找到log_history键")
            except Exception as e:
                log(f"  ✗ 处理数据时出错: {str(e)}")
                traceback.print_exc()
        else:
            log(f"  ✗ 文件不存在")
    
    log(f"可用的scales: {available_scales}")
    for scale in available_scales:
        log(f"scale_{scale} 的结果: {results[scale]}")
    
    return len(available_scales) > 0

def debug_simple_plotting():
    """测试简单的绘图功能"""
    log("开始测试简单绘图...")
    
    if not available_scales:
        log("没有可用数据，无法绘图")
        return False
    
    try:
        # 创建一个简单的条形图
        plt.figure(figsize=(10, 6))
        
        # 收集损失值数据
        labels = []
        loss_values = []
        
        for scale in available_scales:
            scale_label = "Fixed Beta" if scale == "0.0" else f"Dynamic Beta (c={scale})"
            labels.append(scale_label)
            
            # 现在loss是标量值，不再是列表
            loss_val = results[scale]["loss"]
            loss_values.append(loss_val)
            log(f"{scale_label} 的 loss 值: {loss_val}")
        
        # 绘制条形图
        x_pos = np.arange(len(labels))
        bars = plt.bar(x_pos, loss_values)
        plt.xticks(x_pos, labels)
        plt.ylabel("Loss Value")
        plt.title("Loss Comparison")
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{height:.4f}", ha="center", va="bottom")
        
        # 保存图表
        output_path = os.path.join(base_dir, "debug_simple_plot.png")
        plt.savefig(output_path)
        log(f"简单图表已保存到: {output_path}")
        return True
    
    except Exception as e:
        log(f"绘图时出错: {str(e)}")
        traceback.print_exc()
        return False

def debug_full_plotting():
    """测试完整的绘图功能，逐步构建"""
    log("开始测试完整绘图...")
    
    if not available_scales:
        log("没有可用数据，无法绘图")
        return False
    
    try:
        # 创建一个多面板图表
        plt.figure(figsize=(15, 10))
        
        # 1. 损失值对比 (柱状图)
        plt.subplot(2, 2, 1)
        labels = []
        loss_values = []
        
        for scale in available_scales:
            labels.append("Fixed Beta" if scale == "0.0" else f"Dynamic Beta (c={scale})")
            loss_val = results[scale]["loss"]
            loss_values.append(float(loss_val))
        
        x_pos = np.arange(len(labels))
        bars = plt.bar(x_pos, loss_values)
        plt.xticks(x_pos, labels)
        plt.ylabel("Loss Value")
        plt.title("Loss Comparison")
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{height:.4f}", ha="center", va="bottom")
        
        # 2. Beta值对比
        plt.subplot(2, 2, 2)
        labels = []
        beta_values = []
        
        for scale in available_scales:
            beta_val = results[scale]["beta_value"]
            if beta_val is not None:
                labels.append("Fixed Beta" if scale == "0.0" else f"Dynamic Beta (c={scale})")
                beta_values.append(float(beta_val))
        
        if beta_values:
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
        else:
            plt.text(0.5, 0.5, "No Beta Values Available", 
                    ha="center", va="center", fontsize=12)
            plt.axis("off")
        
        # 3. 算法比较摘要表格
        plt.subplot(2, 1, 2)
        plt.axis("off")  # 关闭坐标轴
        
        # 创建表格数据
        table_data = []
        table_data.append(["Metric", "Fixed Beta", "Dynamic Beta", "Improvement"])
        
        # 损失值行
        fixed_loss = loss_values[0] if len(loss_values) > 0 else None
        dynamic_loss = loss_values[1] if len(loss_values) > 1 else None
        
        if fixed_loss is not None and dynamic_loss is not None:
            # 修复零除错误
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
        plt.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        plt.title("BetaDPO Algorithm Comparison Summary")
        
        # 保存图表
        plt.tight_layout()
        output_path = os.path.join(base_dir, "debug_full_plot.png")
        plt.savefig(output_path, dpi=300)
        log(f"完整图表已保存到: {output_path}")
        
        # 创建单独的文本摘要
        with open(os.path.join(base_dir, "debug_comparison_summary.txt"), "w") as f:
            f.write("BetaDPO Algorithm Comparison Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # 写入表格
            col_widths = [15, 15, 15, 15]
            
            # 表头
            header = "| "
            for i, col in enumerate(table_data[0]):
                header += f"{col:{col_widths[i]}} | "
            f.write(header + "\n")
            
            # 分隔线
            separator = "| "
            for width in col_widths:
                separator += "-" * width + " | "
            f.write(separator + "\n")
            
            # 数据行
            for row in table_data[1:]:
                data_row = "| "
                for i, cell in enumerate(row):
                    data_row += f"{cell:{col_widths[i]}} | "
                f.write(data_row + "\n")
            
            # 结论
            f.write("\nAnalysis Conclusion:\n")
            if fixed_loss is not None and dynamic_loss is not None:
                if dynamic_loss < fixed_loss:
                    f.write("✓ Dynamic beta strategy has lower loss than fixed beta, performs better\n")
                else:
                    f.write("✗ Dynamic beta strategy has higher loss than fixed beta, performs worse\n")
            else:
                f.write("Insufficient data, cannot draw conclusions\n")
        
        return True
    
    except Exception as e:
        log(f"绘制完整图表时出错: {str(e)}")
        traceback.print_exc()
        return False

def debug_raw_data():
    """直接打印原始数据，不做任何处理"""
    for scale in scales:
        scale_dir = os.path.join(base_dir, f"scale_{scale}")
        train_file = os.path.join(scale_dir, "trainer_state.json")
        
        if os.path.exists(train_file):
            try:
                with open(train_file, "r") as f:
                    data = json.load(f)
                log(f"scale_{scale} 原始数据:")
                log(json.dumps(data, indent=2))
            except Exception as e:
                log(f"读取 scale_{scale} 原始数据出错: {str(e)}")

if __name__ == "__main__":
    # 首先打印原始数据
    debug_raw_data()
    
    # 测试数据加载
    if debug_data_loading():
        # 测试简单绘图
        if debug_simple_plotting():
            # 测试完整绘图
            debug_full_plotting()
    else:
        log("数据加载失败，无法进行绘图测试") 