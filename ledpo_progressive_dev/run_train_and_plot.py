#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完全干净版本：运行标准训练，然后使用绘图脚本生成监控图表
支持标准DPO指标以及LEDPO特定指标(pos_beta, neg_beta)的监控
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# 添加src目录到系统路径
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def main():
    """主函数，按顺序执行训练和绘图"""
    parser = argparse.ArgumentParser(description='运行训练并生成监控图表')
    parser.add_argument('--config', type=str, default='ledpo_progressive_dev/qwen15_lora_foodpo.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default='ledpo_monitoring',
                        help='W&B项目名称')
    parser.add_argument('--no_plot', action='store_true',
                        help='仅进行训练，不生成图表')
    args = parser.parse_args()
    
    # 设置wandb项目名称
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # 1. 运行训练
    print("=" * 60)
    print(f"启动训练，使用配置: {args.config}")
    print("=" * 60)
    
    # 使用llamafactory-cli运行训练
    train_cmd = ["llamafactory-cli", "train", args.config]
    train_process = subprocess.run(train_cmd)
    
    if train_process.returncode != 0:
        print(f"训练失败，返回码: {train_process.returncode}")
        return
    
    # 提取输出目录
    output_dir = None
    with open(args.config, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('output_dir:'):
                output_dir = line.split(':', 1)[1].strip()
                break
    
    if not output_dir:
        print("无法从配置中找到output_dir")
        return
    
    # 确保输出目录是绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repo_root, output_dir)
    
    # 2. 如果需要，生成图表
    if not args.no_plot:
        print("\n" + "=" * 60)
        print(f"训练完成，开始生成图表...")
        print("=" * 60)
        
        # 等待1秒确保所有日志都已写入
        time.sleep(1)
        
        # 调用绘图脚本
        plot_script = os.path.join(os.path.dirname(__file__), 'plot_ledpo_metrics.py')
        plot_cmd = [sys.executable, plot_script, "--result_dir", output_dir]
        
        plot_process = subprocess.run(plot_cmd)
        
        if plot_process.returncode != 0:
            print(f"绘图失败，返回码: {plot_process.returncode}")
            return
        
        print("\n" + "=" * 60)
        print(f"图表已生成到目录: {os.path.join(output_dir, 'ledpo_plots')}")
        print("=" * 60)
    
    print("\n所有操作已完成！")
    
if __name__ == "__main__":
    main() 