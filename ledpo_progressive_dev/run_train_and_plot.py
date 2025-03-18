#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完全干净版本：运行标准训练，然后使用绘图脚本生成监控图表
支持标准DPO指标以及LEDPO特定指标(pos_beta, neg_beta)的监控

使用方法:
    python ledpo_progressive_dev/run_train_and_plot.py --config ledpo_progressive_dev/qwen15_lora_foodpo.yaml --wandb_project ledpo_monitoring_test
    
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行训练和绘图')
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default='', help='wandb项目名称')
    args = parser.parse_args()
    
    # 确保配置文件存在
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return 1
    
    # 构建训练命令
    train_cmd = ['llamafactory-cli', 'train', config_path]
    if args.wandb_project:
        os.environ['WANDB_PROJECT'] = args.wandb_project
    
    # 获取输出目录
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    output_dir = config.get('output_dir', 'results/default')
    
    # 输出信息
    print('=' * 60)
    print(f"启动训练，使用配置: {config_path}")
    print('=' * 60)
    
    # 运行训练
    train_result = subprocess.run(train_cmd)
    if train_result.returncode != 0:
        print(f"训练失败，返回码: {train_result.returncode}")
        return train_result.returncode
    
    # 获取trainer_state.json路径
    trainer_state_path = os.path.join(output_dir, 'trainer_state.json')
    if not os.path.exists(trainer_state_path):
        print(f"警告: 无法找到trainer_state.json: {trainer_state_path}")
        # 不中断执行
    else:
        # 现在脚本将图片直接输出到结果目录，不再创建子目录
        plot_cmd = ['python', 'ledpo_progressive_dev/plot_ledpo_metrics.py', '--result_dir', output_dir]
        subprocess.run(plot_cmd)
    
    print('=' * 60)
    print(f"训练和绘图完成，结果保存在: {output_dir}")
    print('=' * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 