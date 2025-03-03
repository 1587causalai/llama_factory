#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import re

def main():
    """
    运行Qwen DPO实验的Python脚本版本
    等效于run_qwen_dpo_beta.sh，但提供了更好的调试能力
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行Qwen DPO实验')
    parser.add_argument('exp_name', nargs='?', default='default', help='实验名称')
    parser.add_argument('beta_value', nargs='?', default='0.1', help='DPO beta值')
    args = parser.parse_args()

    exp_name = args.exp_name
    beta_value = args.beta_value

    # 打印参数信息
    if exp_name == 'default':
        print(f"未提供实验名称，使用默认名称: {exp_name}")
    if beta_value == '0.1':
        print(f"未提供beta值，使用默认值: {beta_value}")

    # 获取当前时间戳作为运行ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"{exp_name}_beta{beta_value}_{timestamp}"

    # 确保我们在正确的工作目录
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (假设scripts/dpo相对于根目录的位置固定)
    root_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    # 切换到根目录
    os.chdir(root_dir)

    # 创建配置文件
    config_dir = os.path.join(root_dir, "configs", "dpo")
    base_config_path = os.path.join(config_dir, "qwen_lora_dpo_mac_fixed.yaml")
    new_config_path = os.path.join(config_dir, f"qwen_dpo_{run_id}.yaml")
    
    # 复制基础配置文件
    shutil.copy2(base_config_path, new_config_path)
    
    # 读取配置文件内容
    with open(new_config_path, 'r') as file:
        config_content = file.read()
    
    # 替换配置中的值
    # 替换output_dir
    config_content = re.sub(
        r'output_dir:.*', 
        f'output_dir: output/qwen-0.5B/lora/dpo_{run_id}', 
        config_content
    )
    
    # 替换pref_beta
    config_content = re.sub(
        r'pref_beta:.*', 
        f'pref_beta: {beta_value}  # 已设置beta值', 
        config_content
    )
    
    # 替换run_name
    config_content = re.sub(
        r'run_name:.*', 
        f'run_name: {exp_name}_beta_{beta_value}', 
        config_content
    )
    
    # 写回配置文件
    with open(new_config_path, 'w') as file:
        file.write(config_content)
    
    # 检查wandb可用性并设置环境变量
    if shutil.which('wandb'):
        print("正在验证wandb登录状态...")
        # 检查wandb登录状态，但忽略输出
        try:
            subprocess.run(['wandb', 'status'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=True)
        except subprocess.CalledProcessError:
            print("请先登录Weights & Biases:")
            subprocess.run(['wandb', 'login'])
        
        # 设置wandb环境变量
        os.environ['WANDB_PROJECT'] = "qwen_dpo"
        os.environ['WANDB_NAME'] = f"{exp_name}_beta_{beta_value}"
    
    # 运行训练
    print("===============================================")
    print("开始DPO训练实验")
    print(f"实验名称: {exp_name}")
    print(f"Beta值: {beta_value}")
    print(f"配置文件: {new_config_path}")
    print(f"输出目录: output/qwen-0.5B/lora/dpo_{run_id}")
    print("===============================================")
    
    # 执行训练命令
    # 此处可以添加断点，在启动训练前进行调试
    # breakpoint()  # 取消注释此行以启用调试断点
    
    # 运行训练命令
    subprocess.run(['llamafactory-cli', 'train', new_config_path])
    
    # 输出训练完成提示
    print("===============================================")
    print(f"训练完成，结果保存在: output/qwen-0.5B/lora/dpo_{run_id}")
    if shutil.which('wandb'):
        print("可以在W&B界面查看训练进度和结果: https://wandb.ai")
    print("===============================================")

if __name__ == "__main__":
    main() 