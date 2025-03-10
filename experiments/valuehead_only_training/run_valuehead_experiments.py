#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import subprocess
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("ValueHead实验")

# 定义学习率倍率组合
LR_MULTIPLIERS = [10.0, 100.0, 1000.0, 5000.0]  # 不同的ValueHead学习率倍率

def run_experiment(config_path, lr_multiplier, use_conda="llama"):
    """
    运行单个实验
    
    Args:
        config_path: 配置文件路径
        lr_multiplier: ValueHead学习率倍率
        use_conda: 使用的conda环境
    
    Returns:
        bool: 实验是否成功运行
    """
    logger.info(f"使用配置: {config_path}, ValueHead学习率倍率: {lr_multiplier}")
    
    # 构建命令
    cmd = [
        "conda", "run", "-n", use_conda,
        "python", "src/llamafactory/train.py",
        "--config_file", config_path,
        "--value_head_lr_multiplier", str(lr_multiplier),
        "--wandb_name", f"qwen1_5_0_5b_valuehead_lr{lr_multiplier}"
    ]
    
    # 执行命令
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"实验成功完成: ValueHead学习率倍率 = {lr_multiplier}")
        logger.debug(f"输出:\n{process.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"实验运行失败: ValueHead学习率倍率 = {lr_multiplier}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行ValueHead训练实验")
    parser.add_argument("--conda", type=str, default="llama", help="使用的conda环境名称")
    parser.add_argument("--multipliers", type=float, nargs="+", help="指定要运行的学习率倍率")
    parser.add_argument("--wandb-project", type=str, default="qwen1_5_ledpo_valuehead_only", help="Wandb项目名称")
    args = parser.parse_args()
    
    # 设置wandb项目
    os.environ["WANDB_PROJECT"] = args.wandb_project
    logger.info(f"设置WANDB_PROJECT={args.wandb_project}")
    
    # 确定要运行的学习率倍率
    multipliers = args.multipliers if args.multipliers else LR_MULTIPLIERS
    logger.info(f"将运行以下ValueHead学习率倍率实验: {multipliers}")
    
    # 记录实验开始时间
    start_time = datetime.now()
    logger.info(f"实验开始时间: {start_time}")
    
    # 基础配置文件
    base_config = "experiments/valuehead_only_training/configs/qwen1_5_0_5b_valuehead_only.yaml"
    
    # 依次运行不同学习率倍率的实验
    results = {}
    for multiplier in multipliers:
        logger.info(f"========== 开始运行学习率倍率为 {multiplier} 的ValueHead训练实验 ==========")
        success = run_experiment(base_config, multiplier, use_conda=args.conda)
        results[multiplier] = success
    
    # 总结实验结果
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"实验结束时间: {end_time}")
    logger.info(f"总运行时长: {duration}")
    
    # 打印实验结果摘要
    logger.info("实验结果摘要:")
    for multiplier, success in results.items():
        status = "成功" if success else "失败"
        logger.info(f"  - 学习率倍率 {multiplier}: {status}")

if __name__ == "__main__":
    main() 