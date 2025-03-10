#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ValueHead/Policy 分离训练实验脚本
运行三个对比实验：
1. 只训练 ValueHead
2. 只训练 Policy
3. 同时训练两者（对照组）
"""

import os
import sys
import time
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(Path(__file__).parent, "experiment.log"))
    ]
)
logger = logging.getLogger("separate_training_experiment")

# 实验配置文件列表
EXPERIMENTS = [
    {
        "name": "只训练 ValueHead",
        "config": "experiments/valuehead_only_training/qwen1_5_0_5b_valuehead_only.yaml",
        "description": "冻结 Policy 部分，只训练 ValueHead 网络"
    },
    {
        "name": "只训练 Policy",
        "config": "experiments/valuehead_only_training/qwen1_5_0_5b_policy_only.yaml",
        "description": "冻结 ValueHead 网络，只训练 Policy 部分"
    },
    {
        "name": "联合训练",
        "config": "experiments/valuehead_only_training/qwen1_5_0_5b_normal_training.yaml",
        "description": "同时训练 Policy 和 ValueHead（对照组）"
    }
]

def run_experiment(config_path, use_conda="llama", use_deepspeed=True, cuda_devices="0"):
    """运行单个实验"""
    logger.info(f"开始运行实验: {config_path}")
    
    # 准备命令
    cmd = ["python", "src/train_bash.py", "--config", config_path]
    
    # 添加DeepSpeed支持
    if use_deepspeed:
        cmd.extend(["--deepspeed", "deepspeed/zero2.json"])
    
    # 运行命令
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        env = os.environ.copy()
        env["CONDA_DEFAULT_ENV"] = use_conda  # 设置conda环境
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices  # 设置CUDA设备
        
        process = subprocess.run(
            cmd, 
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.info(f"实验成功完成: {config_path}")
        return True, None
    except subprocess.CalledProcessError as e:
        logger.error(f"实验运行失败: {config_path}")
        logger.error(f"错误输出: {e.stderr}")
        return False, e.stderr
    except KeyboardInterrupt:
        logger.warning(f"实验被用户中断: {config_path}")
        return False, "用户中断"

def main():
    parser = argparse.ArgumentParser(description="运行 ValueHead/Policy 分离训练实验")
    parser.add_argument("--conda", type=str, default="llama", help="使用的conda环境名称")
    parser.add_argument("--cuda", type=str, default="0", help="使用的CUDA设备")
    parser.add_argument("--no-deepspeed", action="store_true", help="不使用DeepSpeed")
    parser.add_argument("--exp-id", type=int, nargs="+", help="指定要运行的实验ID（0-2）")
    args = parser.parse_args()
    
    # 记录实验开始时间
    start_time = datetime.now()
    logger.info(f"实验开始时间: {start_time}")
    logger.info(f"使用CUDA设备: {args.cuda}")
    logger.info(f"使用conda环境: {args.conda}")
    
    # 确定要运行的实验
    if args.exp_id:
        try:
            selected_experiments = [EXPERIMENTS[i] for i in args.exp_id]
            logger.info(f"将运行以下实验: {[exp['name'] for exp in selected_experiments]}")
        except IndexError:
            logger.error(f"无效的实验ID: {args.exp_id}，有效范围为0-2")
            return
    else:
        selected_experiments = EXPERIMENTS
        logger.info("将运行所有实验")
    
    # 依次运行实验
    results = {}
    for i, experiment in enumerate(selected_experiments):
        exp_name = experiment["name"]
        config_path = experiment["config"]
        description = experiment["description"]
        
        logger.info(f"\n========== 实验 {i+1}/{len(selected_experiments)}: {exp_name} ==========")
        logger.info(f"描述: {description}")
        logger.info(f"配置: {config_path}")
        
        # 添加一个小延迟，确保上一个实验的资源完全释放
        if i > 0:
            logger.info("等待5秒钟...")
            time.sleep(5)
        
        success, error = run_experiment(
            config_path, 
            use_conda=args.conda, 
            use_deepspeed=not args.no_deepspeed,
            cuda_devices=args.cuda
        )
        results[exp_name] = {"success": success, "error": error}
    
    # 总结实验结果
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"\n实验结束时间: {end_time}")
    logger.info(f"总运行时长: {duration}")
    
    # 打印实验结果摘要
    logger.info("\n========== ValueHead/Policy 分离训练实验结果摘要 ==========")
    for exp_name, result in results.items():
        status = "成功" if result["success"] else "失败"
        logger.info(f"{exp_name}: {status}")
        if not result["success"] and result["error"] and result["error"] != "用户中断":
            logger.info(f"  错误原因: {result['error'][:100]}..." if len(result["error"]) > 100 else result["error"])
    
    # 提示分析结果
    logger.info("\n所有实验完成！")
    logger.info("可以使用以下命令分析实验结果:")
    logger.info("python experiments/valuehead_only_training/analyze_results.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n程序被用户中断")
        logger.info("可以用 --exp-id 参数指定运行特定实验")
    except Exception as e:
        logger.exception("程序异常退出") 