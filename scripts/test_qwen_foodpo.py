#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# ===========================================================================================
# Qwen1.5-0.5B + fooDPO 算法 测试脚本 (Python版)
# ===========================================================================================
#
# 【脚本说明】
# 此脚本用于使用fooDPO算法在小数据集上训练Qwen1.5-0.5B模型
# 适用于macOS MPS设备(Apple Silicon)，数据集很小的情况
#
# 【实验目的】
# 这个脚本能够测试不同动态beta缩放系数对训练结果的影响
# 每次运行会生成独立的输出目录和wandb记录，便于比较分析结果
#
# 【小数据集适应性调整】
# 1. 减小batch_size和gradient_accumulation_steps，避免过度更新
# 2. 提高learning_rate以加速小数据集上的收敛
# 3. 减少训练epochs，防止过拟合
# 4. 增加 logging_steps 和 eval_steps 频率，便于监控训练进度
# 5. 设置max_samples限制数据集大小用于快速测试
# 6. 禁用group_by_length，确保所有样本都被使用
# 7. 设置dataloader_num_workers=0解决MPS设备多进程问题
#
# 【内存优化措施】
# 1. 减小MAX_SEQ_LEN以降低内存占用
# 2. 启用梯度检查点(gradient_checkpointing)以减少显存使用
# 3. 减小lora_rank参数减少额外参数量
# 4. 使用8位优化器减少优化器状态内存占用
# 5. 设置环境变量PYTORCH_MPS_HIGH_WATERMARK_RATIO控制MPS内存使用
#
# 【动态beta设置】- 重要参数
# FooDPO使用基于困惑度的动态beta策略: β(x) = c · log(PPL(x)) · β
# 其中:
# - PPL(x)是模型对输入提示x的困惑度
# - c是通过pref_beta_scale参数控制的缩放系数
# - β是基础beta值(通过pref_beta参数设置)
#
# 【pref_beta_scale参数推荐值】
# - 0.1-0.3: 困惑度对beta影响较小，训练更接近标准DPO
# - 0.4-0.7: 推荐范围，提供适度的动态调整
# - 0.8-1.0: 困惑度影响较大，可能导致训练不稳定
#
# 【wandb配置】
# 使用Weights & Biases进行训练监控
# 运行前确保已经登录: wandb login
#
# 【注意事项】
# - 对于非常小的数据集(<100样本)，可能需要更高的learning_rate和更少的epochs
# - 监控训练过程中的损失曲线，如果出现震荡或无法收敛，考虑调整learning_rate
# - 小数据集训练时wandb可能只有少量数据点，这是正常现象
# ===========================================================================================
"""

import os
import shutil
import argparse
import sys
from pathlib import Path
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入LLaMA Factory的相关模块
try:
    # 尝试导入需要的模块
    from src.llamafactory.config import DataArguments
    from src.llamafactory.config import FinetuningArguments
    from src.llamafactory.config import GeneratingArguments
    from src.llamafactory.config import ModelArguments
    from src.llamafactory.config import TrainingArguments
    from src.llamafactory.train.foodpo import run_foodpo
    
    # 如果导入成功，设置标志
    USE_API = True
except ImportError:
    print("无法导入LLaMA Factory Python API，将使用命令行方式")
    USE_API = False

# 设置环境变量
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_foodpo_training(beta_scale=0.05):
    """
    使用fooDPO算法训练Qwen1.5-0.5B模型
    
    Args:
        beta_scale (float): 动态beta缩放因子，范围[0.01-1.0]
    """
    # 设置wandb配置
    os.environ["WANDB_PROJECT"] = "qwen-foodpo"
    os.environ["WANDB_NAME"] = f"qwen1.5-0.5B-foodpo-scale-{beta_scale}"
    
    # 基本配置参数
    model_path = os.path.expanduser("~/models/Qwen1.5-0.5B")
    dataset_path = "data"
    output_dir = f"output/qwen_foodpo_scale_{beta_scale}"
    
    # 清理上次运行的输出目录（如果需要）
    if os.path.exists(output_dir):
        print(f"清理旧输出目录: {output_dir}")
        shutil.rmtree(output_dir)
    
    print("=" * 56)
    print(f"开始运行FooDPO训练实验，动态beta缩放系数: {beta_scale}")
    print(f"输出目录: {output_dir}")
    print(f"WandB实验: {os.environ['WANDB_NAME']}")
    print("=" * 56)
    
    if USE_API:
        print("使用LLaMA Factory Python API进行训练")
        # 使用API方式训练
        try:
            # 构建命令行参数列表 - 这里我们将直接构建Python对象而不是命令行参数
            model_args = ModelArguments(
                model_name_or_path=model_path
            )
            
            data_args = DataArguments(
                dataset="dpo_zh_demo",
                dataset_dir=dataset_path,
                eval_dataset="dpo_zh_demo",
                cutoff_len=512,
                max_samples=10,
                template="default"
            )
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
                num_train_epochs=1,
                warmup_ratio=0.1,
                logging_steps=1,
                save_steps=5,
                save_total_limit=1,
                do_train=True,
                report_to=["wandb"],
                ddp_find_unused_parameters=False,
                evaluation_strategy="steps",  # 注意：这里修正了参数名
                eval_steps=5,
                load_best_model_at_end=True,
                metric_for_best_model="eval_rewards/margins",
                greater_is_better=True,
                group_by_length=False,
                dataloader_num_workers=0,
                log_level="info",
                disable_tqdm=False,
                remove_unused_columns=False,
                optim="adamw_torch",
                gradient_checkpointing=True,
                fp16=False
            )
            
            if torch.backends.mps.is_available():
                training_args.use_mps_device = True
                
            finetuning_args = FinetuningArguments(
                finetuning_type="lora",
                lora_rank=4,
                lora_alpha=16,
                lora_dropout=0.05,
                stage="foodpo", 
                pref_beta=0.1,
                pref_beta_scale=beta_scale,
                plot_loss=True
            )
            
            # 这里我们直接使用LLaMA Factory的API
            # 创建一个断点，以便调试时停下来
            print("这里设置了一个断点，用于调试，请确认它是否生效")  # <-- 这里可以设置断点
            
            # 准备参数 - 在这里调用run_foodpo时可以设置断点来调试
            generating_args = GeneratingArguments()
            parsed_args = (model_args, data_args, training_args, finetuning_args, generating_args)
            
            # 直接调用LLaMA Factory的run_foodpo函数而不是通过subprocess
            result = run_foodpo(*parsed_args)
            
            print(f"训练完成！结果保存在目录: {output_dir}")
            print(f"请在WandB界面中查看实验: {os.environ['WANDB_NAME']}")
            
            # 返回训练结果
            return result
        
        except Exception as e:
            print(f"使用API方式时出现错误: {e}")
            print("回退到命令行方式...")
            USE_API = False
    
    if not USE_API:
        print("使用命令行方式进行训练")
        # 回退到命令行方式
        import subprocess
        
        cmd = [
            "llamafactory-cli", "train",
            "--model_name_or_path", model_path,
            "--dataset", "dpo_zh_demo",
            "--dataset_dir", dataset_path,
            "--eval_dataset", "dpo_zh_demo",
            "--output_dir", output_dir,
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--learning_rate", "1e-4",
            "--num_train_epochs", "1",
            "--cutoff_len", "512",
            "--warmup_ratio", "0.1",
            "--max_samples", "10",
            "--logging_steps", "1",
            "--save_steps", "5",
            "--save_total_limit", "1",
            "--do_train", "true",
            "--template", "default",
            "--finetuning_type", "lora",
            "--lora_rank", "4",
            "--lora_alpha", "16",
            "--lora_dropout", "0.05",
            "--stage", "foodpo",
            "--pref_beta", "0.1",
            "--pref_beta_scale", str(beta_scale),
            "--fp16", "false",
            "--use_mps_device", "true",
            "--plot_loss", "true",
            "--report_to", "wandb",
            "--ddp_find_unused_parameters", "false",
            "--evaluation_strategy", "steps",
            "--eval_steps", "5",
            "--load_best_model_at_end", "true",
            "--metric_for_best_model", "eval_rewards/margins",
            "--greater_is_better", "true",
            "--group_by_length", "false",
            "--dataloader_num_workers", "0",
            "--log_level", "info",
            "--disable_tqdm", "false",
            "--remove_unused_columns", "false",
            "--optim", "adamw_torch",
            "--gradient_checkpointing", "true"
        ]
        
        # 打印命令行以便调试
        print("执行命令:")
        print(" ".join(cmd))
        
        # 运行命令
        process = subprocess.run(cmd, check=True)
        
        print(f"命令行方式训练完成！结果保存在目录: {output_dir}")
        print(f"请在WandB界面中查看实验: {os.environ['WANDB_NAME']}")
        
        return process.returncode


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Qwen1.5-0.5B + fooDPO 算法训练脚本")
    parser.add_argument(
        "--beta_scale", 
        type=float, 
        default=0.05, 
        help="动态beta缩放因子，范围[0.01-1.0]，默认值为0.05"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行训练 - 这里可以设置断点
    run_foodpo_training(beta_scale=args.beta_scale) 