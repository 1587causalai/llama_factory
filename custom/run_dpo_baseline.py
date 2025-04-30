#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行标准的 DPO (Direct Preference Optimization) 基线算法

使用方法:
    python custom/run_dpo_baseline.py --config custom/dpo_baseline.yaml [--wandb_project PROJECT]
"""

import argparse
import os
import sys
import yaml
import torch
# import matplotlib.pyplot as plt # Removed as plotting is no longer needed for baseline
# import numpy as np # Removed as plotting is no longer needed for baseline
# from transformers.trainer_callback import TrainerCallback # Seems unused
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
from llamafactory.train.dpo.trainer import CustomDPOTrainer # Assuming this is the correct trainer entry point for LLaMA Factory DPO
from llamafactory.train.trainer_utils import create_ref_model
from llamafactory.train.callbacks import LogCallback
from llamafactory.extras.constants import IGNORE_INDEX

def run_dpo_baseline(config_path, wandb_project=None):
    """运行标准DPO基线算法"""
    if wandb_project:
        os.environ['WANDB_PROJECT'] = wandb_project
    
    print('=' * 60)
    print(f"正在加载配置: {config_path}")
    print('=' * 60)
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 将配置转换为命令行参数格式
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:  # 只在 True 时添加
                args.append(f"--{key}")
        elif isinstance(value, list):
            # Handle potential list arguments correctly
            for item in value:
                 args.append(f"--{key}={item}")
        elif value is not None:
            args.append(f"--{key}={value}")

    # 获取训练参数
    print('=' * 60)
    print("处理参数中...")
    print('=' * 60)
    # Note: get_train_args might internally handle list args differently, 
    # the above conversion assumes simple key=value pairs or flags. Review if complex list args are used.
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args) # generating_args seem unused
    
    # Removed checks for use_dynamic_beta and disco_pref as they are not part of baseline DPO

    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    # Setting dataloader_num_workers=0 might be a workaround, consider if it's necessary for stability
    training_args.dataloader_num_workers = 0  
    print(f"强制设置 remove_unused_columns = {training_args.remove_unused_columns}")
    print(f"强制设置 dataloader_num_workers = {training_args.dataloader_num_workers}")

    # 检查设备 (Standard device check is fine)
    device = None
    if torch.cuda.is_available():
        print("使用 CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("使用 MPS (Apple Silicon)")
        device = torch.device("mps")
    else:
        print("使用 CPU")
        device = torch.device("cpu")

    # 阶段1: 准备模型组件
    print('=' * 60)
    print("准备模型组件...")
    print('=' * 60)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # Ensure load_model doesn't implicitly add custom heads based on finetuning_args if they exist
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train) 
    ref_model = create_ref_model(model_args, finetuning_args) if finetuning_args.use_ref_model else None

    # 阶段2: 准备数据集
    print('=' * 60)
    print("准备数据集...")
    print('=' * 60)
    # Stage should be 'rm' for DPO dataset loading
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model, # Pass model or None? Check collator requirements
        pad_to_multiple_of=8, # Standard value
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )
    
    if "train_dataset" in dataset_module:
        print(f"训练集样本数: {len(dataset_module['train_dataset'])}")
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"]:
        print(f"验证集样本数: {len(dataset_module['eval_dataset'])}")
    
    # 阶段3: 设置训练器
    print('=' * 60)
    print("设置训练器...")
    print('=' * 60)
    callbacks = [LogCallback()]
    # Ensure CustomDPOTrainer behaves as standard DPO when no custom finetuning_args are active
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args, # Pass finetuning_args, trainer should ignore custom params if not set in config
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
    
    # 阶段4: 执行训练
    print('=' * 60)
    print("开始训练...")
    print('=' * 60)
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Removed beta head check and plotting logic

    print("保存模型...")
    try:
        trainer.save_model()
        trainer.save_state()
        print(f"模型已保存至: {training_args.output_dir}")
    except Exception as e:
        print(f"保存模型失败: {e}")
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

def main():
    parser = argparse.ArgumentParser(description='运行标准DPO基线算法')
    parser.add_argument('--config', type=str, default='custom/dpo_baseline.yaml', help='训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb项目名称')
    args = parser.parse_args()
    run_dpo_baseline(args.config, args.wandb_project) # Call the renamed function
    return 0

if __name__ == "__main__":
    sys.exit(main())


