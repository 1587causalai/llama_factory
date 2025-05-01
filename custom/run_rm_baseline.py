#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行标准的 RM (Reward Model) 基线训练

使用方法:
    python custom/run_rm_baseline.py --config custom/rm_baseline.yaml [--wandb_project PROJECT]
"""

import argparse
import os
import sys
import yaml
import torch

from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
# 注意：导入 RM 的 Trainer
from llamafactory.train.rm.trainer import PairwiseTrainer 
from llamafactory.train.callbacks import LogCallback
from llamafactory.extras.constants import IGNORE_INDEX

def run_rm_baseline(config_path, wandb_project=None):
    """运行标准RM基线训练"""
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
            for item in value:
                 args.append(f"--{key}={item}")
        elif value is not None:
            args.append(f"--{key}={value}")

    # 获取训练参数
    print('=' * 60)
    print("处理参数中...")
    print('=' * 60)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
    
    # 确保 stage 是 rm
    if finetuning_args.stage != "rm":
        print(f"错误：配置文件中的 stage 应为 'rm'，当前为 '{finetuning_args.stage}'")
        sys.exit(1)

    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    training_args.dataloader_num_workers = 0  # 与 DPO 脚本保持一致
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
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # 加载模型，注意 RM 训练时会自动添加 ValueHead (由 loader.py 和 trainer 控制)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True) 
    # RM 训练不需要 ref_model

    # 阶段2: 准备数据集
    print('=' * 60)
    print("准备数据集...")
    print('=' * 60)
    # RM 训练需要成对数据，stage='rm'
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model, # Collator 可能需要 model
        pad_to_multiple_of=8, 
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
    # 使用 PairwiseTrainer
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args, 
        data_collator=data_collator,
        callbacks=callbacks,
        processor=processor, # 传递 processor
        **dataset_module,
        tokenizer=tokenizer, # Trainer 需要 tokenizer
    )
    
    # 阶段4: 执行训练
    print('=' * 60)
    print("开始训练...")
    print('=' * 60)
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
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
    parser = argparse.ArgumentParser(description='运行标准RM基线训练')
    parser.add_argument('--config', type=str, default='custom/rm_baseline.yaml', help='训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb项目名称')
    args = parser.parse_args()
    run_rm_baseline(args.config, args.wandb_project)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 