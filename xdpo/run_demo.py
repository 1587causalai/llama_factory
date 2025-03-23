#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接运行定制的 DPO 变体算法

使用方法:
    python xdpo/run_demo.py --config xdpo/demo.yaml [--wandb_project PROJECT]
"""

import argparse
import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers.trainer_callback import TrainerCallback
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
from llamafactory.train.dpo.trainer import CustomDPOTrainer
from llamafactory.train.trainer_utils import create_ref_model
from llamafactory.train.callbacks import LogCallback
from llamafactory.extras.constants import IGNORE_INDEX

def run_with_dpo_variants(config_path, wandb_project=None):
    """运行DPO变体算法"""
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
            args.append(f"--{key}={','.join(map(str, value))}")
        elif value is not None:
            args.append(f"--{key}={value}")
    
    # 获取训练参数
    print('=' * 60)
    print("处理参数中...")
    print('=' * 60)
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # 确认自定义参数
    # if finetuning_args.use_dynamic_beta:
    #     print("使用动态beta")
    # if finetuning_args.disco_pref:
    #     print("使用disco偏好概率")  
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    training_args.dataloader_num_workers = 0  # 禁用多进程
    print(f"强制设置 remove_unused_columns = {training_args.remove_unused_columns}")
    print(f"强制设置 dataloader_num_workers = {training_args.dataloader_num_workers}")

    # 检查设备
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
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    ref_model = create_ref_model(model_args, finetuning_args) if finetuning_args.use_ref_model else None

    # 阶段2: 准备数据集
    print('=' * 60)
    print("准备数据集...")
    print('=' * 60)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
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
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
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
    
    if hasattr(trainer, "beta_head"):
        print("训练后beta_head参数指标:")
        # 计算beta_pos_delta 和 beta_neg_delta 的平均值
        beta_pos_delta_avg = np.mean(trainer.beta_pos_delta)
        beta_neg_delta_avg = np.mean(trainer.beta_neg_delta)
        print(f"beta_pos_delta 平均值: {beta_pos_delta_avg}")
        print(f"beta_neg_delta 平均值: {beta_neg_delta_avg}")
    
        # plot beta_pos_delta 和 beta_neg_delta
        plt.figure(figsize=(10, 5))
        plt.plot(trainer.beta_pos_delta, label='beta_pos_delta')
        plt.plot(trainer.beta_neg_delta, label='beta_neg_delta')
        plt.legend()
        plt.savefig(os.path.join(training_args.output_dir, "beta_delta.png"))
        plt.close()
        print(f"beta_delta.png 已保存至: {training_args.output_dir}")

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
    parser = argparse.ArgumentParser(description='运行DPO变体算法')
    parser.add_argument('--config', type=str, default='xdpo/demo.yaml', help='训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb项目名称')
    args = parser.parse_args()
    run_with_dpo_variants(args.config, args.wandb_project)
    return 0

if __name__ == "__main__":
    sys.exit(main())


