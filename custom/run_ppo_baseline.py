#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行标准的 PPO (Proximal Policy Optimization) 基线训练

使用方法:
    python custom/run_ppo_baseline.py --config custom/ppo_baseline.yaml [--wandb_project PROJECT]   
"""

import warnings
# 忽略特定的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Trainer.tokenizer is now deprecated.*")

import argparse
import os
import sys
import yaml
import torch
from copy import deepcopy

from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from transformers import DataCollatorWithPadding
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
# 注意：导入 PPO 的 Trainer (修正导入路径)
from llamafactory.train.ppo.trainer import CustomPPOTrainer
from llamafactory.train.trainer_utils import create_ref_model
from llamafactory.train.callbacks import LogCallback
from llamafactory.extras.constants import IGNORE_INDEX

def run_ppo_baseline(config_path, wandb_project=None):
    """运行标准PPO基线训练"""
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
    generating_args_dict = {}
    # 特殊处理 generate 部分的参数
    if 'generate' in config:
        generating_args_dict = config.pop('generate') # 直接 pop 出来

    # 注意: reward_model 参数需要保留在 config 中传递给 get_train_args
    # reward_model_path_from_config = config.pop('reward_model', None) # <--- 移除这一行
    # if reward_model_path_from_config is None:
    #     print("错误：配置文件中缺少 reward_model 路径")
    #     sys.exit(1)

    for key, value in config.items():
        if isinstance(value, bool):
            if value:  
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
    # 此时 args 包含 reward_model, FinetuningArguments 验证会通过
    model_args, data_args, training_args, finetuning_args, gen_args_obj = get_train_args(args)
    
    # 合并从 YAML 文件解析的 generation 参数
    for key, value in generating_args_dict.items():
        setattr(gen_args_obj, key, value)
        
    # 确保 stage 是 ppo
    if finetuning_args.stage != "ppo":
        print(f"错误：配置文件中的 stage 应为 'ppo'，当前为 '{finetuning_args.stage}'")
        sys.exit(1)

    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    training_args.dataloader_num_workers = 0  
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
    print("准备模型组件 (Actor, Reward Model, Ref Model)...")
    print('=' * 60)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # --- 加载 Actor 模型 (model) --- 
    # model_args 已经配置好了 Actor 的基础模型和可能的 LoRA
    print(f"加载 Actor 模型自: {model_args.model_name_or_path}", end="")
    if model_args.adapter_name_or_path:
        print(f" 并应用 LoRA 适配器: {model_args.adapter_name_or_path}")
    else:
        print()
    # PPO Actor 需要 Value Head
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    print("Actor 模型加载完毕.")

    # --- 加载 RM 模型 (reward_model) --- 
    # 创建独立的 ModelArguments 用于加载 RM
    # 通过复制 Actor 的 model_args 来确保基础参数有效
    reward_model_args = deepcopy(model_args) # 使用 deepcopy 复制
    
    # 获取 RM 的 LoRA checkpoint 路径 (从 finetuning_args)
    reward_model_path = finetuning_args.reward_model
    if reward_model_path is None:
        print("错误：无法从参数中获取 reward_model 路径，请检查配置和 hparams 解析")
        sys.exit(1)
    # 移除之前添加的复杂路径检查和拼接逻辑
    # reward_model_checkpoint_path = os.path.join(reward_model_output_dir, "checkpoint-45") 
    # if not os.path.exists(os.path.join(reward_model_checkpoint_path, "adapter_config.json")):
    #    ...

    # 设置基础模型路径和 LoRA 适配器路径
    reward_model_args.model_name_or_path = model_args.model_name_or_path # 使用 Actor 的基础模型
    reward_model_args.adapter_name_or_path = [reward_model_path] # 直接使用配置传入的 checkpoint 路径
    
    print(f"加载奖励模型 (基座: {reward_model_args.model_name_or_path}, LoRA: {reward_model_path})...")
    # load_model 会加载基础模型，然后 init_adapter 会加载 LoRA adapter
    # init_adapter 之后，loader 会尝试从 adapter_name_or_path 加载 value_head.safetensors
    reward_model = load_model(tokenizer, reward_model_args, finetuning_args, is_trainable=False, add_valuehead=True)
    print("奖励模型加载完毕.")

    # --- 创建 Ref 模型 (ref_model) --- 
    # 使用 Actor 的 model_args 创建 Ref 模型
    # create_ref_model 内部会处理是否加载基础模型或带 LoRA 的模型
    print("创建参考模型...")
    ref_model = create_ref_model(model_args, finetuning_args)
    if ref_model is not None:
        print("参考模型已创建.")
    else:
        print("未创建参考模型 (或 finetuning_args.use_ref_model=False).")

    # 阶段2: 准备数据集 (PPO 使用 Prompt 数据集)
    print('=' * 60)
    print("准备 PPO 数据集...")
    print('=' * 60)
    # PPO 训练需要 Prompt 数据集，stage='ppo'
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    # PPO 使用标准的 DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8, 
    )
    
    if "train_dataset" in dataset_module:
        print(f"PPO 训练集 (Prompts) 样本数: {len(dataset_module['train_dataset'])}")
    # PPO trainer 通常不直接使用 eval_dataset

    # 阶段3: 设置训练器
    print('=' * 60)
    print("设置 PPO 训练器...")
    print('=' * 60)
    callbacks = [LogCallback()]
    # 使用 CustomPPOTrainer, 传入显式加载的 reward_model
    trainer = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=gen_args_obj, # 传递 Generation 参数对象
        callbacks=callbacks,
        model=model, # Actor 模型
        reward_model=reward_model, # 显式传递加载好的 RM 对象
        ref_model=ref_model, # 参考模型
        tokenizer=tokenizer,
        processor=processor,
        data_collator=data_collator,
        **dataset_module,
    )
    
    # 阶段4: 执行 PPO 训练
    print('=' * 60)
    print("开始 PPO 训练...")
    print('=' * 60)
    trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    print("保存模型...")
    try:
        trainer.save_model()
        trainer.save_state()
        print(f"模型已保存至: {training_args.output_dir}")
    except Exception as e:
        print(f"保存模型失败: {e}")
    
    # PPO 的 metrics 通常在训练过程中记录 (reward, kl 等)
    # train_result 可能不存在或格式不同
    # metrics = train_result.metrics
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)

def main():
    parser = argparse.ArgumentParser(description='运行标准PPO基线训练')
    parser.add_argument('--config', type=str, default='custom/ppo_baseline.yaml', help='训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb项目名称')
    args = parser.parse_args()
    run_ppo_baseline(args.config, args.wandb_project)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 