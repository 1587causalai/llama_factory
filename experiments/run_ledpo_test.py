#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA-Factory LEDPO训练脚本
=====================================================================

此脚本实现了与llamafactory-cli train examples/ledpo_test.yaml相同的功能，
但提供了更清晰的模块化结构和详细的执行过程，便于理解LEDPO训练流程。

使用方法:
--------
python run_ledpo_test.py [配置文件路径]

默认使用examples/ledpo_test.yaml作为配置文件
"""

import os
import sys
import logging
import yaml
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import torch

# 确保能够导入llamafactory模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入LLaMA-Factory的关键组件
from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    TrainingArguments, 
    FinetuningArguments,
    GeneratingArguments,
    get_train_args,
    read_args
)
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import (
    get_dataset, 
    get_template_and_fix_tokenizer, 
    PairwiseDataCollatorWithPadding
)
from llamafactory.train.ledpo.trainer import LEDPOTrainer
from llamafactory.train.trainer_utils import create_ref_model
from llamafactory.train.callbacks import LogCallback, ReporterCallback, SaveProcessorCallback
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras import logging
from llamafactory.extras.misc import calculate_tps
from llamafactory.extras.ploting import plot_loss

# 设置日志
logger = logging.get_logger(__name__)

#######################
# 阶段1: 配置加载和处理 #
#######################

def load_and_process_config(config_path: str) -> Tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]:
    """
    加载配置文件并处理参数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        处理后的参数对象元组
    """
    logger.info(f"阶段1: 加载配置文件 {config_path}")
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 将配置转换为命令行参数格式
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            # 修改布尔值处理逻辑，确保False值也能被传递
            args.append(f"--{key}={'true' if value else 'false'}")
        elif isinstance(value, list):
            args.append(f"--{key}={','.join(map(str, value))}")
        elif value is not None:
            args.append(f"--{key}={value}")
    
    # 打印参数列表，用于调试
    logger.info(f"处理后的参数列表: {args}")
    
    # 处理命令行参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    # 根据参数设置输出目录
    append_str = "_disco" if finetuning_args.use_disco else ""
    append_str += "_dynamic_beta" if finetuning_args.use_dynamic_beta else ""
    append_str += "_freeze_policy_model" if finetuning_args.freeze_policy_model else ""
    training_args.output_dir = training_args.output_dir + append_str
    
    # 打印关键参数，帮助调试
    logger.info(f"模型: {model_args.model_name_or_path}")
    logger.info(f"数据集: {data_args.dataset}")
    logger.info(f"训练阶段: {finetuning_args.stage}")
    logger.info(f"微调类型: {finetuning_args.finetuning_type}")
    logger.info(f"输出目录: {training_args.output_dir}")
    # 打印重要的FinetuningArguments
    logger.info(f"pref_beta: {finetuning_args.pref_beta}")
    logger.info(f"use_dynamic_beta: {finetuning_args.use_dynamic_beta}")
    logger.info(f"beta_min: {finetuning_args.beta_min}")
    logger.info(f"beta_max: {finetuning_args.beta_max}")
    
    return model_args, data_args, training_args, finetuning_args, generating_args

#########################
# 阶段2: 模型和分词器准备 #
#########################

def prepare_model_components(model_args, finetuning_args, data_args, training_args, do_train=True):
    """
    准备tokenizer、模板和模型
    
    Args:
        model_args: 模型参数
        finetuning_args: 微调参数
        data_args: 数据参数
        training_args: 训练参数
        do_train: 是否进行训练
        
    Returns:
        tokenizer、template、model及相关模块
    """
    logger.info("阶段2: 准备模型组件")
    
    # 1. 加载tokenizer
    logger.info("加载Tokenizer...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 2. 获取模板
    logger.info("获取模板...")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 3. 加载模型
    logger.info(f"加载模型: {model_args.model_name_or_path}...")
    model = load_model(tokenizer, model_args, finetuning_args, do_train)
    
    # 4. 创建参考模型
    logger.info("准备参考模型...")
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not do_train):
            logger.info("参考模型与主模型相同")
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        logger.info("未使用参考模型")
        ref_model = None
    
    # 打印模型信息摘要
    logger.info(f"Tokenizer类型: {tokenizer.__class__.__name__}")
    logger.info(f"模板: {data_args.template}")
    logger.info(f"模型类型: {model.__class__.__name__}")
    if ref_model is not None:
        ref_model_type = "同主模型" if id(model) == id(ref_model) else ref_model.__class__.__name__
    else:
        ref_model_type = "无"
    logger.info(f"参考模型: {ref_model_type}")
    
    return tokenizer, template, model, ref_model, tokenizer_module

##################
# 阶段3: 数据准备 #
##################

def prepare_training_data(template, model_args, data_args, training_args, tokenizer_module):
    """
    准备训练数据集
    
    Args:
        template: 模板
        model_args: 模型参数
        data_args: 数据参数
        training_args: 训练参数
        tokenizer_module: tokenizer模块
        
    Returns:
        处理后的数据集和数据整理器
    """
    logger.info("阶段3: 准备训练数据")
    
    # 获取数据集
    logger.info("加载数据集...")
    dataset_module = get_dataset(
        template, 
        model_args, 
        data_args, 
        training_args, 
        stage="rm",  # DPO使用RM阶段的数据处理逻辑
        **tokenizer_module
    )
    
    # 打印数据集信息
    if "train_dataset" in dataset_module:
        train_size = len(dataset_module["train_dataset"])
        logger.info(f"训练集样本数: {train_size}")
        
        # 打印几个样本的键，以便调试
        if train_size > 0:
            logger.info(f"训练集样本的键: {list(dataset_module['train_dataset'][0].keys())}")
        
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"] is not None:
        eval_size = len(dataset_module["eval_dataset"])
        logger.info(f"验证集样本数: {eval_size}")
    
    # 创建数据整理器
    tokenizer = tokenizer_module["tokenizer"]
    model = dataset_module.get("model", None)
    
    logger.info("创建数据整理器...")
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )
    
    return dataset_module, data_collator

####################
# 阶段4: 训练器设置 #
####################

def setup_dpo_trainer(
    model, 
    ref_model, 
    training_args, 
    finetuning_args, 
    data_collator, 
    dataset_module, 
    tokenizer_module,
    model_args,
    data_args,
    generating_args
):
    """
    设置DPO训练器
    
    Args:
        model: 模型
        ref_model: 参考模型
        training_args: 训练参数
        finetuning_args: 微调参数
        data_collator: 数据整理器
        dataset_module: 数据集模块
        tokenizer_module: tokenizer模块
        model_args: 模型参数
        data_args: 数据参数
        generating_args: 生成参数
        
    Returns:
        DPO训练器
    """
    logger.info("阶段4: 设置训练器")
    
    # 强制设置remove_unused_columns=False，确保它不会被覆盖
    training_args.remove_unused_columns = False
    logger.info(f"已强制设置 remove_unused_columns = {training_args.remove_unused_columns}")
    
    # 创建回调函数
    callbacks = []
    callbacks.append(LogCallback())
    
    if finetuning_args.pissa_convert:
        callbacks.append(SaveProcessorCallback())
    
    # 添加SwanLab回调，如果启用
    # if hasattr(finetuning_args, 'use_swanlab') and finetuning_args.use_swanlab:
    #     from llamafactory.train.trainer_utils import get_swanlab_callback
    #     callbacks.append(get_swanlab_callback(finetuning_args))
    
    # 添加Reporter回调
    callbacks.append(ReporterCallback(
        model_args=model_args,
        data_args=data_args,
        finetuning_args=finetuning_args, 
        generating_args=generating_args
    ))
    
    # 初始化DPO训练器
    logger.info("初始化LEDPO训练器...")
    trainer = LEDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
    
    logger.info(f"训练设备: {training_args.device}")
    
    # 显示优化器参数
    if hasattr(trainer, "optimizer") and trainer.optimizer:
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            logger.info(f"参数组 {i}: {len(param_group['params'])} 个参数, 学习率: {param_group['lr']}")
    
    return trainer

##################
# 阶段5: 执行训练 #
##################

def run_training(trainer, training_args, finetuning_args, dataset_module=None):
    """
    执行DPO训练
    
    Args:
        trainer: 训练器
        training_args: 训练参数
        finetuning_args: 微调参数
        dataset_module: 数据集模块
    """
    logger.info("阶段5: 执行训练")
    
    # 训练
    if training_args.do_train:
        logger.info("开始训练...")
        
        # 执行训练
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        logger.info("保存模型...")
        trainer.save_model()
        trainer.save_state()
        
        # 记录指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # 添加有效token处理速度指标
        if hasattr(finetuning_args, 'include_effective_tokens_per_second') and finetuning_args.include_effective_tokens_per_second and dataset_module and "train_dataset" in dataset_module:
            metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], metrics, stage="rm"
            )
            logger.info(f"有效token处理速度: {metrics['effective_tokens_per_sec']:.2f} tokens/sec")
        
        # 绘制损失曲线
        if finetuning_args.plot_loss:
            logger.info("绘制损失曲线...")
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])
    
    # 评估
    if training_args.do_eval:
        logger.info("开始评估...")
        
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        # 如果参考模型就是模型本身，无法计算奖励指标
        model = trainer.model
        ref_model = trainer.ref_model
        if id(model) == id(ref_model):
            logger.warning("警告: 参考模型与主模型相同，无法计算奖励指标")
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")

####################
# 完整的训练工作流程 #
####################

def run_ledpo_workflow(config_path: str):
    """
    运行LEDPO训练流程的主函数
    
    Args:
        config_path: 配置文件路径
    """
    try:
        logger.info("启动LEDPO训练工作流程")
        
        # 阶段1: 加载配置
        model_args, data_args, training_args, finetuning_args, generating_args = load_and_process_config(config_path)
        
        # 直接从YAML设置关键参数 - 解决布尔值参数解析问题
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
            # 检查并直接设置关键参数
            if 'use_dynamic_beta' in config:
                finetuning_args.use_dynamic_beta = config['use_dynamic_beta']
                logger.info(f"直接设置 use_dynamic_beta = {finetuning_args.use_dynamic_beta}")
            
            if 'beta_min' in config:
                finetuning_args.beta_min = float(config['beta_min'])
                logger.info(f"直接设置 beta_min = {finetuning_args.beta_min}")
                
            if 'beta_max' in config:
                finetuning_args.beta_max = float(config['beta_max'])
                logger.info(f"直接设置 beta_max = {finetuning_args.beta_max}")
        
        # 确保remove_unused_columns=False - 这对于DPO/LEDPO训练非常重要
        training_args.remove_unused_columns = False
        logger.info(f"设置 remove_unused_columns = {training_args.remove_unused_columns}")
        
        # 检查是否为LEDPO训练
        if finetuning_args.stage != "ledpo":
            raise ValueError(f"配置文件指定的训练阶段不是LEDPO，而是: {finetuning_args.stage}")
        

        # 确保输出目录存在
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # 阶段2: 准备模型组件
        tokenizer, template, model, ref_model, tokenizer_module = prepare_model_components(
            model_args, finetuning_args, data_args, training_args, training_args.do_train
        )
        
        # 阶段3: 准备数据集
        dataset_module, data_collator = prepare_training_data(
            template, model_args, data_args, training_args, tokenizer_module
        )
        
        # 检查数据集结构
        if "train_dataset" in dataset_module:
            logger.info("检查训练集结构...")
            if len(dataset_module["train_dataset"]) > 0:
                sample = dataset_module["train_dataset"][0]
                logger.info(f"样本键: {list(sample.keys())}")
                for key in sample.keys():
                    if isinstance(sample[key], torch.Tensor):
                        logger.info(f"  {key}: shape={sample[key].shape}, dtype={sample[key].dtype}")
                    elif isinstance(sample[key], list):
                        logger.info(f"  {key}: 列表长度={len(sample[key])}")
                    else:
                        logger.info(f"  {key}: 类型={type(sample[key])}")
        
        # 阶段4: 设置训练器
        trainer = setup_dpo_trainer(
            model, ref_model, training_args, finetuning_args, 
            data_collator, dataset_module, tokenizer_module,
            model_args=model_args,
            data_args=data_args,
            generating_args=generating_args
        )
        
        # 阶段5: 执行训练
        run_training(trainer, training_args, finetuning_args, dataset_module)
        
        logger.info("LEDPO训练流程全部完成！")
        
    except Exception as e:
        logger.error(f"LEDPO训练过程中出错: {str(e)}")
        # 打印完整的错误堆栈
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    """主函数，处理命令行参数并启动训练"""
    logger.info("LLaMA-Factory LEDPO训练工具")
    
    # 从命令行获取配置文件路径
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "experiments/ledpo_test.yaml"
        logger.info(f"未指定配置文件，使用默认配置: {config_path}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        logger.error(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 激活conda环境（需要时取消注释）
    # import subprocess
    # subprocess.run(["conda", "activate", "llama"], shell=True)
    
    # 执行LEDPO训练
    run_ledpo_workflow(config_path)

if __name__ == "__main__":
    main() 


# 运行命令
# python experiments/run_ledpo_test.py experiments/ledpo_test.yaml