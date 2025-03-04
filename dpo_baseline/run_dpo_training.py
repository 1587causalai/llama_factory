#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA-Factory DPO训练脚本 (Streamlined DPO Training Script)
========================================================

此脚本提供了一个简洁高效的DPO训练实现，功能与LLaMA-Factory命令行接口完全等效，
但跳过了命令行解析部分，直接调用核心训练组件，并添加了额外的结果分析功能。

设计目的:
--------
1. 高效实现 - 提供一个简洁而完整的DPO训练脚本，减少不必要的中间层
2. 结果分析 - 增强对训练结果的统计和分析能力
3. 生产就绪 - 适合在生产环境中使用，易于集成到自动化流程

主要功能模块:
-----------
- setup_logging: 配置日志系统
- run_custom_training: 运行训练并提供详细的结果分析
- direct_run_dpo: 直接执行DPO训练流程，跳过CLI命令解析

与其他脚本的区别:
--------------
- 相比dpo_detailed.py: 本脚本更加简洁，减少了教学性质的代码，更适合生产使用
- 相比run_dpo_detailed.py: 本脚本减少了中间过程的详细输出，侧重于最终结果的分析

使用场景:
--------
- 在生产环境中运行DPO训练
- 需要简洁而可靠的训练脚本
- 关注训练结果分析而非过程细节

使用方法:
--------
python run_dpo_training.py [配置文件路径]

如果不提供配置文件路径，默认使用 'examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml'

关于Callbacks:
------------
Callbacks（回调函数）是一种钩子机制，允许在训练的特定阶段执行自定义代码，无需修改训练器核心逻辑。
它们可以监控训练状态、记录指标、可视化进度、提前停止训练等。本脚本使用的主要callbacks包括：

1. LogCallback: 负责记录训练日志和进度
2. PissaConvertCallback: 用于PISSA格式模型的转换
3. ReporterCallback: 报告训练配置和结果
4. SwanLabCallback: 与SwanLab可视化工具集成(可选)

这些callbacks会在训练的不同阶段被自动调用，如训练开始前、每步训练后、评估后等。
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any

# 导入LLaMA-Factory的关键组件
from llamafactory.train.tuner import run_exp, _training_function
from llamafactory.hparams import (
    get_train_args, 
    read_args,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments
)
from llamafactory.train.dpo import run_dpo
# 导入各种回调函数，用于在训练过程的不同阶段执行特定操作
from llamafactory.train.callbacks import LogCallback, ReporterCallback, PissaConvertCallback
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.logging import get_logger

# 设置日志
logger = get_logger(__name__)

def setup_logging(level=logging.INFO):
    """
    设置日志配置
    
    Args:
        level: 日志级别，默认为INFO
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    

def run_custom_training(args, callbacks):
    """
    运行训练流程并添加详细的训练统计信息打印
    
    这个函数是对LLaMA-Factory标准训练流程的包装，增加了训练结果的详细分析。
    
    Args:
        args: 解析后的训练参数
        callbacks: 回调函数列表，用于在训练的不同阶段执行特定操作
    
    回调函数的作用:
    -----------
    回调函数在训练的特定时刻被触发，例如：
    - on_init_end: 初始化完成时
    - on_train_begin: 训练开始时
    - on_step_end: 每步训练结束时
    - on_evaluate: 评估时
    - on_log: 记录日志时
    - on_train_end: 训练结束时
    
    通过这些回调，可以实现日志记录、进度可视化、提前停止等功能，
    而无需修改训练器的核心代码。
    """
    # 获取参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    # 使用LLaMA-Factory的标准训练流程
    # _training_function是LLaMA-Factory的核心训练函数，它会根据参数执行相应的训练流程
    # 传入的callbacks会在训练过程的不同阶段被调用
    result = _training_function(config={"args": args, "callbacks": callbacks})
    
    # 打印训练汇总
    logger.info("=" * 50)
    logger.info("DPO训练完成")
    logger.info("=" * 50)
    
    # 尝试获取训练的state对象
    # trainer_state.json包含了训练过程中的详细信息，如步数、损失、检查点等
    trainer_state_path = os.path.join(training_args.output_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        import json
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)
            if "global_step" in state:
                logger.info(f"总训练步数: {state['global_step']}")
            if "best_model_checkpoint" in state and state["best_model_checkpoint"]:
                logger.info(f"最佳检查点: {state['best_model_checkpoint']}")
            if "log_history" in state and len(state["log_history"]) > 0:
                last_log = state["log_history"][-1]
                if "loss" in last_log:
                    logger.info(f"最终损失: {last_log['loss']:.4f}")
                if "epoch" in last_log:
                    logger.info(f"总训练轮次: {last_log['epoch']:.2f}")
    
    # 打印模型和输出信息
    logger.info(f"模型保存于: {training_args.output_dir}")
    logger.info(f"使用LoRA微调，rank={finetuning_args.lora_rank}, alpha={getattr(finetuning_args, 'lora_alpha', 'N/A')}")
    
    return result

def direct_run_dpo(config_path: str):
    """
    直接执行DPO训练流程，跳过CLI命令解析
    
    这个函数是脚本的核心，负责设置训练环境、创建回调函数并执行训练。
    
    Args:
        config_path: DPO训练配置文件路径
    
    主要步骤:
    --------
    1. 读取配置文件
    2. 创建回调函数
    3. 执行训练
    """
    # 保存并修改命令行参数，以便read_args能正确读取配置文件
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], config_path]
    
    try:
        # 读取配置文件
        logger.info(f"加载配置文件: {config_path}")
        args = read_args()
        
        # 创建回调函数列表
        # 回调函数是训练过程中的钩子，可以在特定时刻执行特定操作
        callbacks = []
        
        # LogCallback: 负责记录训练过程中的日志和进度
        # 会在每个训练步骤后更新进度条，在训练结束时汇总信息
        callbacks.append(LogCallback())
        
        # 添加PISSA转换回调，如果启用
        # PissaConvertCallback: 用于在训练后将模型转换为PISSA格式
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
        if finetuning_args.pissa_convert:
            callbacks.append(PissaConvertCallback())
            
        # 添加SwanLab回调，如果启用
        # SwanLabCallback: 与SwanLab集成，用于可视化训练过程
        if hasattr(finetuning_args, 'use_swanlab') and finetuning_args.use_swanlab:
            from llamafactory.train.trainer_utils import get_swanlab_callback
            callbacks.append(get_swanlab_callback(finetuning_args))
            
        # 添加Reporter回调
        # ReporterCallback: 在训练开始和结束时报告训练配置和结果
        # 它会保存模型信息、训练参数，便于后续分析和复现
        callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))
        
        # 检查是否为DPO训练
        if finetuning_args.stage != "dpo":
            raise ValueError(f"配置文件指定的训练阶段不是DPO，而是: {finetuning_args.stage}")
        
        # 执行DPO训练
        logger.info("开始DPO训练流程")
        run_custom_training(args, callbacks)
        
    except Exception as e:
        logger.error(f"DPO训练过程中出错: {str(e)}")
        raise
    finally:
        # 恢复原始命令行参数
        sys.argv = original_argv


def main():
    """
    主函数，处理命令行参数并启动训练
    
    这是脚本的入口点，负责解析命令行参数、设置日志系统并启动训练。
    """
    setup_logging()
    
    # 默认配置文件路径
    default_config = "examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml"
    
    # 从命令行获取配置文件路径，如果有的话
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = default_config
        logger.info(f"未指定配置文件，使用默认配置: {default_config}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 执行DPO训练
    direct_run_dpo(config_path)


if __name__ == "__main__":
    main()


# 运行脚本
# python dpo_baseline/run_dpo_training.py examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml
