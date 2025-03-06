#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
详细拆解的BetaDPO训练脚本 (BetaDPO Training Script with Detailed Components)
==============================================================

此脚本对LLaMA-Factory中的BetaDPO训练流程进行了详细拆解，主要用于调试和研究目的。
它将官方实现的各个组件和过程分离为独立函数，并添加了丰富的调试信息和注释。

设计目的:
--------
1. 调试工具 - 清晰展示BetaDPO训练的各个关键步骤和组件，便于调试
2. 组件解耦 - 将训练流程分解为多个独立函数，便于单独分析和修改
3. 原理探索 - 通过代码注释和调试输出帮助理解BetaDPO算法原理
4. 验证计算 - 人工确认动态beta计算逻辑正确性

主要功能模块:
-----------
- setup_logging: 设置日志记录格式
- print_section: 打印分隔区域标题
- load_config_file: 加载YAML配置文件
- process_args: 处理和转换参数
- prepare_tokenizer_and_model: 准备tokenizer和模型
- prepare_dataset: 准备数据集
- setup_trainer: 设置BetaDPO训练器
- run_betadpo_training: 执行训练过程
- create_callbacks: 创建训练回调函数
- run_betadpo_workflow: 整合所有步骤运行完整工作流

使用场景:
--------
- 调试BetaDPO特定训练过程中的问题
- 修改和扩展BetaDPO训练流程
- 分析困惑度和动态beta计算实现
- 验证可学习beta_scale参数的工作方式
"""

import os
import sys
import yaml
import logging
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn.functional as F

# 导入所需的LLaMA-Factory模块
from transformers import TrainerCallback

from llamafactory.hparams import (
    ModelArguments, 
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
    get_train_args
)
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
from llamafactory.train.betadpo.trainer import CustomBetaDPOTrainer  # 使用BetaDPO的训练器
from llamafactory.train.callbacks import SaveProcessorCallback
from llamafactory.train.betadpo.workflow import run_betadpo  # 导入BetaDPO工作流
from llamafactory.train.trainer_utils import create_ref_model
from llamafactory.extras.ploting import plot_loss
from llamafactory.extras.constants import IGNORE_INDEX


def setup_logging(output_dir=None, level=logging.INFO):
    """
    设置日志级别和格式
    
    Args:
        output_dir: 输出目录，如果提供，日志将保存到此目录下
        level: 日志级别
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format = f"%(asctime)s [%(levelname)s] %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if output_dir:
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"betadpo_training_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    
    if output_dir:
        logger.info(f"日志将保存到: {log_file}")
        
    return logger


def print_section(title):
    """打印分隔区域标题"""
    logger = logging.getLogger(__name__)
    separator = "=" * 80
    logger.info(f"\n{separator}\n{title}\n{separator}")


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def process_args(config_path: str) -> Tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]:
    """
    处理配置文件中的参数
    """
    logger = logging.getLogger(__name__)
    
    # 保存原始命令行参数
    original_argv = sys.argv.copy()
    
    try:
        # 设置命令行参数为配置文件路径
        sys.argv = [sys.argv[0], config_path]
        
        # 获取训练参数
        logger.info("处理训练参数...")
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
        
        # 检查并确保stage设置为betadpo
        if finetuning_args.stage != "betadpo":
            logger.warning(f"配置文件中stage={finetuning_args.stage}，但此脚本为BetaDPO，强制将stage设置为betadpo")
            finetuning_args.stage = "betadpo"
        
        # 恢复原始命令行参数
        sys.argv = original_argv
        
        return model_args, data_args, training_args, finetuning_args, generating_args
    
    except Exception as e:
        # 确保恢复原始命令行参数
        sys.argv = original_argv
        logger.error(f"处理参数时出错: {e}")
        raise


def prepare_tokenizer_and_model(model_args, finetuning_args, data_args, training_args, do_train=True):
    """
    准备tokenizer和模型
    """
    logger = logging.getLogger(__name__)
    print_section("准备Tokenizer和模型")
    
    # 加载tokenizer
    logger.info(f"从 {model_args.model_name_or_path} 加载tokenizer...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 获取模板并修复tokenizer
    logger.info(f"使用模板: {data_args.template}")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 加载模型
    logger.info(f"从 {model_args.model_name_or_path} 加载模型...")
    model = load_model(tokenizer, model_args, finetuning_args, do_train)
    
    # 创建参考模型（如果需要）
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):
            logger.info("使用模型自身作为参考模型")
            ref_model = model
        else:
            logger.info(f"创建参考模型，源: {finetuning_args.ref_model or model_args.model_name_or_path}")
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        logger.info("不使用参考模型")
        ref_model = None
    
    return tokenizer, tokenizer_module, template, model, ref_model


def prepare_dataset(template, model_args, data_args, training_args, tokenizer_module):
    """
    准备数据集
    """
    logger = logging.getLogger(__name__)
    print_section("准备数据集")
    
    logger.info(f"数据集: {data_args.dataset}")
    logger.info(f"最大样本数: {data_args.max_samples}")
    
    # 获取数据集
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="rm",  # BetaDPO与DPO一样使用rm数据
        **tokenizer_module
    )
    
    # 打印数据集信息
    if "train_dataset" in dataset_module:
        train_size = len(dataset_module["train_dataset"])
        logger.info(f"训练集大小: {train_size} 样本")
    
    if "eval_dataset" in dataset_module:
        eval_size = len(dataset_module["eval_dataset"])
        logger.info(f"验证集大小: {eval_size} 样本")
    
    # 随机抽取一个样本进行展示
    if "train_dataset" in dataset_module and len(dataset_module["train_dataset"]) > 0:
        random_idx = 0  # 显示第一个样本
        sample = dataset_module["train_dataset"][random_idx]
        logger.info(f"数据集样本示例 (索引 {random_idx}):")
        for k, v in sample.items():
            if k in ["prompt_ids", "chosen_ids", "rejected_ids"]:
                logger.info(f"  {k}: 长度={len(v)}")
            else:
                logger.info(f"  {k}: {v}")
    
    return dataset_module


def setup_trainer(
    model, 
    ref_model, 
    training_args, 
    finetuning_args, 
    data_collator, 
    dataset_module, 
    tokenizer_module,
    callbacks
):
    """
    设置BetaDPO训练器
    """
    logger = logging.getLogger(__name__)
    print_section("设置BetaDPO训练器")
    
    # 更新训练参数
    logger.info("设置训练参数...")
    training_args.remove_unused_columns = False  # 对于多模态和成对数据集很重要
    
    # 调试信息-BetaDPO超参数
    logger.info(f"BetaDPO超参数:")
    logger.info(f"  beta: {finetuning_args.pref_beta}")
    logger.info(f"  损失类型: {finetuning_args.pref_loss}")
    
    # 检查动态beta相关参数
    logger.info(f"  beta_scale 初始值: {finetuning_args.pref_beta_scale} (可学习参数)")
    
    # 初始化BetaDPO训练器
    logger.info("初始化CustomBetaDPOTrainer...")
    trainer = CustomBetaDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
    
    # 断点调试提示 - beta_scale初始化点
    logger.info("断点调试提示: 在trainer.py的__init__方法中设置断点，检查beta_scale初始化")
    logger.info("断点调试提示: 在trainer.py的create_optimizer方法中设置断点，检查beta_scale添加到优化器")
    
    return trainer


def debug_compute_preference_loss(trainer, batch):
    """
    调试compute_preference_loss函数的计算逻辑
    """
    logger = logging.getLogger(__name__)
    print_section("调试动态beta计算逻辑")
    
    # 断点调试提示
    logger.info("请在trainer.py的compute_preference_loss函数中设置断点")
    logger.info("特别关注perplexity计算、beta_scale应用和dynamic_beta计算过程")
    
    # 验证前向传播和动态beta计算
    logger.info("执行一次前向传播和损失计算以触发断点...")
    try:
        # 获取输入批次
        if batch is None and hasattr(trainer, "get_train_dataloader"):
            dataloader = trainer.get_train_dataloader()
            batch = next(iter(dataloader))
        
        if batch is not None:
            # 计算批次损失以触发断点
            with torch.no_grad():
                logger.info("正在计算损失，请在compute_preference_loss函数中观察动态beta计算...")
                loss = trainer.compute_loss(trainer.model, batch)
                logger.info(f"批次损失计算完成，损失值: {loss.item()}")
        else:
            logger.warning("未能获取有效批次进行调试")
    
    except Exception as e:
        logger.error(f"调试过程中出错: {e}", exc_info=True)


def run_betadpo_training(trainer, training_args, finetuning_args, dataset_module=None):
    """
    运行BetaDPO训练
    """
    logger = logging.getLogger(__name__)
    print_section("运行BetaDPO训练")
    
    # 首先执行调试逻辑
    debug_compute_preference_loss(trainer, None)
    
    train_result = None
    
    # 记录训练前beta_scale的值
    if hasattr(trainer, "beta_scale"):
        beta_scale_value = torch.clamp(trainer.beta_scale, min=1e-8).item()
        logger.info(f"训练前beta_scale值: {beta_scale_value}")
    
    if training_args.do_train:
        logger.info(f"开始训练...")
        logger.info(f"设备数量: {trainer.args.n_gpu}")
        logger.info(f"训练批量大小: {training_args.per_device_train_batch_size} * {training_args.gradient_accumulation_steps}")
        logger.info(f"训练轮次: {training_args.num_train_epochs}")
        
        # 断点调试提示
        logger.info("断点调试建议: 在trainer.py的每个epoch结束时记录beta_scale当前值")
        
        # 运行训练
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # 保存模型
        logger.info("保存模型...")
        trainer.save_model()
        
        # 计算和记录指标
        logger.info("记录训练指标...")
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        # 绘制损失曲线
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            logger.info("绘制损失曲线...")
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies", "train_beta_scale"])
    
    # 执行评估
    if training_args.do_eval:
        logger.info("开始评估...")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        # 如果参考模型就是模型本身，则无法计算奖励
        if hasattr(trainer, "ref_model") and id(trainer.model) == id(trainer.ref_model):
            logger.warning("参考模型是模型本身，无法计算奖励指标")
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        
        # 记录评估指标
        logger.info("记录评估指标...")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    return train_result


def create_callbacks(model_args, data_args, training_args, finetuning_args, generating_args):
    """
    创建训练回调函数
    """
    logger = logging.getLogger(__name__)
    print_section("创建回调函数")
    
    callbacks: List[TrainerCallback] = []
    
    if getattr(model_args, "processor", None):
        logger.info("添加SaveProcessorCallback")
        callbacks.append(
            SaveProcessorCallback(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
            )
        )
    
    # 可以添加一个自定义回调函数来监控beta_scale的变化
    class BetaScaleMonitorCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if "trainer" in kwargs and hasattr(kwargs["trainer"], "beta_scale"):
                trainer = kwargs["trainer"]
                beta_scale_value = F.softplus(trainer.beta_scale).item()
                logger.info(f"[Epoch {state.epoch}] beta_scale值: {beta_scale_value}")
    
    logger.info("添加BetaScaleMonitorCallback")
    callbacks.append(BetaScaleMonitorCallback())
    
    logger.info(f"总计 {len(callbacks)} 个回调函数")
    return callbacks if callbacks else None


def run_betadpo_workflow(config_path: str):
    """
    运行完整的BetaDPO工作流程
    """
    # 首先设置基本控制台日志
    temp_logger = setup_logging()
    temp_logger.info(f"开始BetaDPO工作流程，配置文件: {config_path}")
    
    try:
        # 1. 处理参数
        model_args, data_args, training_args, finetuning_args, generating_args = process_args(config_path)
        
        # 确保输出目录存在
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # 使用输出目录重新设置日志
        logger = setup_logging(output_dir=training_args.output_dir)
        logger.info(f"输出目录: {training_args.output_dir}")
        
        # 打印报告工具信息
        logger.info(f"  报告工具: {training_args.report_to}")
        
        # 也可以通过环境变量设置WANDB项目
        if not hasattr(training_args, 'wandb_project') or not training_args.wandb_project:
            os.environ["WANDB_PROJECT"] = "betadpo-logppl"
            logger.info("通过环境变量设置WANDB项目名称为: betadpo-logppl")
        
        # 2. 准备tokenizer和模型
        tokenizer, tokenizer_module, template, model, ref_model = prepare_tokenizer_and_model(
            model_args, finetuning_args, data_args, training_args, do_train=training_args.do_train
        )
        
        # 3. 准备数据集
        dataset_module = prepare_dataset(template, model_args, data_args, training_args, tokenizer_module)
        
        # 4. 创建数据整理器
        data_collator = PairwiseDataCollatorWithPadding(
            template=template,
            model=model,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )
        
        # 5. 创建回调函数
        callbacks = create_callbacks(model_args, data_args, training_args, finetuning_args, generating_args)
        
        # 6. 设置训练器
        trainer = setup_trainer(
            model=model,
            ref_model=ref_model,
            training_args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            dataset_module=dataset_module,
            tokenizer_module=tokenizer_module,
            callbacks=callbacks
        )
        
        # 7. 运行训练与评估
        run_betadpo_training(trainer, training_args, finetuning_args, dataset_module)
        
        # 8. 创建模型卡片和推送（如果需要）
        if hasattr(trainer, "create_modelcard_and_push"):
            trainer.create_modelcard_and_push(
                model_args, data_args, training_args, finetuning_args
            )
        
        logger.info("BetaDPO工作流程完成！")
        return 0
    
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}", exc_info=True)
        return 1


def main():
    """
    主函数
    """
    # 设置日志
    logger = setup_logging(level=logging.INFO)
    
    try:
        # 解析命令行参数
        if len(sys.argv) < 2:
            logger.warning("未提供配置文件路径，使用默认配置文件路径")
            config_path = "examples/train_lora/qwen1_5_0_5b_lora_betadpo.yaml"
        else:
            config_path = sys.argv[1]
        
        logger.info(f"使用配置文件: {config_path}")
        
        # 运行BetaDPO工作流程
        return run_betadpo_workflow(config_path)
    
    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        return 130
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    main() 


# 运行命令
# python dpo_baseline/run_betadpo_detailed.py examples/train_lora/qwen1_5_0_5b_lora_betadpo.yaml
