#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
详细拆解的LEDPO训练脚本 (LEDPO Training Script with Detailed Components)
==============================================================

此脚本对LEDPO(Learnable Beta DPO)训练流程进行了详细拆解，主要用于调试和研究目的。
它将LEDPO实现的各个组件和过程分离为独立函数，并添加了丰富的调试信息和注释。

设计目的:
--------
1. 调试工具 - 清晰展示LEDPO训练的各个关键步骤和组件，便于调试
2. 组件解耦 - 将训练流程分解为多个独立函数，便于单独分析和修改
3. 原理探索 - 通过代码注释和调试输出帮助理解可学习beta参数DPO算法原理
4. 验证计算 - 监控beta_scale参数的学习过程

主要功能模块:
-----------
- setup_logging: 设置日志记录格式
- print_section: 打印分隔区域标题
- load_config_file: 加载YAML配置文件
- process_args: 处理和转换参数
- prepare_tokenizer_and_model: 准备tokenizer和模型
- prepare_dataset: 准备数据集
- setup_trainer: 设置LEDPO训练器
- run_ledpo_training: 执行训练过程
- create_callbacks: 创建训练回调函数
- run_ledpo_workflow: 整合所有步骤运行完整工作流

使用场景:
--------
- 调试LEDPO特定训练过程中的问题
- 修改和扩展LEDPO训练流程
- 分析beta_scale参数学习过程
- 验证可学习beta参数的工作效果
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
from llamafactory.train.ledpo.trainer import LEDPOTrainer  # 使用LEDPO的训练器
from llamafactory.train.callbacks import SaveProcessorCallback
from llamafactory.train.ledpo.workflow import run_ledpo  # 导入LEDPO工作流
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
        log_file = os.path.join(output_dir, f"ledpo_training_{timestamp}.log")
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
        
        # 检查并确保stage设置为ledpo
        if finetuning_args.stage != "ledpo":
            logger.warning(f"配置文件中stage={finetuning_args.stage}，但此脚本为LEDPO，强制将stage设置为ledpo")
            finetuning_args.stage = "ledpo"
        
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
    logger.info(f"加载tokenizer: {model_args.model_name_or_path}")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 获取模板并修复tokenizer
    logger.info(f"模板类型: {data_args.template}")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 加载模型
    logger.info(f"加载模型: {model_args.model_name_or_path}")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    # 创建参考模型
    ref_model = None
    if finetuning_args.use_ref_model:
        logger.info("创建参考模型...")
        ref_model = create_ref_model(model_args, finetuning_args)
    
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
        stage="rm",  # LEDPO与DPO一样使用rm数据
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
    设置LEDPO训练器
    """
    logger = logging.getLogger(__name__)
    print_section("设置LEDPO训练器")
    
    # 更新训练参数
    logger.info("设置训练参数...")
    training_args.remove_unused_columns = False  # 对于多模态和成对数据集很重要
    
    logger.info(f"基础beta值: {finetuning_args.pref_beta}")
    logger.info(f"损失函数类型: {finetuning_args.pref_loss}")
    
    # 创建训练器
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
    
    # 获取可学习的beta_scale参数初始值
    if hasattr(trainer, "beta_scale"):
        logger.info(f"初始beta_scale值: {trainer.beta_scale.item()}")
        logger.info(f"初始动态beta值: {trainer.get_dynamic_beta().item()}")
    
    # 打印训练器信息
    logger.info(f"梯度累积步数: {training_args.gradient_accumulation_steps}")
    logger.info(f"学习率: {training_args.learning_rate}")
    logger.info(f"优化器: {training_args.optim}")
    
    return trainer


def debug_compute_preference_loss(trainer, batch):
    """
    调试preference_loss计算过程
    """
    logger = logging.getLogger(__name__)
    print_section("调试LEDPO损失计算")
    
    # 前向传播
    logger.info("计算模型输出...")
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_chosen_avg_logps,
    ) = trainer.concatenated_forward(trainer.model, batch)
    
    # 获取参考模型的日志概率
    logger.info("计算参考模型输出...")
    reference_chosen_logps, reference_rejected_logps = trainer.compute_reference_log_probs(trainer.model, batch)
    
    # 获取动态beta值
    dynamic_beta = trainer.get_dynamic_beta()
    logger.info(f"当前beta_scale值: {trainer.beta_scale.item()}")
    logger.info(f"当前动态beta值: {dynamic_beta.item()}")
    
    # 计算损失
    logger.info("计算preference损失...")
    losses, chosen_rewards, rejected_rewards = trainer.compute_preference_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )
    
    # 打印损失相关值
    logger.info(f"损失平均值: {losses.mean().item()}")
    logger.info(f"选择奖励平均值: {chosen_rewards.mean().item()}")
    logger.info(f"拒绝奖励平均值: {rejected_rewards.mean().item()}")
    logger.info(f"奖励差值平均值: {(chosen_rewards - rejected_rewards).mean().item()}")
    logger.info(f"准确率: {(chosen_rewards > rejected_rewards).float().mean().item()}")
    
    return losses, chosen_rewards, rejected_rewards


def run_ledpo_training(trainer, training_args, finetuning_args, dataset_module=None):
    """
    执行LEDPO训练过程
    """
    logger = logging.getLogger(__name__)
    print_section("执行LEDPO训练")
    
    # 调试模式：检查一个批次的损失计算
    debug_mode = False
    if debug_mode and dataset_module and "train_dataset" in dataset_module:
        # 创建一个小批次用于调试
        from torch.utils.data import DataLoader
        
        logger.info("调试模式：分析一个批次的损失计算...")
        debug_loader = DataLoader(
            dataset_module["train_dataset"],
            batch_size=4,
            shuffle=False,
            collate_fn=trainer.data_collator
        )
        for batch in debug_loader:
            # 将批次移动到设备上
            batch = {k: v.to(trainer.args.device) if hasattr(v, "to") else v for k, v in batch.items()}
            debug_compute_preference_loss(trainer, batch)
            break  # 只处理一个批次
    
    # 执行训练
    if training_args.do_train:
        logger.info("开始训练...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # 保存最终模型
        logger.info("保存最终模型...")
        trainer.save_model()
        
        # 记录训练指标
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        # 输出最终beta_scale值
        if hasattr(trainer, "beta_scale"):
            final_beta_scale = trainer.beta_scale.item()
            final_dynamic_beta = trainer.get_dynamic_beta().item()
            logger.info(f"训练后beta_scale值: {final_beta_scale}")
            logger.info(f"训练后动态beta值: {final_dynamic_beta}")
            
            # 将beta值保存到额外指标中
            extra_metrics = {
                "train_final_beta_scale": final_beta_scale,
                "train_final_dynamic_beta": final_dynamic_beta,
            }
            trainer.save_metrics("train_extra", extra_metrics)
        
        # 绘制损失曲线
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            logger.info("绘制损失曲线...")
            plot_loss(
                training_args.output_dir, 
                keys=["loss", "eval_loss", "beta_scale", "dynamic_beta", "rewards/accuracies"]
            )
    
    # 执行评估
    if training_args.do_eval:
        logger.info("开始评估...")
        metrics = trainer.evaluate()
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def create_callbacks(model_args, data_args, training_args, finetuning_args, generating_args):
    """
    创建训练回调函数
    """
    logger = logging.getLogger(__name__)
    print_section("创建回调函数")
    
    callbacks = []
    
    # 定义一个监控beta_scale变化的回调函数
    class BetaScaleMonitorCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            """在每个epoch结束时记录beta_scale的值"""
            trainer = kwargs.get("trainer", None)
            if trainer and hasattr(trainer, "beta_scale"):
                beta_scale = trainer.beta_scale.item()
                dynamic_beta = trainer.get_dynamic_beta().item()
                logger.info(f"Epoch {state.epoch}，beta_scale值: {beta_scale}，动态beta值: {dynamic_beta}")
    
    # 添加beta_scale监控回调
    callbacks.append(BetaScaleMonitorCallback())
    
    logger.info(f"创建了 {len(callbacks)} 个回调函数")
    return callbacks


def run_ledpo_workflow(config_path: str):
    """
    运行完整的LEDPO工作流程
    """
    # 首先设置基本控制台日志
    temp_logger = setup_logging()
    temp_logger.info(f"开始LEDPO工作流程，配置文件: {config_path}")
    
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
            os.environ["WANDB_PROJECT"] = "ledpo"
            logger.info("通过环境变量设置WANDB项目名称为: ledpo")
        
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
        run_ledpo_training(trainer, training_args, finetuning_args, dataset_module)
        
        # 8. 创建模型卡片和推送（如果需要）
        if hasattr(trainer, "create_modelcard_and_push"):
            trainer.create_modelcard_and_push(
                model_args, data_args, training_args, finetuning_args
            )
        
        logger.info("LEDPO工作流程完成！")
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
            config_path = "examples/train_lora/qwen1_5_0_5b_lora_dpo.yaml"
        else:
            config_path = sys.argv[1]
        
        logger.info(f"使用配置文件: {config_path}")
        
        # 运行LEDPO工作流程
        return run_ledpo_workflow(config_path)
    
    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        return 130
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    main() 