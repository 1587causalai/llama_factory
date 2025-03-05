#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
详细拆解的fooDPO训练脚本 (fooDPO Training Script with Detailed Components)
==============================================================

此脚本对LLaMA-Factory中的fooDPO训练流程进行了详细拆解，主要用于教学和研究目的。
它将官方实现的各个组件和过程分离为独立函数，并添加了丰富的调试信息和注释。

设计目的:
--------
1. 教学展示 - 清晰展示fooDPO训练的各个关键步骤和组件
2. 原理解析 - 通过代码注释和调试输出帮助理解fooDPO算法原理
3. 组件解耦 - 将训练流程分解为多个独立函数，便于单独分析和修改

主要功能模块:
-----------
- debug_print_attrs: 打印对象属性，用于调试
- load_config: 加载YAML配置文件
- prepare_tokenizer_and_template: 准备tokenizer和模板
- prepare_model: 加载和配置模型
- prepare_dataset: 准备数据集
- setup_foodpo_trainer: 设置fooDPO训练器
- run_training: 执行训练过程

与其他脚本的区别:
--------------
- 相比run_foodpo_detailed.py: 本脚本更侧重于组件和原理的展示，添加了更多调试功能
- 相比run_foodpo_training.py: 本脚本完全拆解了训练流程，牺牲了简洁性换取更高的可读性和教学价值

使用场景:
--------
- 学习和理解fooDPO训练原理
- 调试特定训练组件
- 修改和扩展fooDPO训练流程
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# 导入所需的LLaMA-Factory模块
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
from llamafactory.train.dpo.trainer import CustomDPOTrainer
from llamafactory.train.trainer_utils import create_ref_model
from llamafactory.extras.misc import calculate_tps
from llamafactory.extras.logging import get_logger
from llamafactory.extras.ploting import plot_loss
from transformers import TrainerCallback


# 创建logger
logger = get_logger(__name__)


def debug_print_attrs(obj, name="object"):
    """
    打印对象的所有属性和方法，用于调试
    """
    print(f"\n== {name} 属性: ==")
    attrs = [attr for attr in dir(obj) if not attr.startswith('_')]
    attrs.sort()
    for attr in attrs:
        try:
            value = getattr(obj, attr)
            if not callable(value):
                print(f"{attr}: {value}")
        except Exception as e:
            print(f"{attr}: [错误: {e}]")
    print(f"== {name} 属性打印完成 ==\n")


def load_config(config_path: str) -> Tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]:
    """
    加载YAML配置文件并解析成参数对象
    
    Args:
        config_path: 配置文件的路径
        
    Returns:
        解析后的参数对象元组
    """
    print(f"加载配置文件: {config_path}")
    # 临时修改sys.argv以使用get_train_args加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + [config_path]
    
    # 获取参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    
    # 设置remove_unused_columns=False，解决数据集列名与模型forward方法签名不匹配的问题
    training_args.remove_unused_columns = False
    
    # 恢复原始命令行参数
    sys.argv = original_argv
    
    # 打印关键参数类的属性（调试用）
    debug_print_attrs(training_args, "training_args")
    debug_print_attrs(data_args, "data_args")
    debug_print_attrs(finetuning_args, "finetuning_args")
    
    # 显示关键参数
    print("模型路径:", model_args.model_name_or_path)
    print("训练阶段:", finetuning_args.stage)
    print("微调类型:", finetuning_args.finetuning_type)
    print("学习率:", training_args.learning_rate)
    print("数据集:", data_args.dataset)
    
    return model_args, data_args, training_args, finetuning_args, generating_args


def prepare_tokenizer_and_template(model_args: ModelArguments, data_args: DataArguments):
    """
    准备tokenizer和模板
    
    Args:
        model_args: 模型参数
        data_args: 数据参数
        
    Returns:
        tokenizer和template
    """
    print("准备tokenizer和模板...")
    # 加载tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 获取模板并根据需要修复tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    print(f"使用模板: {data_args.template}")
    return tokenizer, template, tokenizer_module


def prepare_model(tokenizer, model_args: ModelArguments, finetuning_args: FinetuningArguments, do_train: bool):
    """
    准备模型
    
    Args:
        tokenizer: 分词器
        model_args: 模型参数
        finetuning_args: 微调参数
        do_train: 是否进行训练
        
    Returns:
        加载的模型
    """
    print("加载模型...")
    model = load_model(tokenizer, model_args, finetuning_args, do_train)
    print(f"模型类型: {model.__class__.__name__}")
    return model


def prepare_dataset(template, tokenizer_module, model_args, data_args, training_args):
    """
    准备数据集
    
    Args:
        template: 模板
        tokenizer_module: tokenizer模块
        model_args: 模型参数
        data_args: 数据参数
        training_args: 训练参数
        
    Returns:
        处理后的数据集
    """
    print("准备数据集...")
    # 获取数据集，在DPO中使用'rm'阶段的数据处理方式
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    
    print(f"训练样本数: {len(dataset_module['train_dataset']) if 'train_dataset' in dataset_module else 'N/A'}")
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"] is not None:
        print(f"验证样本数: {len(dataset_module['eval_dataset'])}")
    
    # 调试：打印一个训练样本，了解数据结构
    if 'train_dataset' in dataset_module and len(dataset_module['train_dataset']) > 0:
        print("\n== 数据样本示例 ==")
        sample = dataset_module['train_dataset'][0]
        for key, value in sample.items():
            print(f"{key}:")
            print(value)
        print("== 样本打印完成 ==\n")
    
    return dataset_module


def calculate_avg_tokens_per_sample(dataset, tokenizer, keys=("query", "response", "rejected")):
    """
    计算数据集中每个样本的平均token数
    
    Args:
        dataset: 数据集
        tokenizer: 分词器
        keys: 需要计算的字段
        
    Returns:
        平均token数
    """
    if len(dataset) == 0:
        return 0
    
    total_tokens = 0
    for sample in dataset:
        for key in keys:
            if key in sample and isinstance(sample[key], str):
                total_tokens += len(tokenizer.encode(sample[key]))
    
    return total_tokens / len(dataset)


def setup_foodpo_trainer(
    model, 
    ref_model, 
    training_args, 
    finetuning_args,
    tokenizer, 
    template,
    data_collator, 
    callbacks, 
    dataset_module
):
    """
    设置fooDPO训练器
    
    Args:
        model: 模型
        ref_model: 参考模型
        training_args: 训练参数
        finetuning_args: 微调参数
        tokenizer: 分词器
        template: 模板
        data_collator: 数据整理器
        callbacks: 回调函数
        dataset_module: 数据集模块
    
    Returns:
        fooDPO训练器
    """
    print("设置fooDPO训练器...")
    
    # 计算训练预估速度
    train_dataset = dataset_module.get("train_dataset")
    if train_dataset is not None:
        # 计算每个样本平均token数
        print("计算每个样本的平均token数...")
        total_tokens = 0
        sample_count = 0
        
        for sample in train_dataset:
            if "chosen_input_ids" in sample:
                total_tokens += len(sample["chosen_input_ids"])
                sample_count += 1
            if "rejected_input_ids" in sample:
                total_tokens += len(sample["rejected_input_ids"])
                sample_count += 1
        
        if sample_count > 0:
            avg_tokens = total_tokens / sample_count
            print(f"每个样本平均token数: {avg_tokens:.2f}")
    
    # 创建fooDPO训练器
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        processor=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module
    )
    
    return trainer


def run_training(trainer, training_args):
    """
    运行训练过程
    
    Args:
        trainer: 训练器
        training_args: 训练参数
    
    Returns:
        训练结果
    """
    print("开始训练...")
    
    # 训练模型
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # 保存模型
    trainer.save_model()
    trainer.save_state()
    
    # 打印训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 绘制损失曲线
    if hasattr(training_args, 'plot_loss') and training_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss", "chosen_reward", "rejected_reward"])
    
    print("训练完成！")
    return train_result


def main():
    """
    fooDPO训练主函数
    """
    print("=" * 50)
    print("开始fooDPO训练详细流程")
    print("=" * 50)
    
    try:
        # 1. 加载配置
        config_path = "examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml"
        model_args, data_args, training_args, finetuning_args, generating_args = load_config(config_path)
        
        # 2. 准备tokenizer和模板
        tokenizer, template, tokenizer_module = prepare_tokenizer_and_template(model_args, data_args)
        
        # 3. 准备模型
        model = prepare_model(tokenizer, model_args, finetuning_args, training_args.do_train)
        
        # 4. 准备数据集
        dataset_module = prepare_dataset(template, tokenizer_module, model_args, data_args, training_args)
        
        # 5. 设置训练器
        trainer = setup_foodpo_trainer(
            model=model,
            ref_model=create_ref_model(model, finetuning_args),
            training_args=training_args,
            finetuning_args=finetuning_args,
            tokenizer=tokenizer,
            template=template,
            data_collator=PairwiseDataCollatorWithPadding(
                template=template,
                model=model,
                tokenizer=tokenizer,
                label_pad_token_id=-100,
                pad_to_multiple_of=8
            ),
            callbacks=None,
            dataset_module=dataset_module
        )
        
        # 6. 运行训练
        if training_args.do_train:
            train_result = run_training(trainer, training_args)
        
        # 7. 运行评估
        if training_args.do_eval:
            print("开始评估...")
            eval_results = trainer.evaluate()
            trainer.log_metrics("eval", eval_results)
            trainer.save_metrics("eval", eval_results)
        
        print("=" * 50)
        print("fooDPO训练流程完成")
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 