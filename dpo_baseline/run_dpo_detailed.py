#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA-Factory DPO训练详细实现脚本 (Detailed DPO Training Implementation)
=====================================================================

此脚本提供了与LLaMA-Factory命令行工具相同的DPO训练功能，但采用了更直观的流程划分和丰富的输出信息。
它是一个独立可执行的Python脚本，无需依赖命令行接口即可完成完整的DPO训练流程。

设计目的:
--------
1. 实用性实现 - 提供一个可立即使用的DPO训练脚本，功能与CLI命令等效
2. 流程可视化 - 通过分段打印信息，使训练过程更加透明可见
3. 便于集成和扩展 - 适合集成到其他项目或进行自定义修改

主要功能模块:
-----------
- setup_logging: 配置日志系统
- print_section: 打印格式化的段落标题
- load_config_file: 加载和显示配置文件内容
- process_args: 处理配置文件并返回参数对象
- prepare_tokenizer_and_model: 准备tokenizer、模板和模型
- prepare_dataset: 准备数据集和数据整理器
- setup_trainer: 设置DPO训练器
- run_dpo_training: 执行DPO训练和评估
- run_dpo_workflow: 组织完整的DPO训练工作流程

与其他脚本的区别:
--------------
- 相比dpo_detailed.py: 本脚本更侧重于实际使用，减少了调试代码，增加了流程的清晰度
- 相比run_dpo_training.py: 本脚本提供了更详细的训练过程输出，适合学习和监控训练过程

使用场景:
--------
- 需要详细了解训练过程时使用
- 需要监控训练中各个阶段性能和状态
- 作为开发新功能的基础框架

使用方法:
--------
python run_dpo_detailed.py [配置文件路径]

如果不提供配置文件路径，默认使用 'examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml'
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

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
from llamafactory.train.dpo.trainer import CustomDPOTrainer
from llamafactory.train.trainer_utils import create_ref_model
from llamafactory.train.callbacks import LogCallback, ReporterCallback, PissaConvertCallback
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps
from llamafactory.extras.ploting import plot_loss

# 设置日志
logger = get_logger(__name__)

def setup_logging(level=logging.INFO):
    """
    设置日志配置
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def print_section(title):
    """打印带分隔线的段落标题"""
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}\n")

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    print_section(f"加载配置文件: {config_path}")
    
    # 读取YAML文件内容
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 打印配置内容
    print("配置内容预览:")
    for section, values in config.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {values}")
    
    return config

def process_args(config_path: str) -> Tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]:
    """
    处理配置文件并返回参数对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        解析后的参数对象元组
    """
    # 保存并修改命令行参数，以便read_args能正确读取配置文件
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], config_path]
    
    # 读取参数
    args = read_args()
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    # 恢复原始命令行参数
    sys.argv = original_argv
    
    # 设置remove_unused_columns=False，这对成对数据集很重要
    training_args.remove_unused_columns = False
    
    # 打印关键参数
    print_section("关键训练参数")
    print(f"模型: {model_args.model_name_or_path}")
    print(f"数据集: {data_args.dataset}")
    print(f"训练阶段: {finetuning_args.stage}")
    print(f"微调类型: {finetuning_args.finetuning_type}")
    print(f"偏好beta值: {finetuning_args.pref_beta}")
    print(f"批处理大小: {training_args.per_device_train_batch_size}")
    print(f"学习率: {training_args.learning_rate}")
    print(f"训练轮次: {training_args.num_train_epochs}")
    print(f"输出目录: {training_args.output_dir}")
    
    return model_args, data_args, training_args, finetuning_args, generating_args

def prepare_tokenizer_and_model(model_args, finetuning_args, data_args, training_args, do_train=True):
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
    print_section("准备Tokenizer和模型")
    
    # 加载tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    print(f"Tokenizer类型: {tokenizer.__class__.__name__}")
    
    # 获取模板并修复tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    print(f"使用模板: {data_args.template}")
    
    # 加载模型
    print(f"加载模型: {model_args.model_name_or_path}")
    model = load_model(tokenizer, model_args, finetuning_args, do_train)
    print(f"模型类型: {model.__class__.__name__}")
    
    # 创建参考模型
    print("准备参考模型...")
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not do_train):
            print("使用主模型作为参考模型")
            ref_model = model
        else:
            print(f"使用指定的参考模型: {finetuning_args.ref_model or '基于主模型创建'}")
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        print("不使用参考模型")
        ref_model = None
    
    return tokenizer, template, model, ref_model, tokenizer_module

def prepare_dataset(template, model_args, data_args, training_args, tokenizer_module):
    """
    准备数据集
    
    Args:
        template: 模板
        model_args: 模型参数
        data_args: 数据参数
        training_args: 训练参数
        tokenizer_module: tokenizer模块
        
    Returns:
        处理后的数据集和数据整理器
    """
    print_section("准备数据集")
    
    # 获取数据集，在DPO中使用'rm'阶段的数据处理方式
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
        print(f"训练集样本数: {train_size}")
        
        # 打印一个样本示例
        if train_size > 0:
            print("\n训练样本示例:")
            sample = dataset_module["train_dataset"][0]
            for k, v in sample.items():
                if k.endswith("_ids") or k.endswith("_mask"):
                    print(f"  {k}: [张量, 长度={len(v)}]")
                else:
                    print(f"  {k}: {v}")
    
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"] is not None:
        print(f"验证集样本数: {len(dataset_module['eval_dataset'])}")
    
    # 创建数据整理器
    tokenizer = tokenizer_module["tokenizer"]
    model = dataset_module.get("model", None)
    
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )
    print("创建了PairwiseDataCollatorWithPadding数据整理器")
    
    return dataset_module, data_collator

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
    设置DPO训练器
    
    Args:
        model: 模型
        ref_model: 参考模型
        training_args: 训练参数
        finetuning_args: 微调参数
        data_collator: 数据整理器
        dataset_module: 数据集模块
        tokenizer_module: tokenizer模块
        callbacks: 回调函数列表
        
    Returns:
        DPO训练器
    """
    print_section("设置DPO训练器")
    
    # 初始化DPO训练器
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
    
    print(f"创建了CustomDPOTrainer，训练设备: {training_args.device}")
    return trainer

def run_dpo_training(trainer, training_args, finetuning_args, dataset_module=None):
    """
    执行DPO训练
    
    Args:
        trainer: 训练器
        training_args: 训练参数
        finetuning_args: 微调参数
        dataset_module: 数据集模块，用于计算指标（可选）
    """
    print_section("执行DPO训练")
    
    if training_args.do_train:
        print("开始训练...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        print("保存模型...")
        trainer.save_model()
        trainer.save_state()
        
        # 记录指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # 添加有效token处理速度指标
        if finetuning_args.include_effective_tokens_per_second and dataset_module and "train_dataset" in dataset_module:
            metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], metrics, stage="rm"
            )
            print(f"有效token处理速度: {metrics['effective_tokens_per_sec']:.2f} tokens/sec")
        
        # 绘制损失曲线
        if finetuning_args.plot_loss:
            print("绘制损失曲线...")
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])
        
        # 计算训练步数
        train_steps = None
        if hasattr(trainer.state, "global_step"):
            train_steps = trainer.state.global_step
        elif "train_steps_per_second" in metrics and "train_runtime" in metrics:
            # 如果没有直接的步数，可以从每秒步数和总运行时间计算
            train_steps = round(metrics["train_steps_per_second"] * metrics["train_runtime"])
        elif "train_samples_per_second" in metrics and "train_runtime" in metrics:
            # 或者从每秒样本数、批量大小和总运行时间估算
            effective_batch_size = (
                training_args.per_device_train_batch_size * 
                training_args.gradient_accumulation_steps * 
                (training_args.n_gpu if hasattr(training_args, "n_gpu") and training_args.n_gpu > 0 else 1)
            )
            train_steps = round(
                (metrics["train_samples_per_second"] * metrics["train_runtime"]) / effective_batch_size
            )
        
        print(f"训练完成！训练步数: {train_steps or '未能计算'}")
        print(f"总训练时间: {metrics.get('train_runtime', '未知')} 秒")
        print(f"平均每步时间: {1.0/metrics.get('train_steps_per_second', 0) if metrics.get('train_steps_per_second', 0) > 0 else '未知'} 秒")
    
    # 评估
    if training_args.do_eval:
        print("开始评估...")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        # 如果参考模型就是模型本身，无法计算奖励指标
        model = trainer.model
        ref_model = trainer.ref_model
        if id(model) == id(ref_model):
            print("参考模型与主模型相同，无法计算奖励指标")
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"评估完成！评估指标: {metrics}")

def create_callbacks(model_args, data_args, training_args, finetuning_args, generating_args):
    """
    创建训练回调函数
    
    Args:
        model_args: 模型参数
        data_args: 数据参数
        training_args: 训练参数
        finetuning_args: 微调参数
        generating_args: 生成参数
        
    Returns:
        回调函数列表
    """
    callbacks = []
    
    # 添加日志回调
    callbacks.append(LogCallback())
    
    # 添加PISSA转换回调，如果启用
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())
    
    # 添加SwanLab回调，如果启用
    if hasattr(finetuning_args, 'use_swanlab') and finetuning_args.use_swanlab:
        from llamafactory.train.trainer_utils import get_swanlab_callback
        callbacks.append(get_swanlab_callback(finetuning_args))
    
    # 添加Reporter回调
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))
    
    return callbacks
    
def run_dpo_workflow(config_path: str):
    """
    完整的DPO训练工作流程
    
    Args:
        config_path: 配置文件路径
    """
    try:
        # 处理参数
        model_args, data_args, training_args, finetuning_args, generating_args = process_args(config_path)
        
        # 检查是否为DPO训练
        if finetuning_args.stage != "dpo":
            raise ValueError(f"配置文件指定的训练阶段不是DPO，而是: {finetuning_args.stage}")
        
        # 创建回调函数
        callbacks = create_callbacks(model_args, data_args, training_args, finetuning_args, generating_args)
        
        # 准备tokenizer和模型
        tokenizer, template, model, ref_model, tokenizer_module = prepare_tokenizer_and_model(
            model_args, finetuning_args, data_args, training_args, training_args.do_train
        )
        
        # 准备数据集
        dataset_module, data_collator = prepare_dataset(
            template, model_args, data_args, training_args, tokenizer_module
        )
        
        # 设置训练器
        trainer = setup_trainer(
            model, ref_model, training_args, finetuning_args, 
            data_collator, dataset_module, tokenizer_module, callbacks
        )
        
        # 执行训练
        run_dpo_training(trainer, training_args, finetuning_args, dataset_module)
        
        print("\n🎉 DPO训练流程全部完成！")
        
    except Exception as e:
        logger.error(f"DPO训练过程中出错: {str(e)}")
        raise

def main():
    """
    主函数，处理命令行参数并启动训练
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
    
    # 加载配置预览
    _ = load_config_file(config_path)
    
    # 执行DPO训练
    run_dpo_workflow(config_path)

if __name__ == "__main__":
    main() 


# 运行命令
# python dpo_baseline/run_dpo_detailed.py examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml

# 运行结果
# 加载配置文件: examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml
# 配置内容预览:
