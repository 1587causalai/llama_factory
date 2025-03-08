#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA-Factory DPO训练调试脚本
=====================================================================

此脚本是run_ledpo_rich.py的简化版本，专为断点调试设计。
提供清晰的模块化结构与断点调试指南，方便对训练流程的关键步骤进行检查。

使用方法:
--------
python run_ledpo_debug.py [配置文件路径]

断点调试指南:
----------
在标记为 "# BREAKPOINT: [说明]" 的位置设置断点，可以检查各个训练阶段的关键数据和状态
"""

import os
import sys
import logging
import yaml
import time
import pdb  # 用于手动断点调试
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# 导入Rich库核心组件（仅用于基础美化）
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.syntax import Syntax

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
from llamafactory.train.callbacks import LogCallback, ReporterCallback, PissaConvertCallback
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps
from llamafactory.extras.ploting import plot_loss

# 创建简洁控制台
console = Console(highlight=True)
logger = get_logger(__name__)

def setup_logging(level=logging.INFO, output_dir=None):
    """设置日志配置"""
    handlers = [RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)]
    
    if output_dir is not None:
        log_file = os.path.join(output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    logging.basicConfig(
        format="%(message)s",
        datefmt="[%X]",
        level=level,
        handlers=handlers
    )
    
    return logging.getLogger("ledpo_debug")

def create_rich_table(title, columns, rows):
    """创建美观的Rich表格"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    return table

def create_rich_panel(content, title=None, style="green"):
    """创建美观的Rich面板"""
    return Panel(content, title=title, border_style=style)

#######################
# 阶段1: 配置加载和处理 #
#######################

def load_and_process_config(config_path: str) -> Tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]:
    """加载并处理配置文件"""
    # 读取YAML配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # BREAKPOINT: 配置加载 - 检查原始配置
    # pdb.set_trace()
    
    # 显示加载的配置
    console.print(create_rich_panel(f"从文件加载配置: {config_path}", title="配置加载", style="cyan"))
    
    # 处理命令行参数
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], config_path]
    
    # 读取参数
    args = read_args()
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    # 恢复原始命令行参数
    sys.argv = original_argv
    
    # 设置remove_unused_columns=False
    training_args.remove_unused_columns = False
    
    # 使用Rich表格显示配置信息
    config_tables = []
    
    # 模型配置表
    model_rows = []
    for k, v in vars(model_args).items():
        model_rows.append([k, str(v)])
    config_tables.append(create_rich_table("模型配置", ["参数", "值"], model_rows))
    
    # 数据配置表
    data_rows = []
    for k, v in vars(data_args).items():
        data_rows.append([k, str(v)])
    config_tables.append(create_rich_table("数据配置", ["参数", "值"], data_rows))
    
    # 训练配置表
    train_rows = []
    for k, v in vars(training_args).items():
        if k not in ["_n_gpu", "local_rank", "xpu_backend"]:  # 跳过不重要的参数
            train_rows.append([k, str(v)])
    config_tables.append(create_rich_table("训练配置", ["参数", "值"], train_rows))
    
    # 微调配置表
    finetune_rows = []
    for k, v in vars(finetuning_args).items():
        finetune_rows.append([k, str(v)])
    config_tables.append(create_rich_table("微调配置", ["参数", "值"], finetune_rows))
    
    # 显示配置表格
    console.print(Panel(config_tables[0], title="模型配置详情", border_style="blue"))
    console.print(Panel(config_tables[3], title="微调配置详情", border_style="blue"))
    
    return model_args, data_args, training_args, finetuning_args, generating_args

#########################
# 阶段2: 模型和分词器准备 #
#########################

def prepare_model_components(model_args, finetuning_args, data_args, training_args, do_train=True):
    """准备模型、分词器和模板"""
    console.print(f"\n[bold cyan]阶段2: 准备模型组件[/bold cyan]")
    
    # 1. 加载tokenizer
    console.print("[green]加载Tokenizer...[/green]")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 2. 获取模板
    console.print("[green]获取模板...[/green]")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 3. 加载模型
    console.print(f"[green]加载模型: {model_args.model_name_or_path}...[/green]")
    model = load_model(tokenizer, model_args, finetuning_args, do_train)
    
    # 4. 创建参考模型
    console.print("[green]准备参考模型...[/green]")
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not do_train):
            console.print("[yellow]参考模型与主模型相同[/yellow]")
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        console.print("[yellow]未使用参考模型[/yellow]")
        ref_model = None
    
    # 打印模型信息摘要
    console.print(f"[green]Tokenizer类型:[/green] {tokenizer.__class__.__name__}")
    console.print(f"[green]模板:[/green] {data_args.template}")
    console.print(f"[green]模型类型:[/green] {model.__class__.__name__}")
    if ref_model is not None:
        ref_model_type = "同主模型" if id(model) == id(ref_model) else ref_model.__class__.__name__
    else:
        ref_model_type = "无"
    console.print(f"[green]参考模型:[/green] {ref_model_type}")
    
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
    console.print(f"\n[bold cyan]阶段3: 准备训练数据[/bold cyan]")
    
    # BREAKPOINT: 数据集加载前 - 检查数据参数
    # pdb.set_trace()
    
    # 获取数据集
    console.print("[green]加载数据集...[/green]")
    dataset_module = get_dataset(
        template, 
        model_args, 
        data_args, 
        training_args, 
        stage="rm",  # DPO使用RM阶段的数据处理逻辑
        **tokenizer_module
    )
    
    # BREAKPOINT: 数据集加载后 - 检查数据集结构和样本
    # pdb.set_trace()
    
    # 打印数据集信息
    if "train_dataset" in dataset_module:
        train_size = len(dataset_module["train_dataset"])
        console.print(f"[green]训练集样本数:[/green] {train_size}")
        
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"] is not None:
        eval_size = len(dataset_module["eval_dataset"])
        console.print(f"[green]验证集样本数:[/green] {eval_size}")
    
    # 创建数据整理器
    tokenizer = tokenizer_module["tokenizer"]
    model = dataset_module.get("model", None)
    
    console.print("[green]创建数据整理器...[/green]")
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )
    
    # BREAKPOINT: 检查数据整理器 - 确认数据整理逻辑正确
    # pdb.set_trace()
    
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
    model_args=None,  # 添加model_args参数
    data_args=None,   # 添加data_args参数
    generating_args=None  # 添加generating_args参数
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
        model_args: 模型参数（新增）
        data_args: 数据参数（新增）
        generating_args: 生成参数（新增）
        
    Returns:
        DPO训练器
    """
    console.print(f"\n[bold cyan]阶段4: 设置训练器[/bold cyan]")
    
    # BREAKPOINT: 创建回调函数前
    # pdb.set_trace()
    
    # 创建回调函数
    callbacks = []
    callbacks.append(LogCallback())
    
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())
    
    # 添加SwanLab回调，如果启用
    if hasattr(finetuning_args, 'use_swanlab') and finetuning_args.use_swanlab:
        from llamafactory.train.trainer_utils import get_swanlab_callback
        callbacks.append(get_swanlab_callback(finetuning_args))
    
    # 修复：正确传递参数给ReporterCallback
    callbacks.append(ReporterCallback(
        model_args=model_args,  # 使用传入的model_args
        data_args=data_args,    # 使用传入的data_args
        finetuning_args=finetuning_args, 
        generating_args=generating_args  # 使用传入的generating_args
    ))
    
    # BREAKPOINT: 初始化训练器前 - 检查回调函数
    # pdb.set_trace()
    
    # 初始化DPO训练器
    console.print("[green]初始化DPO训练器...[/green]")
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
    
    console.print(f"[green]训练设备:[/green] {training_args.device}")
    
    # BREAKPOINT: 初始化训练器后 - 检查训练器配置和优化器
    # pdb.set_trace()
    
    # 显示优化器参数
    if hasattr(trainer, "optimizer") and trainer.optimizer:
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            console.print(f"[green]参数组 {i}:[/green] {len(param_group['params'])} 个参数, 学习率: {param_group['lr']}")
    
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
    console.print(f"\n[bold cyan]阶段5: 执行训练[/bold cyan]")
    
    # 训练
    if training_args.do_train:
        console.print("[green]开始训练...[/green]")
        
        # BREAKPOINT: 训练前 - 检查训练参数和模型状态
        # pdb.set_trace()
        
        # 执行训练
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # BREAKPOINT: 训练后 - 检查训练结果和指标
        # pdb.set_trace()
        
        console.print("[green]保存模型...[/green]")
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
            console.print(f"[green]有效token处理速度:[/green] {metrics['effective_tokens_per_sec']:.2f} tokens/sec")
        
        # 绘制损失曲线
        if finetuning_args.plot_loss:
            console.print("[green]绘制损失曲线...[/green]")
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])
    
    # 评估
    if training_args.do_eval:
        console.print("[green]开始评估...[/green]")
        
        # BREAKPOINT: 评估前 - 检查评估设置
        # pdb.set_trace()
        
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        # BREAKPOINT: 评估后 - 检查评估结果
        # pdb.set_trace()
        
        # 如果参考模型就是模型本身，无法计算奖励指标
        model = trainer.model
        ref_model = trainer.ref_model
        if id(model) == id(ref_model):
            console.print("[yellow]警告: 参考模型与主模型相同，无法计算奖励指标[/yellow]")
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        for key, value in metrics.items():
            console.print(f"[green]{key}:[/green] {value}")

####################
# 完整的训练工作流程 #
####################

def run_dpo_workflow(config_path: str):
    """
    完整的DPO训练工作流程
    
    Args:
        config_path: 配置文件路径
    """
    # 设置日志
    logger = setup_logging()
    
    try:
        # BREAKPOINT: 工作流开始 - 整体流程开始前检查
        # pdb.set_trace()
        
        console.print("[bold green]启动DPO训练工作流程[/bold green]")
        
        # 阶段1: 加载配置
        console.print(Panel("[1/5 配置加载] ⟳ → [2/5 模型准备] ⟳ → [3/5 数据准备] ⟳ → [4/5 训练器设置] ⟳ → [5/5 执行训练]", 
                           title="[bold yellow]训练进度[/bold yellow]", 
                           border_style="yellow",
                           padding=(1, 2)))
        model_args, data_args, training_args, finetuning_args, generating_args = load_and_process_config(config_path)
        
        # 检查是否为DPO/LEDPO训练
        if finetuning_args.stage not in ["dpo", "ledpo"]:
            raise ValueError(f"配置文件指定的训练阶段不是DPO或LEDPO，而是: {finetuning_args.stage}")
        
        # 确保输出目录存在
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # 使用输出目录重新设置日志
        logger = setup_logging(output_dir=training_args.output_dir)
        logger.info(f"输出目录: {training_args.output_dir}")
        
        # 显示训练进度
        console.print(Panel("[1/5 配置加载] ✓ → [2/5 模型准备] ⟳ → [3/5 数据准备] → [4/5 训练器设置] → [5/5 执行训练]", 
                           title="[bold yellow]训练进度[/bold yellow]", 
                           border_style="yellow",
                           padding=(1, 2)))
        
        # 阶段2: 准备模型组件
        tokenizer, template, model, ref_model, tokenizer_module = prepare_model_components(
            model_args, finetuning_args, data_args, training_args, training_args.do_train
        )
        
        # 更新训练进度
        console.print(Panel("[1/5 配置加载] ✓ → [2/5 模型准备] ✓ → [3/5 数据准备] ⟳ → [4/5 训练器设置] → [5/5 执行训练]", 
                           title="[bold yellow]训练进度[/bold yellow]", 
                           border_style="yellow",
                           padding=(1, 2)))
        
        # 阶段3: 准备数据集
        console.print("\n[bold cyan]阶段3: 准备数据集[/bold cyan]")
        dataset_module, data_collator = prepare_training_data(
            template, model_args, data_args, training_args, tokenizer_module
        )
        
        # 更新训练进度
        console.print(Panel("[1/5 配置加载] ✓ → [2/5 模型准备] ✓ → [3/5 数据准备] ✓ → [4/5 训练器设置] ⟳ → [5/5 执行训练]", 
                           title="[bold yellow]训练进度[/bold yellow]", 
                           border_style="yellow",
                           padding=(1, 2)))
        
        # 阶段4: 设置训练器
        console.print("\n[bold cyan]阶段4: 设置训练器[/bold cyan]")
        trainer = setup_dpo_trainer(
            model, ref_model, training_args, finetuning_args, 
            data_collator, dataset_module, tokenizer_module,
            model_args=model_args,
            data_args=data_args,
            generating_args=generating_args
        )
        
        # 更新训练进度
        console.print(Panel("[1/5 配置加载] ✓ → [2/5 模型准备] ✓ → [3/5 数据准备] ✓ → [4/5 训练器设置] ✓ → [5/5 执行训练] ⟳", 
                           title="[bold yellow]训练进度[/bold yellow]", 
                           border_style="yellow",
                           padding=(1, 2)))
        
        # 阶段5: 执行训练
        console.print("\n[bold cyan]阶段5: 执行训练[/bold cyan]")
        run_training(trainer, training_args, finetuning_args, dataset_module)
        
        # 更新训练进度
        console.print(Panel("[1/5 配置加载] ✓ → [2/5 模型准备] ✓ → [3/5 数据准备] ✓ → [4/5 训练器设置] ✓ → [5/5 执行训练] ✓", 
                           title="[bold yellow]训练进度[/bold yellow]", 
                           border_style="yellow",
                           padding=(1, 2)))
        
        # BREAKPOINT: 工作流结束 - 整体流程完成后检查
        # pdb.set_trace()
        
        console.print("[bold green]DPO训练流程全部完成！[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")
        logger.error(f"DPO训练过程中出错: {str(e)}")
        # BREAKPOINT: 错误处理 - 异常发生时检查
        # pdb.set_trace()
        raise

def main():
    """主函数，处理命令行参数并启动训练"""
    # 设置日志
    logger = setup_logging()
    
    console.print(Panel.fit(
        "[bold green]LLaMA-Factory DPO训练调试工具[/bold green]\n"
        "[cyan]专为断点调试设计，提供清晰的模块化结构与关键断点[/cyan]",
        title="🔍 调试工具",
        border_style="green"
    ))
    
    # 从命令行获取配置文件路径
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml"
        logger.info(f"未指定配置文件，使用默认配置: {config_path}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        console.print(Panel(
            f"[bold red]配置文件不存在: {config_path}[/bold red]\n"
            "请提供有效的配置文件路径作为命令行参数",
            title="❌ 错误",
            border_style="red"
        ))
        sys.exit(1)
    
    # 显示配置文件语法高亮
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_content = f.read()
        yaml_syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
        console.print(Panel(yaml_syntax, title=f"配置文件: {config_path}", border_style="blue"))
    except Exception:
        pass  # 如果无法显示语法高亮，则忽略
    
    # 执行DPO训练
    run_dpo_workflow(config_path)

if __name__ == "__main__":
    main()



# 运行命令示例:
# python dpo_baseline/run_ledpo_debug.py examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml

# 相关conda环境和模型:
# conda activate llama
# 本地模型路径: $home/models/Qwen1.5-0.5B 


# 调试指南:
# 1. 取消需要调试步骤的 # pdb.set_trace() 注释
# 2. 运行脚本，会在断点处停止
# 3. 在pdb提示符下，可以使用以下命令检查变量:
#    - p 变量名    # 打印变量值
#    - pp 变量名   # 美化打印变量
#    - dir(对象)   # 查看对象的属性和方法
#    - n          # 执行下一行
#    - s          # 步入函数
#    - c          # 继续执行
#    - q          # 退出调试

# 常用断点位置:
# 1. 配置加载: 检查配置文件是否正确读取
# 2. 模型准备: 检查模型是否正确加载
# 3. 数据准备: 检查数据集是否正确处理
# 4. 训练开始前: 检查训练参数设置
# 5. 训练后: 检查训练结果
