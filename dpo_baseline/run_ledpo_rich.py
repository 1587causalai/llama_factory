#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA-Factory DPO训练详细实现脚本 (Rich UI 版本)
=====================================================================

此脚本提供了与LLaMA-Factory命令行工具相同的DPO训练功能，使用Rich库美化了终端输出。
它是一个独立可执行的Python脚本，无需依赖命令行接口即可完成完整的DPO训练流程。

设计目的:
--------
1. 美观的终端界面 - 使用Rich库提供彩色、进度条等丰富的UI元素
2. 流程可视化 - 通过分段打印信息，使训练过程更加透明可见
3. 便于集成和扩展 - 适合集成到其他项目或进行自定义修改

使用方法:
--------
python run_ledpo_rich.py [配置文件路径]

如果不提供配置文件路径，默认使用 'examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml'
"""

import os
import sys
import logging
import yaml
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# 导入Rich库组件
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.logging import RichHandler
from rich.live import Live
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

# 创建Rich控制台
console = Console(highlight=True)

# 设置日志
logger = get_logger(__name__)

def setup_logging(level=logging.INFO, output_dir=None):
    """
    设置日志配置，使用Rich美化日志输出
    
    Args:
        level: 日志级别
        output_dir: 输出目录，如果提供则同时将日志输出到文件
    """
    handlers = [RichHandler(console=console, rich_tracebacks=True)]
    
    # 如果提供了输出目录，添加文件处理程序
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
    
    # 返回logger实例
    return logging.getLogger("ledpo_trainer_rich")

def print_section(title):
    """打印带分隔线的段落标题，使用Rich面板"""
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", border_style="cyan", expand=False))

def print_config_table(config):
    """以表格形式打印配置信息"""
    table = Table(title="配置参数", show_header=True, header_style="bold magenta")
    table.add_column("参数", style="dim")
    table.add_column("值", style="green")
    
    for section, values in config.items():
        if isinstance(values, dict):
            for k, v in values.items():
                table.add_row(f"{section}.{k}", str(v))
        else:
            table.add_row(section, str(values))
    
    console.print(table)

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件并以美化方式显示
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    print_section(f"加载配置文件: {config_path}")
    
    with console.status("[bold green]读取配置文件...[/bold green]"):
        # 读取YAML文件内容
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 使用语法高亮显示YAML文件内容
    yaml_str = yaml.dump(config, default_flow_style=False)
    # Rich的Syntax类可能不支持title参数，使用Panel包装Syntax对象
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="配置文件内容", border_style="green"))
    
    return config

def process_args(config_path: str) -> Tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]:
    """
    处理配置文件并返回参数对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        解析后的参数对象元组
    """
    with console.status("[bold green]处理配置参数...[/bold green]") as status:
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
    
    # 打印关键参数表格
    print_section("训练参数概览")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("模型", model_args.model_name_or_path)
    table.add_row("数据集", str(data_args.dataset))
    table.add_row("训练阶段", finetuning_args.stage)
    table.add_row("微调类型", finetuning_args.finetuning_type)
    table.add_row("偏好beta值", str(finetuning_args.pref_beta))
    table.add_row("批处理大小", str(training_args.per_device_train_batch_size))
    table.add_row("学习率", str(training_args.learning_rate))
    table.add_row("训练轮次", str(training_args.num_train_epochs))
    table.add_row("输出目录", training_args.output_dir)
    
    console.print(table)
    
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
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # 加载tokenizer任务
        tokenizer_task = progress.add_task("[green]加载Tokenizer...", total=1)
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        progress.update(tokenizer_task, completed=1)
        
        # 获取模板任务
        template_task = progress.add_task("[yellow]获取模板...", total=1)
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        progress.update(template_task, completed=1)
        
        # 加载模型任务
        model_task = progress.add_task(f"[blue]加载模型: {model_args.model_name_or_path}...", total=1)
        model = load_model(tokenizer, model_args, finetuning_args, do_train)
        progress.update(model_task, completed=1)
        
        # 创建参考模型任务
        ref_model_task = progress.add_task("[magenta]准备参考模型...", total=1)
        if finetuning_args.use_ref_model: # 如果使用参考模型
            if finetuning_args.ref_model is None and (not do_train): # 如果参考模型为None且不进行训练
                ref_model = model # 参考模型与主模型相同
            else: # 否则创建参考模型
                ref_model = create_ref_model(model_args, finetuning_args)
        else: # 否则参考模型为None
            ref_model = None
        progress.update(ref_model_task, completed=1)
    
    # 打印模型信息
    model_info = Table(title="模型信息", show_header=True, header_style="bold blue")
    model_info.add_column("组件", style="cyan")
    model_info.add_column("类型/值", style="green")
    
    model_info.add_row("Tokenizer", tokenizer.__class__.__name__)
    model_info.add_row("模板", data_args.template)
    model_info.add_row("模型", model.__class__.__name__)
    model_info.add_row("参考模型", "同主模型" if id(model) == id(ref_model) else ("无" if ref_model is None else ref_model.__class__.__name__))
    
    console.print(model_info)
    
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
    
    with console.status("[bold green]加载数据集...[/bold green]") as status:
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
    dataset_info = Table(title="数据集信息", show_header=True, header_style="bold green")
    dataset_info.add_column("数据集", style="cyan")
    dataset_info.add_column("样本数", style="green")
    
    if "train_dataset" in dataset_module:
        train_size = len(dataset_module["train_dataset"])
        dataset_info.add_row("训练集", str(train_size))
        
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"] is not None:
        eval_size = len(dataset_module["eval_dataset"])
        dataset_info.add_row("验证集", str(eval_size))
    
    console.print(dataset_info)
    
    # 如果有训练集，显示样本示例
    if "train_dataset" in dataset_module and len(dataset_module["train_dataset"]) > 0:
        console.print("[bold]训练样本示例:[/bold]")
        sample = dataset_module["train_dataset"][0]
        
        sample_table = Table(show_header=True, header_style="bold")
        sample_table.add_column("字段", style="cyan")
        sample_table.add_column("值", style="green")
        
        for k, v in sample.items():
            if k.endswith("_ids") or k.endswith("_mask"):
                sample_table.add_row(k, f"[张量, 长度={len(v)}]")
            else:
                sample_table.add_row(k, str(v))
        
        console.print(sample_table)
    
    # 创建数据整理器
    tokenizer = tokenizer_module["tokenizer"]
    model = dataset_module.get("model", None)
    
    with console.status("[bold green]创建数据整理器...[/bold green]"):
        data_collator = PairwiseDataCollatorWithPadding(
            template=template,
            model=model,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )
        console.print("[bold green]✓[/bold green] 创建了PairwiseDataCollatorWithPadding数据整理器")
    
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
    
    with console.status("[bold green]初始化DPO训练器...[/bold green]"):
        # 初始化DPO训练器
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
    
    console.print(f"[bold green]✓[/bold green] 创建了LEDPOTrainer，训练设备: [cyan]{training_args.device}[/cyan]")
    
    # 显示优化器参数
    if hasattr(trainer, "optimizer") and trainer.optimizer:
        console.print("\n[bold]优化器参数组:[/bold]")
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            console.print(f"[cyan]参数组 {i}:[/cyan] {len(param_group['params'])} 个参数, 学习率: {param_group['lr']}")
    
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
        console.print("[bold green]开始训练...[/bold green]")
        
        # 显示优化器参数
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            console.print("\n[bold]优化器参数组:[/bold]")
            for i, param_group in enumerate(trainer.optimizer.param_groups):
                console.print(f"[cyan]参数组 {i}:[/cyan] {len(param_group['params'])} 个参数, 学习率: {param_group['lr']}")
        
        # 执行训练
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        console.print("\n[bold green]保存模型...[/bold green]")
        trainer.save_model()
        trainer.save_state()
        
        # 记录指标,通过 callbacks 回调函数, 写入wandb
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # 添加有效token处理速度指标
        if finetuning_args.include_effective_tokens_per_second and dataset_module and "train_dataset" in dataset_module:
            metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], metrics, stage="rm"
            )
            console.print(f"[bold]有效token处理速度:[/bold] [cyan]{metrics['effective_tokens_per_sec']:.2f}[/cyan] tokens/sec")
        
        # 绘制损失曲线
        if finetuning_args.plot_loss:
            with console.status("[bold green]绘制损失曲线...[/bold green]"):
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
        
        # 显示训练结果表格
        result_table = Table(title="训练结果", show_header=True, header_style="bold green")
        result_table.add_column("指标", style="cyan")
        result_table.add_column("值", style="green")
        
        result_table.add_row("训练步数", str(train_steps or "未能计算"))
        result_table.add_row("总训练时间", f"{metrics.get('train_runtime', '未知')} 秒")
        
        if metrics.get("train_steps_per_second", 0) > 0:
            result_table.add_row("平均每步时间", f"{1.0/metrics.get('train_steps_per_second')} 秒")
        else:
            result_table.add_row("平均每步时间", "未知")
            
        # 添加其他重要指标
        for key in metrics:
            if key.startswith("train_") or key in ["train_runtime", "train_samples_per_second", "train_steps_per_second"]:
                continue
            result_table.add_row(key, str(metrics[key]))
        
        console.print(result_table)
    
    # 评估
    if training_args.do_eval:
        console.print("\n[bold green]开始评估...[/bold green]")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
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
        
        # 显示评估结果表格
        eval_table = Table(title="评估结果", show_header=True, header_style="bold blue")
        eval_table.add_column("指标", style="cyan")
        eval_table.add_column("值", style="green")
        
        for key, value in metrics.items():
            eval_table.add_row(key, str(value))
        
        console.print(eval_table)

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
    # 确保logger在异常处理前定义
    logger = setup_logging()
    
    try:
        # 显示启动信息
        console.print(Panel(
            "[bold yellow]LLaMA-Factory LEDPO训练[/bold yellow]\n"
            "[cyan]Rich UI版本[/cyan]", 
            title="启动工作流程", 
            border_style="green"
        ))
        
        # 处理参数
        model_args, data_args, training_args, finetuning_args, generating_args = process_args(config_path)
        
        # 检查是否为DPO/LEDPO训练
        if finetuning_args.stage not in ["dpo", "ledpo"]:
            raise ValueError(f"配置文件指定的训练阶段不是DPO或LEDPO，而是: {finetuning_args.stage}")
        
        # 确保输出目录存在
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # 使用输出目录重新设置日志
        logger = setup_logging(output_dir=training_args.output_dir)
        logger.info(f"输出目录: {training_args.output_dir}")
        
        # 打印报告工具信息
        logger.info(f"报告工具: {training_args.report_to}")
        
        # 也可以通过环境变量设置WANDB项目
        if not hasattr(training_args, 'wandb_project') or not training_args.wandb_project:
            os.environ["WANDB_PROJECT"] = "ledpo_full"
            logger.info("通过环境变量设置WANDB项目名称为: ledpo_full")

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
        
        console.print(Panel(
            "[bold green]DPO训练流程全部完成！[/bold green]", 
            border_style="green"
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[bold red]错误: {str(e)}[/bold red]", 
            title="训练失败", 
            border_style="red"
        ))
        logger.error(f"DPO训练过程中出错: {str(e)}")
        raise

def main():
    """
    主函数，处理命令行参数并启动训练
    """
    logger = setup_logging()
    
    # 显示欢迎信息
    console.print(Panel(
        "[bold yellow]LLaMA-Factory LEDPO训练工具[/bold yellow]\n"
        "[cyan]Rich UI版本 - 美化终端输出[/cyan]", 
        border_style="green"
    ))
    
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
        console.print(f"[bold red]错误: 配置文件不存在: {config_path}[/bold red]")
        sys.exit(1)
    
    # 加载配置预览
    _ = load_config_file(config_path)
    
    # 显示启动训练信息
    console.print("\n[bold green]正在启动LEDPO训练流程...[/bold green]")
    time.sleep(1)  # 为了视觉效果的短暂停顿
    
    # 执行DPO训练
    run_dpo_workflow(config_path)

if __name__ == "__main__":
    main() 


# 运行命令
# python dpo_baseline/run_ledpo_rich.py examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml

# 使用llamafactory-cli训练
# llamafactory-cli train examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml 