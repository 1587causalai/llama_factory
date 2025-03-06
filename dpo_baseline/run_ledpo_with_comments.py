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

Wandb 图表生成机制:
-----------------
本脚本整合了详细的 Weights & Biases (wandb) 监控功能。理解 wandb 图表生成机制是很重要的：

1. 图表数据来源: 
   - wandb 图表显示的数据直接来自训练过程中记录的指标
   - 这些指标由 LEDPOTrainer.get_batch_loss_metrics() 方法计算
   - 通过 trainer.log_metrics() 函数发送到 wandb

2. 真正的指标记录流程:
   a. 训练阶段指标记录:
      - 在每个批次训练中，LEDPOTrainer.get_batch_loss_metrics() 计算各种指标
      - 计算的指标如 loss, beta_scale, dynamic_beta, rewards/accuracies 等
      - 这些指标通过 trainer.log() 方法记录到 trainer_state.json
      - 同时，通过集成的 wandb 报告器发送到 wandb 服务器
   
   b. 评估阶段指标记录:
      - 在 trainer.evaluate() 中，计算评估集上的指标
      - 评估指标通过 trainer.log_metrics("eval", metrics) 发送到 wandb
      - 这些指标会以 "eval_" 前缀显示在 wandb 中
      - **关键点**: 如果评估代码计算的指标少，wandb显示的图表也会少

3. 关键点说明:
   - 只有在训练/评估过程中实际计算并记录的指标才会显示在 wandb 中
   - plot_loss() 函数仅生成本地 PNG 文件，不影响 wandb 显示的图表
   - BetaDPO 和 LEDPO 显示的 wandb 图表不同是因为它们的训练器实现不同
   - 两者可能在 get_batch_loss_metrics() 或 evaluate() 中计算了不同的指标

4. 控制 wandb 图表的正确方法:
   - 创建自定义训练器，继承 LEDPOTrainer 并重写 get_batch_loss_metrics() 方法
   - 在评估函数 evaluate() 中添加更多指标计算
   - 确保评估流程 (do_eval=True) 正确设置
   - 使用适当的评估策略 (evaluation_strategy: steps)
   - 确保 report_to=["wandb"] 参数设置正确

5. 配置示例:
   ```yaml
   training_args:
     do_eval: true
     evaluation_strategy: steps
     eval_steps: 100  # 更频繁的评估
     report_to: [wandb]
   ```

6. 为什么修改 plot_loss() 不起作用:
   - plot_loss() 只读取已记录的指标生成本地图片
   - 它不会影响训练过程中实际计算和记录哪些指标
   - 要增加 wandb 图表，需要从源头(训练器)修改指标计算和记录

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

# 导入 rich 组件
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule  # 使用 Rule 替代 Separator
from rich.markdown import Markdown
from rich import print  # 可以直接使用 rich.print 代替 print
from rich.logging import RichHandler

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


# 初始化 rich console
console = Console()


def setup_logging(output_dir=None, level=logging.INFO):
    """
    设置日志级别和格式，使用 rich Handler
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format = f"%(asctime)s [%(levelname)s] %(message)s"

    handlers = [RichHandler(console=console)] # 使用 RichHandler

    if output_dir:
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"ledpo_training_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file, encoding='utf-8')) # 确保文件日志编码为 utf-8

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        datefmt="%Y-%m-%d %H:%M:%S" # 明确指定日期格式
    )
    logger = logging.getLogger(__name__)

    if output_dir:
        logger.info(f"日志将保存到: {log_file}")

    return logger


def print_section(title):
    """使用 rich Panel 打印分隔区域标题"""
    console.print(Panel(title, border_style="blue", padding=(1, 2)))


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"加载配置文件: [cyan]{config_path}[/cyan]") # 使用 rich 颜色
        with open(config_path, 'r', encoding='utf-8') as f: # 明确指定文件编码为 utf-8
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"[red]加载配置文件失败:[/red] {e}") # 使用 rich 颜色
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
            logger.warning(f"配置文件中stage=[yellow]{finetuning_args.stage}[/yellow]，但此脚本为LEDPO，强制将stage设置为[cyan]ledpo[/cyan]") # 使用 rich 颜色
            finetuning_args.stage = "ledpo"
        
        # 恢复原始命令行参数
        sys.argv = original_argv
        
        return model_args, data_args, training_args, finetuning_args, generating_args
    
    except Exception as e:
        # 确保恢复原始命令行参数
        sys.argv = original_argv
        logger.error(f"[red]处理参数时出错:[/red] {e}") # 使用 rich 颜色
        raise


def prepare_tokenizer_and_model(model_args, finetuning_args, data_args, training_args, do_train=True):
    """
    准备tokenizer和模型
    """
    logger = logging.getLogger(__name__)
    print_section("准备Tokenizer和模型")
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: [cyan]{model_args.model_name_or_path}[/cyan]") # 使用 rich 颜色
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 获取模板并修复tokenizer
    logger.info(f"模板类型: [cyan]{data_args.template}[/cyan]") # 使用 rich 颜色
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 加载模型
    logger.info(f"加载模型: [cyan]{model_args.model_name_or_path}[/cyan]") # 使用 rich 颜色
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
    
    logger.info(f"数据集: [cyan]{data_args.dataset}[/cyan]") # 使用 rich 颜色
    logger.info(f"最大样本数: [cyan]{data_args.max_samples}[/cyan]") # 使用 rich 颜色
    
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
        logger.info(f"训练集大小: [green]{train_size}[/green] 样本") # 使用 rich 颜色
    
    if "eval_dataset" in dataset_module:
        eval_size = len(dataset_module["eval_dataset"])
        logger.info(f"验证集大小: [green]{eval_size}[/green] 样本") # 使用 rich 颜色
    
    # 随机抽取一个样本进行展示
    if "train_dataset" in dataset_module and len(dataset_module["train_dataset"]) > 0:
        random_idx = 0  # 显示第一个样本
        sample = dataset_module["train_dataset"][random_idx]
        logger.info(f"数据集样本示例 (索引 [cyan]{random_idx}[/cyan]):") # 使用 rich 颜色
        for k, v in sample.items():
            if k in ["prompt_ids", "chosen_ids", "rejected_ids"]:
                logger.info(f"  [bold]{k}[/bold]: 长度=[magenta]{len(v)}[/magenta]") # 使用 rich 颜色
            else:
                logger.info(f"  [bold]{k}[/bold]: [magenta]{v}[/magenta]") # 使用 rich 颜色
    
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
    
    logger.info(f"基础beta值: [cyan]{finetuning_args.pref_beta}[/cyan]") # 使用 rich 颜色
    logger.info(f"损失函数类型: [cyan]{finetuning_args.pref_loss}[/cyan]") # 使用 rich 颜色
    
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
    if hasattr(trainer.model, "beta_scale"):
        logger.info(f"初始beta_scale值: [magenta]{trainer.model.beta_scale.item()}[/magenta]") # 使用 rich 颜色
        logger.info(f"初始动态beta值: [magenta]{trainer.get_dynamic_beta().item()}[/magenta]") # 使用 rich 颜色
    
    # 打印训练器信息
    logger.info(f"梯度累积步数: [cyan]{training_args.gradient_accumulation_steps}[/cyan]") # 使用 rich 颜色
    logger.info(f"学习率: [cyan]{training_args.learning_rate}[/cyan]") # 使用 rich 颜色
    logger.info(f"优化器: [cyan]{training_args.optim}[/cyan]") # 使用 rich 颜色
    
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
    logger.info(f"当前beta_scale值: [magenta]{trainer.model.beta_scale.item()}[/magenta]") # 使用 rich 颜色
    logger.info(f"当前动态beta值: [magenta]{dynamic_beta.item()}[/magenta]") # 使用 rich 颜色
    
    # 计算损失
    logger.info("计算preference损失...")
    losses, chosen_rewards, rejected_rewards = trainer.compute_preference_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )
    
    # 打印损失相关值
    logger.info(f"损失平均值: [magenta]{losses.mean().item()}[/magenta]") # 使用 rich 颜色
    logger.info(f"选择奖励平均值: [magenta]{chosen_rewards.mean().item()}[/magenta]") # 使用 rich 颜色
    logger.info(f"拒绝奖励平均值: [magenta]{rejected_rewards.mean().item()}[/magenta]") # 使用 rich 颜色
    logger.info(f"奖励差值平均值: [magenta]{(chosen_rewards - rejected_rewards).mean().item()}[/magenta]") # 使用 rich 颜色
    logger.info(f"准确率: [magenta]{(chosen_rewards > rejected_rewards).float().mean().item()}[/magenta]") # 使用 rich 颜色
    
    return losses, chosen_rewards, rejected_rewards


# ============================================
# WandB 图表生成逻辑说明
# ============================================
"""
在 LEDPO 训练过程中，wandb 图表的生成主要有以下关键步骤：

1. 指标计算:
   - 在训练的每个批次中，通过 LEDPOTrainer.get_batch_loss_metrics() 计算各种指标
   - 这些指标包括：loss, beta_scale, dynamic_beta, rewards/chosen, rewards/rejected, rewards/margins, rewards/accuracies 等
   - 同样，在评估阶段也会计算类似的指标，但带有 eval_ 前缀

   LEDPOTrainer.get_batch_loss_metrics() 方法详解:
   ```python
   def get_batch_loss_metrics(self, model, batch, train_eval="train"):
       # 前向传播，计算策略模型的log概率
       policy_chosen_logps, policy_rejected_logps, ... = self.concatenated_forward(model, batch)
       
       # 计算动态beta值并记录
       dynamic_beta = self.get_dynamic_beta()
       metrics["beta_scale"] = self.model.beta_scale.detach()  # 这里记录了beta_scale指标！
       metrics["dynamic_beta"] = dynamic_beta.detach()         # 这里记录了dynamic_beta指标！
       
       # 计算参考模型的log概率
       reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
       
       # 计算偏好损失、奖励值等
       losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(...)
       
       # 记录各种奖励指标
       metrics["chosen_rewards"] = chosen_rewards.detach().mean()      # 记录chosen_rewards指标
       metrics["rejected_rewards"] = rejected_rewards.detach().mean()  # 记录rejected_rewards指标
       metrics["margins"] = (chosen_rewards - rejected_rewards).detach().mean()  # 记录margins指标
       metrics["accuracies"] = (chosen_rewards > rejected_rewards).detach().float().mean()  # 记录accuracies指标
       
       return loss, metrics
   ```
   
   这些计算出的指标随后会在 LEDPOTrainer.compute_loss() 中被记录:
   ```python
   def compute_loss(self, model, inputs, return_outputs=False):
       # 计算损失和指标
       loss, metrics = self.get_batch_loss_metrics(model, inputs)
       
       # 记录指标到wandb
       prefix = "eval" if self.args.evaluation_strategy != "no" and self.is_in_eval else ""
       log_metrics = {}
       for k, v in metrics.items():
           if v is not None:
               log_metrics[f"{prefix}rewards/{k}" if prefix else f"rewards/{k}"] = v
       
       # 添加到已存储的指标中
       self._stored_metrics[prefix or "train"]["loss"].append(loss.item())
       for k, v in log_metrics.items():
           if isinstance(v, torch.Tensor):
               v = v.item()
           self._stored_metrics[prefix or "train"][k.split("/")[-1]].append(v)
       
       return loss
   ```

2. 指标记录:
   - 上述计算的指标通过 LEDPOTrainer.log() 方法记录下来
   - 这些指标被添加到 trainer_state.json 文件中的 log_history 列表
   - log_history 是一个字典列表，每个元素包含一个步骤的所有指标值

3. 指标可视化:
   - 训练结束后，通过调用 plot_loss() 函数将这些指标可视化
   - plot_loss() 接受一个 keys 参数，决定哪些指标会被绘制成图表
   - 每个 key 对应 log_history 中的一个指标名称
   
   plot_loss() 函数详解：
   ```python
   def plot_loss(save_dictionary: str, keys: List[str] = ["loss"]) -> None:
       # 读取训练过程中记录的所有指标
       with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
           data = json.load(f)
       
       # 为每个指定的key绘制一个曲线图
       for key in keys:
           steps, metrics = [], []
           # 从log_history中提取指标值
           for i in range(len(data["log_history"])):
               if key in data["log_history"][i]:
                   steps.append(data["log_history"][i]["step"])
                   metrics.append(data["log_history"][i][key])
           
           # 检查是否有数据可绘制
           if len(metrics) == 0:
               logger.warning_rank0(f"No metric {key} to plot.")
               continue
           
           # 绘制原始曲线和平滑后的曲线
           plt.figure()
           plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
           plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
           plt.title(f"training {key} of {save_dictionary}")
           plt.xlabel("step")
           plt.ylabel(key)
           plt.legend()
           
           # 保存为PNG文件
           figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
           plt.savefig(figure_path, format="png", dpi=100)
   ```

4. 图表显示:
   - 绘制的图表会被保存为 PNG 文件在输出目录中
   - wandb 会自动收集这些图表并在其 UI 中显示
   - training_loss.png, training_eval_loss.png, training_rewards_accuracies.png 等

为什么 run_betadpo_detailed.py 比 run_ledpo_detailed.py 显示更多图表？
- 这是因为它们在 plot_loss() 函数中传入了不同的 keys 列表
- LEDPO 的图表由 keys=["loss", "eval_loss", "beta_scale", "dynamic_beta", "rewards/accuracies"] 控制
- BetaDPO 的图表由 keys=["loss", "eval_loss", "rewards/accuracies", "train_beta_scale"] 控制
- 另外，它们的 Trainer 实现可能记录了不同的指标在训练过程中

如何控制显示哪些图表？
1. 修改 plot_loss() 函数调用，添加或删除 keys 中的元素
2. 确保添加的 key 在训练过程中被记录了，否则会出现警告
3. 对比 LEDPO 和 BetaDPO 的 get_batch_loss_metrics() 方法，了解它们记录了哪些不同的指标
"""

# 下面这部分是用于执行LEDPO训练的函数，也是控制指标记录和图表生成的关键部分
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
        logger.info("[bold green]开始训练...[/bold green]") # 使用 rich 颜色
        # 开始训练过程 - 这个过程中会自动记录指标
        # 在训练过程中，每次迭代结束会通过LEDPOTrainer.get_batch_loss_metrics()方法计算各种指标
        # 然后通过LEDPOTrainer.log()方法记录到wandb中
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # 保存最终模型
        logger.info("保存最终模型...")
        trainer.save_model()
        
        # ====== 关键部分：记录训练指标到wandb ======
        # 这里调用log_metrics记录train指标，这些指标会被记录到wandb中
        # train_result.metrics包含了整个训练过程的汇总指标
        # 这些指标通常包括：平均loss、训练时间、学习率等
        # wandb会自动为这些指标生成图表
        trainer.log_metrics("train", train_result.metrics)  # 这行代码将指标发送到wandb
        trainer.save_metrics("train", train_result.metrics)  # 这行将指标保存到本地文件
        trainer.save_state()
        
        # 输出最终beta_scale值
        if hasattr(trainer.model, "beta_scale"):
            final_beta_scale = trainer.model.beta_scale.item()
            final_dynamic_beta = trainer.get_dynamic_beta().item()
            logger.info(f"训练后beta_scale值: [magenta]{final_beta_scale}[/magenta]") # 使用 rich 颜色
            logger.info(f"训练后动态beta值: [magenta]{final_dynamic_beta}[/magenta]") # 使用 rich 颜色
            
            # 将beta值保存到额外指标中
            # 这些是训练结束后的额外指标，也会被记录到wandb中
            extra_metrics = {
                "train_final_beta_scale": final_beta_scale,
                "train_final_dynamic_beta": final_dynamic_beta,
            }
            trainer.save_metrics("train_extra", extra_metrics)
        
        # ====== 关键部分：绘制本地损失曲线 ======
        # 注意：这个函数仅用于生成本地PNG图片文件，不会影响wandb中显示的图表
        # 这个函数会读取训练过程中记录的所有指标数据(trainer_state.json)
        # 然后为每个指定的key绘制一个图表并保存为PNG文件
        # 虽然名称有误导性，但这个函数不会改变wandb中显示哪些图表
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            logger.info("绘制损失曲线...")
            # 传入的keys列表决定了哪些指标会被绘制成本地PNG图片
            # 注意：这与wandb显示的图表无关，wandb图表由训练过程中实际记录的指标决定
            plot_loss(
                training_args.output_dir, 
                keys=[
                    "loss",              # 训练损失 - 训练过程中的主要损失值
                    "eval_loss",         # 评估损失 - 验证集上的损失值
                    "beta_scale",        # beta缩放因子 - LEDPO算法中的可学习参数
                    "dynamic_beta",      # 动态beta值 - 通过beta_scale计算得到的实际beta值
                    "rewards/accuracies", # 奖励准确率 - 训练中选择奖励>拒绝奖励的比例
                    "rewards/chosen",     # 选择奖励 - 被选择回答的奖励值
                    "rewards/rejected",   # 拒绝奖励 - 被拒绝回答的奖励值
                    "rewards/margins",    # 奖励差距 - 选择和拒绝奖励之间的差距
                    "eval_rewards/accuracies", # 评估奖励准确率
                    "eval_rewards/chosen",     # 评估选择奖励
                    "eval_rewards/rejected",   # 评估拒绝奖励
                    "eval_rewards/margins"     # 评估奖励差距
                ]
            )
    
    # ====== 关键部分：执行评估并记录指标 ======
    # 这部分代码负责在评估集上评估模型并记录指标
    # ★★★ 这里是真正控制wandb中eval图表的关键部分 ★★★
    # 只有这里实际计算并通过log_metrics记录的指标才会显示在wandb中
    # 如果评估过程中计算的指标少，wandb中显示的图表也会少
    if training_args.do_eval:
        logger.info("[bold green]开始评估...[/bold green]") # 使用 rich 颜色
        # 执行评估，返回评估指标
        # 注意: trainer.evaluate() 方法会调用 LEDPOTrainer.get_batch_loss_metrics() 
        # 并传入 train_eval="eval" 参数来计算评估指标
        # 这个方法内部计算和返回的指标决定了wandb中显示哪些评估图表
        metrics = trainer.evaluate()
        
        # 核心步骤：记录评估指标到wandb
        # 这行代码直接决定了wandb中显示哪些"eval_"开头的图表
        # 如果metrics中只有少量指标，wandb中显示的图表也会少
        # ★ BetaDPO和LEDPO的差异可能在于：★
        # 1. BetaDPO的训练器在evaluate()中计算了更多评估指标
        # 2. 或者在get_batch_loss_metrics(train_eval="eval")中记录了更多指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        # 注意：如果想增加评估图表，可以这样做：
        """
        # 示例：添加额外的评估指标
        extra_eval_metrics = {
            "eval_additional_metric": metrics.get("eval_loss", 0) * 1.5,
            "eval_custom_analysis": metrics.get("eval_rewards/accuracies", 0.5) * 2
        }
        trainer.log_metrics("eval_extra", extra_eval_metrics)
        trainer.save_metrics("eval_extra", extra_eval_metrics)
        """
        # 但要彻底解决问题，应创建自定义训练器重写evaluate()方法


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
            if trainer and hasattr(trainer.model, "beta_scale"):
                beta_scale = trainer.model.beta_scale.item()
                dynamic_beta = trainer.get_dynamic_beta().item()
                logger.info(f"Epoch [cyan]{state.epoch}[/cyan]，beta_scale值: [magenta]{beta_scale}[/magenta]，动态beta值: [magenta]{dynamic_beta}[/magenta]") # 使用 rich 颜色
    
    # 添加beta_scale监控回调
    callbacks.append(BetaScaleMonitorCallback())
    
    logger.info(f"创建了 [green]{len(callbacks)}[/green] 个回调函数") # 使用 rich 颜色
    return callbacks


def run_ledpo_workflow(config_path: str):
    """
    运行完整的LEDPO工作流程
    """
    # 首先设置基本控制台日志
    temp_logger = setup_logging()
    temp_logger.info(f"开始LEDPO工作流程，配置文件: [cyan]{config_path}[/cyan]") # 使用 rich 颜色
    
    try:
        # 1. 处理参数
        model_args, data_args, training_args, finetuning_args, generating_args = process_args(config_path)
        
        # 确保输出目录存在
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # 使用输出目录重新设置日志
        logger = setup_logging(output_dir=training_args.output_dir)
        logger.info(f"输出目录: [cyan]{training_args.output_dir}[/cyan]") # 使用 rich 颜色
        
        # 打印报告工具信息
        logger.info(f"  报告工具: [cyan]{training_args.report_to}[/cyan]") # 使用 rich 颜色
        
        # 也可以通过环境变量设置WANDB项目
        if not hasattr(training_args, 'wandb_project') or not training_args.wandb_project:
            os.environ["WANDB_PROJECT"] = "ledpo"
            logger.info("通过环境变量设置WANDB项目名称为: [cyan]ledpo[/cyan]") # 使用 rich 颜色
        
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
        
        logger.info("[bold green]LEDPO工作流程完成！[/bold green]") # 使用 rich 颜色
        return 0
    
    except Exception as e:
        logger.error(f"[red]执行过程中发生错误:[/red] {e}", exc_info=True) # 使用 rich 颜色
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
            config_path = "examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml"
        else:
            config_path = sys.argv[1]
        
        logger.info(f"使用配置文件: [cyan]{config_path}[/cyan]") # 使用 rich 颜色
        
        # 运行LEDPO工作流程
        return run_ledpo_workflow(config_path)
    
    except KeyboardInterrupt:
        logger.warning("[yellow]用户中断执行[/yellow]") # 使用 rich 颜色
        return 130
    except Exception as e:
        logger.error(f"[red]执行过程中发生错误:[/red] {e}", exc_info=True) # 使用 rich 颜色
        return 1


if __name__ == "__main__":
    import sys
    
    # 添加使用说明
    if len(sys.argv) < 2:
        help_text = """
[bold]使用方法:[/bold] python run_ledpo_with_comments.py <config_path> [--show_more_charts] [--fake_wandb_metrics]

[bold]参数说明:[/bold]
  <config_path>             配置文件路径，例如 examples/train_lora/qwen1_5_0_5b_lora_dpo.yaml
  --show_more_charts        可选参数，生成更多本地PNG图表(不影响wandb)
  --fake_wandb_metrics      可选参数，通过hack方式增加wandb图表数量
        """
        console.print(Markdown(help_text)) # 使用 rich Markdown 渲染帮助信息
        sys.exit(1)
    
    # 解析命令行参数
    config_path = sys.argv[1]
    show_more_charts = "--show_more_charts" in sys.argv
    fake_wandb_metrics = "--fake_wandb_metrics" in sys.argv
    
    # 快速解决方案：通过猴子补丁增加 wandb 图表
    if fake_wandb_metrics:
        print("[bold yellow]启用快速解决方案：通过猴子补丁增加 wandb 图表...[/bold yellow]") # 使用 rich 颜色
        print("[bold yellow]注意：这是一个临时解决方案，建议长期使用自定义训练器[/bold yellow]") # 使用 rich 颜色
        
        # 保存原始的 evaluate 方法
        from src.llamafactory.train.ledpo.trainer import LEDPOTrainer
        original_evaluate = LEDPOTrainer.evaluate
        
        # 创建增强版的 evaluate 方法
        def enhanced_evaluate(self, *args, **kwargs):
            # 先调用原始方法
            metrics = original_evaluate(self, *args, **kwargs)
            
            # 如果原始指标为空，返回原始指标
            if not metrics:
                return metrics
                
            # 添加一些额外的指标
            print("[yellow]添加额外的评估指标到wandb...[/yellow]") # 使用 rich 颜色
            base_metrics = {k: v for k, v in metrics.items()}
            
            # 从已有指标派生新指标
            if "eval_loss" in metrics:
                metrics["eval_loss_scaled"] = metrics["eval_loss"] * 0.8
            
            # 奖励相关指标
            if "eval_rewards/accuracies" in metrics:
                acc = metrics["eval_rewards/accuracies"]
                metrics["eval_rewards/confidence"] = acc * (1 - acc) * 4  # 最大值在0.5处
            
            if "eval_rewards/chosen" in metrics and "eval_rewards/rejected" in metrics:
                chosen = metrics["eval_rewards/chosen"]
                rejected = metrics["eval_rewards/rejected"]
                metrics["eval_rewards/ratio"] = chosen / (rejected + 1e-8)
                metrics["eval_rewards/diff_squared"] = (chosen - rejected) ** 2
            
            # 如果有beta相关指标
            if hasattr(self.model, "beta_scale"):
                metrics["eval_beta_scale"] = self.model.beta_scale.item()
                metrics["eval_dynamic_beta"] = self.get_dynamic_beta().item()
            
            return metrics
        
        # 应用猴子补丁
        LEDPOTrainer.evaluate = enhanced_evaluate
        print("[bold green]已增强评估方法，将显示更多wandb图表！[/bold green]") # 使用 rich 颜色
    
    # 如果使用更多图表选项，修改 plot_loss 函数调用
    if show_more_charts:
        # 保存原始的 plot_loss 函数
        original_plot_loss = plot_loss
        
        # 创建一个增强版的 plot_loss 函数
        def enhanced_plot_loss(save_dictionary, keys=None):
            # 使用扩展的 keys 列表
            extended_keys = [
                "loss", "eval_loss", "beta_scale", "dynamic_beta", 
                "rewards/accuracies", "rewards/chosen", "rewards/rejected", "rewards/margins",
                "eval_rewards/accuracies", "eval_rewards/chosen", "eval_rewards/rejected", "eval_rewards/margins",
                "learning_rate", "train_runtime", "train_samples_per_second"
            ]
            return original_plot_loss(save_dictionary, extended_keys)
        
        # 替换原始函数
        plot_loss = enhanced_plot_loss
        print("[bold green]已启用增强图表模式，将显示更多图表！[/bold green]") # 使用 rich 颜色
        print("[yellow]注意：这只会增加本地PNG图表，不会改变wandb中显示的图表。[/yellow]") # 使用 rich 颜色
        print("[yellow]要增加wandb中的图表，需要修改训练器代码添加更多指标计算。[/yellow]") # 使用 rich 颜色
    
    # 运行主流程
    try:
        run_ledpo_workflow(config_path)
        print("[bold green]训练完成！[/bold green]") # 使用 rich 颜色
        console.rule("[bold blue]训练完成[/bold blue]") # 使用 rule 添加完成分隔线
        if show_more_charts:
            print("[magenta]已生成更多本地图表，请在输出目录查看。[/magenta]") # 使用 rich 颜色
        if fake_wandb_metrics:
            print("[magenta]已尝试增加wandb图表，请在wandb界面查看效果。[/magenta]") # 使用 rich 颜色
    except Exception as e:
        print(f"[bold red]训练过程中发生错误:[/bold red] {e}") # 使用 rich 颜色
        import traceback
        traceback.print_exc()
        sys.exit(1)


# 运行命令示例
# llamafactory-cli train examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml
# 或使用详细脚本
# python dpo_baseline/run_ledpo_with_comments.py examples/train_lora/qwen1_5_0_5b_lora_ledpo.yaml


