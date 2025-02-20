"""
测试 DPO 的前向传播和损失计算
"""
import os
import sys
import yaml
import torch
import logging
import time
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from typing import Optional, List

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llamafactory.data import PairwiseDataCollatorWithPadding
from src.llamafactory.extras.constants import IGNORE_INDEX
from src.llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments
)
from src.llamafactory.train.dpo.trainer import CustomDPOTrainer
from src.llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from src.llamafactory.data.processor import PairwiseDatasetProcessor
from src.llamafactory.model import load_model, load_tokenizer, load_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timer:
    """简单的计时器类"""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        rprint(f"[yellow]⏱️ {self.name} 耗时: {duration:.2f} 秒[/yellow]")

def print_section(title: str, content: str = None):
    """打印带格式的段落标题和内容"""
    rprint(Panel(f"[bold cyan]{title}[/bold cyan]"))
    if content:
        rprint(content)
    rprint("")

def test_forward_pass():
    """测试前向传播和损失计算的完整流程"""
    total_start_time = time.time()
    print_section("DPO Forward Pass Testing")
    
    # 1. 加载配置
    print_section("1. 配置加载")
    with Timer("配置加载"):
        parser = HfArgumentParser((
            ModelArguments,
            DataArguments,
            Seq2SeqTrainingArguments,
            FinetuningArguments
        ))
        
        config_file = os.path.join(os.path.dirname(__file__), "configs", "test_dpo_qwen_0.5b.yaml")
        rprint(f"[yellow]正在加载配置文件: {config_file}[/yellow]")
        model_args, data_args, training_args, finetuning_args = parser.parse_yaml_file(config_file)
        
        # 打印配置信息
        config_table = Table(title="测试配置", show_header=True, header_style="bold magenta")
        config_table.add_column("参数", style="cyan")
        config_table.add_column("值", style="green")
        
        config_table.add_row("模型路径", model_args.model_name_or_path)
        config_table.add_row("Batch Size", str(training_args.per_device_train_batch_size))
        config_table.add_row("Loss Type", str(finetuning_args.pref_loss))
        
        rprint(config_table)
    
    # 2. 加载模型和分词器
    print_section("2. 模型初始化")
    with Timer("模型和分词器加载"):
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    
    # 3. 加载和预处理数据
    print_section("3. 数据加载")
    with Timer("数据集加载和预处理"):
        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="rm",  # 对于 DPO 数据，使用 rm 阶段
            **tokenizer_module
        )
        train_dataset = dataset_module["train_dataset"]
    
    # 4. 创建训练器
    print_section("4. 训练器初始化")
    with Timer("训练器初始化"):
        # 1) 创建数据整理器
        data_collator = PairwiseDataCollatorWithPadding(
            template=template,
            model=model,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )
        
        # 设置 remove_unused_columns=False，这对于多模态和成对数据集很重要
        training_args.remove_unused_columns = False
        
        # 2) 初始化训练器
        trainer = CustomDPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            **dataset_module,
            **tokenizer_module,
        )
    
    # 5. 获取一个批次的数据
    print_section("5. 批次数据准备")
    with Timer("批次数据准备"):
        dataloader = trainer.get_train_dataloader()
        batch = next(iter(dataloader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 打印批次数据的信息
        batch_info_table = Table(title="批次数据信息", show_header=True, header_style="bold magenta")
        batch_info_table.add_column("键", style="cyan")
        batch_info_table.add_column("形状", style="yellow")
        batch_info_table.add_column("类型", style="green")
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_info_table.add_row(k, str(v.shape), str(v.dtype))
            else:
                batch_info_table.add_row(k, str(type(v)), str(type(v)))
        
        rprint(batch_info_table)
        
        # 检查分词器中的特殊token
        token_table = Table(title="特殊Token信息", show_header=True, header_style="bold magenta")
        token_table.add_column("Token", style="cyan")
        token_table.add_column("ID", style="yellow")
        
        special_tokens = [
            "<|im_start|>assistant",
            "<|im_start|>",
            "assistant",
            "<|im_end|>",
            "<s>",
            "</s>"
        ]
        
        for token in special_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                token_table.add_row(token, str(token_id))
            except:
                token_table.add_row(token, "未找到")
        
        rprint(token_table)
    
    # 6. 测试前向传播
    print_section("6. 前向传播测试")
    with Timer("前向传播计算"):
        try:
            outputs = trainer.concatenated_forward(model, batch)
            rprint("[green]前向传播成功![/green]")
            
            # 修改输出解包,移除 betas 和 ppls
            chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps_avg = outputs
            
            output_table = Table(title="前向传播输出", show_header=True, header_style="bold magenta")
            output_table.add_column("张量", style="cyan")
            output_table.add_column("形状", style="yellow")
            output_table.add_column("均值", style="green")
            
            # 移除 betas 和 ppls 相关的行
            output_table.add_row("chosen_logps", str(chosen_logps.shape), f"{chosen_logps.mean().item():.4f}")
            output_table.add_row("rejected_logps", str(rejected_logps.shape), f"{rejected_logps.mean().item():.4f}")
            output_table.add_row("chosen_logits", str(chosen_logits.shape), f"{chosen_logits.mean().item():.4f}")
            output_table.add_row("rejected_logits", str(rejected_logits.shape), f"{rejected_logits.mean().item():.4f}")
            output_table.add_row("chosen_logps_avg", str(chosen_logps_avg.shape), f"{chosen_logps_avg.mean().item():.4f}")
            
            rprint(output_table)
            
        except Exception as e:
            rprint(f"[red]前向传播失败:[/red]")
            rprint(f"[red]{str(e)}[/red]")
            raise
    
    # 7. 测试损失计算
    print_section("7. 损失计算测试")
    with Timer("损失计算"):
        try:
            loss, metrics = trainer.get_batch_loss_metrics(model, batch)
            rprint("[green]损失计算成功![/green]")
            
            metrics_table = Table(title="训练指标", show_header=True, header_style="bold magenta")
            metrics_table.add_column("指标", style="cyan")
            metrics_table.add_column("值", style="green")
            
            metrics_table.add_row("loss", f"{loss.item():.4f}")
            for key, value in metrics.items():
                metrics_table.add_row(key, f"{value:.4f}")
            
            rprint(metrics_table)
                
        except Exception as e:
            rprint(f"[red]损失计算失败:[/red]")
            rprint(f"[red]{str(e)}[/red]")
            raise
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print_section(f"测试完成! 总耗时: {total_duration:.2f} 秒")
    return loss, metrics

if __name__ == "__main__":
    # 运行测试
    loss, metrics = test_forward_pass() 