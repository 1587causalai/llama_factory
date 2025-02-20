"""
测试 Qwen1.5-0.5B-Chat 模型的 DPO 训练流程
"""
import os
import sys
import yaml
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments
)

def print_section(title: str, content: str = None):
    """打印带格式的段落标题和内容"""
    rprint(Panel(f"[bold cyan]{title}[/bold cyan]"))
    if content:
        rprint(content)
    rprint("")

def test_dpo_qwen_0_5b():
    print_section("DPO with Qwen1.5-0.5B-Chat Testing")
    
    # 1. 加载配置
    print_section("1. 配置加载")
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        Seq2SeqTrainingArguments,
        FinetuningArguments
    ))
    
    config_file = os.path.join(os.path.dirname(__file__), "configs", "test_dpo_qwen_0.5b.yaml")
    rprint(f"[yellow]正在加载配置文件: {config_file}[/yellow]")
    model_args, data_args, training_args, finetuning_args = parser.parse_yaml_file(config_file)
    
    # 2. 打印配置信息
    config_table = Table(title="训练配置", show_header=True, header_style="bold magenta")
    config_table.add_column("参数", style="cyan")
    config_table.add_column("值", style="green")
    
    config_table.add_row("模型路径", model_args.model_name_or_path)
    config_table.add_row("Batch Size", str(training_args.per_device_train_batch_size))
    config_table.add_row("梯度累积步数", str(training_args.gradient_accumulation_steps))
    config_table.add_row("学习率", str(training_args.learning_rate))
    config_table.add_row("最大样本数", str(data_args.max_samples))
    config_table.add_row("输出目录", training_args.output_dir)
    config_table.add_row("LoRA Rank", str(finetuning_args.lora_rank))
    config_table.add_row("DPO Beta", str(finetuning_args.pref_beta))
    config_table.add_row("Loss Type", str(finetuning_args.pref_loss))
    
    rprint(config_table)
    
    # 3. 运行 DPO 训练
    print_section("2. DPO 训练开始")
    from src.llamafactory.train.dpo.workflow import run_dpo
    
    try:
        run_dpo(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args
        )
        rprint("[green]DPO 训练完成![/green]")
    except Exception as e:
        rprint(f"[red]训练过程中发生错误:[/red]")
        rprint(f"[red]{str(e)}[/red]")
        import traceback
        rprint(traceback.format_exc())
        return
    
    # 4. 检查训练输出
    print_section("3. 训练输出检查")
    output_table = Table(title="输出文件检查", show_header=True, header_style="bold magenta")
    output_table.add_column("文件类型", style="cyan")
    output_table.add_column("状态", style="green")
    output_table.add_column("路径", style="yellow")
    
    # 检查 loss 图表
    loss_png = os.path.join(training_args.output_dir, "loss.png")
    output_table.add_row(
        "Loss 图表",
        "[green]已生成[/green]" if os.path.exists(loss_png) else "[red]未生成[/red]",
        loss_png
    )
    
    # 检查 LoRA adapter
    adapter_path = os.path.join(training_args.output_dir, "adapter_model")
    output_table.add_row(
        "LoRA Adapter",
        "[green]已保存[/green]" if os.path.exists(adapter_path) else "[red]未保存[/red]",
        adapter_path
    )
    
    rprint(output_table)
    print_section("测试完成")

if __name__ == "__main__":
    test_dpo_qwen_0_5b() 