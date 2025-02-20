"""
测试 DPO 数据处理机制

这个脚本用于分析 DPO (Direct Preference Optimization) 的数据处理流程，包括：
1. 数据加载和预处理
2. 模板应用
3. 标记化处理
4. 标签生成
5. 数据整理和批处理
"""
import os
import sys
import yaml
import torch
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from typing import Dict, List, Optional
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
from src.llamafactory.train.dpo.workflow import run_dpo
from src.llamafactory.data.template import get_template_and_fix_tokenizer
from src.llamafactory.data.loader import get_dataset
from src.llamafactory.extras.constants import IGNORE_INDEX

def print_section(title: str, content: str = None):
    """打印带格式的段落标题和内容"""
    rprint(Panel(f"[bold cyan]{title}[/bold cyan]"))
    if content:
        rprint(content)
    rprint("")

def analyze_tokenized_sample(tokenizer, input_ids: List[int], labels: List[int], title: str = "样本分析"):
    """分析标记化后的样本"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("类型", style="cyan")
    table.add_column("长度", style="green")
    table.add_column("内容", style="yellow")
    
    # 分析 input_ids
    prompt_end = None
    for i, token_id in enumerate(input_ids):
        if tokenizer.decode([token_id]).startswith("<|im_start|>assistant"):
            prompt_end = i
            break
    
    if prompt_end is not None:
        prompt_text = tokenizer.decode(input_ids[:prompt_end])
        response_text = tokenizer.decode(input_ids[prompt_end:])
        table.add_row("Prompt", str(prompt_end), prompt_text)
        table.add_row("Response", str(len(input_ids) - prompt_end), response_text)
    else:
        table.add_row("完整序列", str(len(input_ids)), tokenizer.decode(input_ids))
    
    # 分析标签
    valid_labels = [x for x in labels if x != IGNORE_INDEX]
    ignored_count = len([x for x in labels if x == IGNORE_INDEX])
    table.add_row("标签 (非IGNORE)", str(len(valid_labels)), tokenizer.decode(valid_labels))
    table.add_row("IGNORE标签数量", str(ignored_count), f"占比: {ignored_count/len(labels):.2%}")
    
    rprint(table)

def test_dpo_data_processing():
    """测试 DPO 数据处理机制"""
    print_section("DPO 数据处理机制测试")
    
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
    
    config_info = f"""
    模型配置:
    - 模型路径: {model_args.model_name_or_path}
    - 数据集: {data_args.dataset}
    - 模板: {data_args.template}
    - 最大样本数: {data_args.max_samples}
    
    训练配置:
    - Batch Size: {training_args.per_device_train_batch_size}
    - 梯度累积: {training_args.gradient_accumulation_steps}
    - 学习率: {training_args.learning_rate}
    
    DPO 配置:
    - Beta: {finetuning_args.pref_beta}
    - Loss Type: {finetuning_args.pref_loss}
    """
    rprint(Panel(config_info, title="配置信息"))
    
    # 2. 加载分词器和模板
    print_section("2. 分词器和模板加载")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    rprint(f"[green]分词器词表大小: {len(tokenizer)}[/green]")
    rprint(f"[green]特殊标记: {tokenizer.all_special_tokens}[/green]")
    
    # 3. 加载数据集
    print_section("3. 数据集加载")
    try:
        # 打印数据集信息
        rprint("[yellow]数据集配置:[/yellow]")
        rprint(f"数据集名称: {data_args.dataset}")
        rprint(f"数据集目录: {data_args.dataset_dir}")
        rprint(f"最大样本数: {data_args.max_samples}")
        
        # 检查数据集属性
        import json
        dataset_info_path = os.path.join(data_args.dataset_dir, "dataset_info.json")
        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)
        
        dataset_name = data_args.dataset[0] if isinstance(data_args.dataset, list) else data_args.dataset
        if dataset_name in dataset_info:
            info = dataset_info[dataset_name]
            rprint("\n[yellow]数据集属性:[/yellow]")
            rprint(f"文件名: {info.get('file_name', 'N/A')}")
            rprint(f"格式化: {info.get('formatting', 'N/A')}")
            rprint(f"列映射: {info.get('columns', {})}")
            rprint(f"是否为排序数据: {info.get('ranking', False)}")
        else:
            rprint(f"[red]警告: 在 dataset_info.json 中未找到数据集 {dataset_name} 的配置信息[/red]")
        
        # 检查数据集文件是否存在
        dataset_path = os.path.join(data_args.dataset_dir, "dpo_en_demo.json")
        if os.path.exists(dataset_path):
            rprint(f"[green]数据集文件存在: {dataset_path}[/green]")
            rprint(f"文件大小: {os.path.getsize(dataset_path)} bytes")
        else:
            rprint(f"[red]数据集文件不存在: {dataset_path}[/red]")
        
        # 尝试加载数据集
        rprint("\n[yellow]正在加载数据集...[/yellow]")
        try:
            dataset = get_dataset(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                stage="rm",
                tokenizer=tokenizer,
                template=template
            )
            rprint(f"[green]数据集加载返回值类型: {type(dataset)}[/green]")
            rprint(f"[green]数据集加载返回值内容: {dataset}[/green]")
            
            if isinstance(dataset, dict):
                if "train_dataset" in dataset:
                    dataset = dataset["train_dataset"]
                    rprint(f"[green]使用训练集，大小: {len(dataset)}[/green]")
                else:
                    rprint("[red]未找到训练集[/red]")
                    return
            
            rprint(f"[green]数据集加载成功![/green]")
            rprint(f"数据集大小: {len(dataset)}")
            
            # 分析第一个样本
            print_section("4. 样本分析")
            example = dataset[0]
            
            rprint("[bold]处理后的数据:[/bold]")
            rprint("\n[bold]Chosen 样本:[/bold]")
            analyze_tokenized_sample(
                tokenizer,
                example["chosen_input_ids"],
                example["chosen_labels"],
                "Chosen 样本分析"
            )
            
            rprint("\n[bold]Rejected 样本:[/bold]")
            analyze_tokenized_sample(
                tokenizer,
                example["rejected_input_ids"],
                example["rejected_labels"],
                "Rejected 样本分析"
            )
            
            # 分析标签分布
            print_section("5. 标签分布分析")
            chosen_ignore_ratio = sum(1 for x in example["chosen_labels"] if x == IGNORE_INDEX) / len(example["chosen_labels"])
            rejected_ignore_ratio = sum(1 for x in example["rejected_labels"] if x == IGNORE_INDEX) / len(example["rejected_labels"])
            
            label_stats = f"""
            Chosen 样本:
            - 总长度: {len(example['chosen_labels'])}
            - IGNORE_INDEX 比例: {chosen_ignore_ratio:.2%}
            - 有效标签数: {len([x for x in example['chosen_labels'] if x != IGNORE_INDEX])}
            
            Rejected 样本:
            - 总长度: {len(example['rejected_labels'])}
            - IGNORE_INDEX 比例: {rejected_ignore_ratio:.2%}
            - 有效标签数: {len([x for x in example['rejected_labels'] if x != IGNORE_INDEX])}
            """
            rprint(Panel(label_stats, title="标签统计"))
            
        except Exception as e:
            rprint(f"[red]数据处理过程中发生错误:[/red]")
            rprint(f"[red]{str(e)}[/red]")
            import traceback
            rprint(traceback.format_exc())
        
    except Exception as e:
        rprint(f"[red]数据处理过程中发生错误:[/red]")
        rprint(f"[red]{str(e)}[/red]")
        import traceback
        rprint(traceback.format_exc())
    
    print_section("测试完成")

if __name__ == "__main__":
    test_dpo_data_processing() 