#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小化LEDPO(可学习beta的DPO)测试脚本
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import logging
import argparse
from peft import LoraConfig

# 添加项目根目录到Python路径
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from src.llamafactory.train.ledpo import LEDPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="最小化LEDPO测试脚本")
    
    parser.add_argument("--model_name_or_path", type=str, required=True, help="模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./saves/ledpo_test", help="输出目录")
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf", help="数据集名称")
    parser.add_argument("--split", type=str, default="train", help="数据集分片")
    parser.add_argument("--max_samples", type=int, default=100, help="最大样本数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--beta", type=float, default=0.1, help="初始beta值")
    parser.add_argument("--value_head_lr", type=float, default=1e-3, help="value head学习率")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="模型学习率")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="最大提示长度")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    
    return parser.parse_args()


def prepare_data(args, tokenizer):
    """准备训练和评估数据"""
    # 加载Anthropic RLHF数据集
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    # 限制样本数
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
    
    # 划分训练和评估集
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")
    
    # 处理数据格式
    def preprocess_function(examples):
        result = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        
        # RLHF数据集处理
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            # 提取共同的prompt部分
            prompt = chosen.split("Human:")[0]
            chosen_response = chosen[len(prompt):]
            rejected_response = rejected[len(prompt):]
            
            result["prompt"].append(prompt)
            result["chosen"].append(chosen_response)
            result["rejected"].append(rejected_response)
        
        return result
    
    # 应用预处理
    train_dataset = preprocess_function(train_dataset)
    eval_dataset = preprocess_function(eval_dataset)
    
    # 转换为Dataset对象
    train_dataset = Dataset.from_dict(train_dataset)
    eval_dataset = Dataset.from_dict(eval_dataset)
    
    # 标记化处理
    def tokenize_function(examples):
        prompt_tokenized = tokenizer(examples["prompt"], truncation=False, padding=False)
        chosen_tokenized = tokenizer(examples["chosen"], truncation=False, padding=False)
        rejected_tokenized = tokenizer(examples["rejected"], truncation=False, padding=False)
        
        result = {
            "prompt_input_ids": prompt_tokenized["input_ids"],
            "prompt_attention_mask": prompt_tokenized["attention_mask"],
            "chosen_input_ids": chosen_tokenized["input_ids"],
            "chosen_attention_mask": chosen_tokenized["attention_mask"],
            "rejected_input_ids": rejected_tokenized["input_ids"],
            "rejected_attention_mask": rejected_tokenized["attention_mask"],
        }
        
        return result
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, eval_dataset


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    # 准备LoRA配置
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    # 准备数据
    train_dataset, eval_dataset = prepare_data(args, tokenizer)
    
    # 创建LEDPO训练器
    trainer = LEDPOTrainer(
        model=model,
        args=None,  # 使用默认的TrainingArguments
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        value_head_lr=args.value_head_lr,
    )
    
    # 设置训练参数
    trainer.train_args.per_device_train_batch_size = args.batch_size
    trainer.train_args.per_device_eval_batch_size = args.batch_size
    trainer.train_args.gradient_accumulation_steps = 1
    trainer.train_args.learning_rate = args.learning_rate
    trainer.train_args.num_train_epochs = args.num_train_epochs
    trainer.train_args.output_dir = args.output_dir
    trainer.train_args.logging_steps = 10
    trainer.train_args.evaluation_strategy = "steps"
    trainer.train_args.eval_steps = 50
    trainer.train_args.save_strategy = "steps"
    trainer.train_args.save_steps = 50
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model(args.output_dir)
    
    print(f"训练完成！模型保存在 {args.output_dir}")


if __name__ == "__main__":
    main() 