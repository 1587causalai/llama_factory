#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用DPO训练器进行模型训练的脚本
"""

import os
import json
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

# 导入我们的DPO训练器
from dpo_trainer import DPOTrainer


class DPODataset(Dataset):
    """
    DPO训练数据集
    """
    def __init__(self, data_path, max_samples=None):
        """
        初始化DPO数据集
        
        Args:
            data_path: 数据路径，支持Hugging Face数据集或本地jsonl文件
            max_samples: 最大样本数，用于调试
        """
        self.data = []
        
        # 加载数据
        if os.path.exists(data_path):
            # 本地文件
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            # Hugging Face数据集
            dataset = load_dataset(data_path)
            if isinstance(dataset, dict):
                dataset = dataset['train']
            for item in dataset:
                self.data.append(item)
        
        # 限制样本数量
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
        
        print(f"加载了 {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 根据数据格式调整键名
        prompt = item.get('prompt', item.get('instruction', item.get('question', '')))
        chosen = item.get('chosen', item.get('chosen_response', item.get('response_j', '')))
        rejected = item.get('rejected', item.get('rejected_response', item.get('response_k', '')))
        
        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        }


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="DPO训练脚本")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default="/root/models/Qwen1.5-0.5B",
                        help="要训练的模型路径")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="参考模型路径，默认使用model_path的副本")
    
    # 数据相关参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径，支持Hugging Face数据集或本地jsonl文件")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数，用于调试")
    
    # 训练相关参数
    parser.add_argument("--output_dir", type=str, default="./dpo_output",
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO温度参数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="每多少步保存一次模型")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--debug", action="store_true",
                        help="是否开启调试模式")
    
    return parser.parse_args()


def set_seed(seed):
    """
    设置随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # 加载数据集
    dataset = DPODataset(args.data_path, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 创建DPO训练器
    trainer = DPOTrainer(
        model_path=args.model_path,
        ref_model_path=args.ref_model_path,
        beta=args.beta,
        max_length=args.max_length,
        debug=args.debug
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=args.learning_rate)
    
    # 创建学习率调度器
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        print(f"开始第 {epoch+1}/{args.epochs} 轮训练")
        
        epoch_loss = 0.0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}") as pbar:
            for batch in dataloader:
                prompts = batch["prompt"]
                chosen_responses = batch["chosen"]
                rejected_responses = batch["rejected"]
                
                # 执行训练步骤
                metrics = trainer.train_step(
                    prompts=prompts,
                    chosen_responses=chosen_responses,
                    rejected_responses=rejected_responses,
                    optimizer=optimizer
                )
                
                # 更新学习率
                scheduler.step()
                
                # 更新进度条
                epoch_loss += metrics["loss"]
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.7f}"
                })
                
                # 存储中间结果
                if global_step % 100 == 0:
                    with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
                        f.write(json.dumps({
                            "step": global_step,
                            "epoch": epoch,
                            **metrics
                        }) + "\n")
                
                # 保存模型
                if global_step > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    trainer.save_model(checkpoint_dir)
                
                global_step += 1
        
        # 每轮结束保存模型
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        trainer.save_model(epoch_dir)
        
        # 输出轮次平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f"第 {epoch+1} 轮平均损失: {avg_loss:.4f}")
    
    # 保存最终模型
    trainer.save_model(os.path.join(args.output_dir, "final-model"))
    print("训练完成！")


if __name__ == "__main__":
    main() 