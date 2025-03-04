#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DPO模型评估脚本，用于比较训练前后模型的表现
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List, Dict, Tuple

# 导入我们的DPO训练器，用于数据处理和评估
from dpo_trainer import DPOTrainer


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="DPO模型评估脚本")
    
    # 模型相关参数
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="基础模型路径（训练前的模型）")
    parser.add_argument("--dpo_model_path", type=str, required=True,
                        help="DPO训练后的模型路径")
    
    # 数据相关参数
    parser.add_argument("--eval_data_path", type=str, required=True,
                        help="评估数据路径，支持Hugging Face数据集或本地jsonl文件")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="最大评估样本数")
    
    # 评估相关参数
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--output_dir", type=str, default="./dpo_eval",
                        help="评估结果输出目录")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO温度参数，用于计算奖励")
    
    return parser.parse_args()


def load_eval_data(data_path, max_samples):
    """
    加载评估数据
    """
    data = []
    
    # 加载数据
    if os.path.exists(data_path):
        # 本地文件
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        # Hugging Face数据集
        dataset = load_dataset(data_path)
        if isinstance(dataset, dict):
            dataset = dataset['test' if 'test' in dataset else 'validation' if 'validation' in dataset else 'train']
        for item in dataset:
            data.append(item)
    
    # 限制样本数量
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
    
    # 提取数据字段
    prompts = []
    chosen_responses = []
    rejected_responses = []
    
    for item in data:
        prompt = item.get('prompt', item.get('instruction', item.get('question', '')))
        chosen = item.get('chosen', item.get('chosen_response', item.get('response_j', '')))
        rejected = item.get('rejected', item.get('rejected_response', item.get('response_k', '')))
        
        prompts.append(prompt)
        chosen_responses.append(chosen)
        rejected_responses.append(rejected)
    
    print(f"加载了 {len(prompts)} 个评估样本")
    return prompts, chosen_responses, rejected_responses


def visualize_results(
    base_results: Dict[str, List[float]],
    dpo_results: Dict[str, List[float]],
    output_dir: str
):
    """
    可视化评估结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图样式
    plt.style.use('ggplot')
    
    # 1. 比较基础模型和DPO模型在偏好数据上的概率
    plt.figure(figsize=(12, 6))
    
    labels = ['Base-Chosen', 'Base-Rejected', 'DPO-Chosen', 'DPO-Rejected']
    means = [
        np.mean(base_results['chosen_logp']),
        np.mean(base_results['rejected_logp']),
        np.mean(dpo_results['chosen_logp']),
        np.mean(dpo_results['rejected_logp'])
    ]
    
    plt.bar(labels, means, color=['blue', 'red', 'green', 'orange'])
    plt.title('Average Log Probability Comparison')
    plt.ylabel('Log Probability')
    plt.savefig(os.path.join(output_dir, 'logp_comparison.png'))
    
    # 2. 奖励分布对比
    plt.figure(figsize=(12, 6))
    
    plt.hist(base_results['rewards'], bins=30, alpha=0.5, label='Base Model')
    plt.hist(dpo_results['rewards'], bins=30, alpha=0.5, label='DPO Model')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'reward_distribution.png'))
    
    # 3. 偏好对齐率
    plt.figure(figsize=(12, 6))
    
    base_alignment = np.mean([1 if c > r else 0 for c, r in zip(
        base_results['chosen_logp'], base_results['rejected_logp'])])
    dpo_alignment = np.mean([1 if c > r else 0 for c, r in zip(
        dpo_results['chosen_logp'], dpo_results['rejected_logp'])])
    
    plt.bar(['Base Model', 'DPO Model'], [base_alignment, dpo_alignment], color=['blue', 'green'])
    plt.title('Preference Alignment Rate')
    plt.ylabel('Rate')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'alignment_rate.png'))
    
    # 保存原始数据
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump({
            'base_results': {k: [float(x) for x in v] for k, v in base_results.items()},
            'dpo_results': {k: [float(x) for x in v] for k, v in dpo_results.items()},
            'summary': {
                'base_chosen_mean_logp': float(np.mean(base_results['chosen_logp'])),
                'base_rejected_mean_logp': float(np.mean(base_results['rejected_logp'])),
                'dpo_chosen_mean_logp': float(np.mean(dpo_results['chosen_logp'])),
                'dpo_rejected_mean_logp': float(np.mean(dpo_results['rejected_logp'])),
                'base_alignment_rate': float(base_alignment),
                'dpo_alignment_rate': float(dpo_alignment),
                'improvement': float(dpo_alignment - base_alignment)
            }
        }, f, indent=2)


def evaluate_model(
    model_path: str,
    prompts: List[str],
    chosen_responses: List[str],
    rejected_responses: List[str],
    batch_size: int,
    max_length: int,
    beta: float
) -> Dict[str, List[float]]:
    """
    评估模型在偏好数据上的表现
    """
    # 创建DPO训练器（仅用于评估）
    trainer = DPOTrainer(
        model_path=model_path,
        ref_model_path=model_path,  # 参考模型与评估模型相同
        beta=beta,
        max_length=max_length
    )
    
    # 准备结果存储
    results = {
        'chosen_logp': [],
        'rejected_logp': [],
        'rewards': []
    }
    
    # 分批处理数据
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"评估 {os.path.basename(model_path)}"):
        batch_prompts = prompts[i:i+batch_size]
        batch_chosen = chosen_responses[i:i+batch_size]
        batch_rejected = rejected_responses[i:i+batch_size]
        
        # 准备批次数据
        batch = trainer.tokenize_batch(batch_prompts, batch_chosen, batch_rejected)
        
        # 前向传播
        with torch.no_grad():
            _, outputs = trainer.forward(batch)
        
        # 收集结果
        results['chosen_logp'].extend(outputs.policy_chosen_logps.cpu().numpy())
        results['rejected_logp'].extend(outputs.policy_rejected_logps.cpu().numpy())
        
        # 计算奖励（chosen - rejected的概率差值）
        rewards = outputs.policy_chosen_logps - outputs.policy_rejected_logps
        results['rewards'].extend(rewards.cpu().numpy())
    
    return results


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载评估数据
    prompts, chosen_responses, rejected_responses = load_eval_data(
        args.eval_data_path, args.max_samples)
    
    # 评估基础模型
    print("开始评估基础模型...")
    base_results = evaluate_model(
        model_path=args.base_model_path,
        prompts=prompts,
        chosen_responses=chosen_responses,
        rejected_responses=rejected_responses,
        batch_size=args.batch_size,
        max_length=args.max_length,
        beta=args.beta
    )
    
    # 评估DPO模型
    print("开始评估DPO模型...")
    dpo_results = evaluate_model(
        model_path=args.dpo_model_path,
        prompts=prompts,
        chosen_responses=chosen_responses,
        rejected_responses=rejected_responses,
        batch_size=args.batch_size,
        max_length=args.max_length,
        beta=args.beta
    )
    
    # 可视化结果
    print("生成评估结果可视化...")
    visualize_results(base_results, dpo_results, args.output_dir)
    
    # 打印主要指标
    base_alignment = np.mean([1 if c > r else 0 for c, r in zip(
        base_results['chosen_logp'], base_results['rejected_logp'])])
    dpo_alignment = np.mean([1 if c > r else 0 for c, r in zip(
        dpo_results['chosen_logp'], dpo_results['rejected_logp'])])
    
    print(f"\n评估结果:")
    print(f"基础模型偏好对齐率: {base_alignment:.4f}")
    print(f"DPO模型偏好对齐率: {dpo_alignment:.4f}")
    print(f"改进幅度: {dpo_alignment - base_alignment:.4f}")
    print(f"\n详细结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main() 