"""
测试 DPO 数据处理流程和 PPL 计算
"""
import os
import sys
import torch
from transformers import AutoTokenizer, HfArgumentParser, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Dict, List, Optional

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments
)
from src.llamafactory.data.processor.pairwise import PairwiseDatasetProcessor
from src.llamafactory.data.collator import PairwiseDataCollatorWithPadding
from src.llamafactory.data.template import get_template_and_fix_tokenizer
from src.llamafactory.train.betadpo.trainer import LearnableBetaDPOTrainer
from src.llamafactory.extras.constants import IGNORE_INDEX

def create_test_sample():
    """创建一个简单的测试样本"""
    return {
        "_prompt": [[{"role": "user", "content": "What is Python?"}]],
        "_response": [
            [
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "assistant", "content": "Python is a snake."}
            ]
        ],
        "_system": [""],
        "_tools": [""],
        "_images": [[]],
        "_videos": [[]],
        "_audios": [[]]
    }

def trace_data_processing(model_path: str = "/root/models/Qwen1.5-0.5B-Chat"):
    """追踪数据处理的完整流程"""
    print("=== 开始追踪数据处理流程 ===")
    
    # 1. 加载分词器和模板
    print("\n1. 加载分词器和模板")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data_args = DataArguments(template="qwen")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 2. 创建数据处理器
    print("\n2. 创建数据处理器")
    processor = PairwiseDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=None,
        data_args=data_args
    )
    
    # 3. 处理测试样本
    print("\n3. 处理测试样本")
    test_sample = create_test_sample()
    processed = processor.preprocess_dataset(test_sample)
    
    print("\n处理后的数据:")
    print("- chosen_input_ids 形状:", len(processed["chosen_input_ids"][0]))
    print("- chosen_labels 形状:", len(processed["chosen_labels"][0]))
    print("- rejected_input_ids 形状:", len(processed["rejected_input_ids"][0]))
    print("- rejected_labels 形状:", len(processed["rejected_labels"][0]))
    
    # 打印 token 解码结果
    print("\nChosen 样本:")
    print("输入:", tokenizer.decode(processed["chosen_input_ids"][0]))
    valid_chosen_labels = [x for x in processed["chosen_labels"][0] if x != IGNORE_INDEX]
    print("标签:", tokenizer.decode(valid_chosen_labels))
    
    print("\nRejected 样本:")
    print("输入:", tokenizer.decode(processed["rejected_input_ids"][0]))
    valid_rejected_labels = [x for x in processed["rejected_labels"][0] if x != IGNORE_INDEX]
    print("标签:", tokenizer.decode(valid_rejected_labels))
    
    # 4. 创建数据整理器
    print("\n4. 使用数据整理器处理 batch")
    collator = PairwiseDataCollatorWithPadding(
        template=template,
        tokenizer=tokenizer,
        model=None,
        pad_to_multiple_of=8
    )
    
    # 创建一个 batch
    features = [{
        "chosen_input_ids": processed["chosen_input_ids"][0],
        "chosen_attention_mask": processed["chosen_attention_mask"][0],
        "chosen_labels": processed["chosen_labels"][0],
        "rejected_input_ids": processed["rejected_input_ids"][0],
        "rejected_attention_mask": processed["rejected_attention_mask"][0],
        "rejected_labels": processed["rejected_labels"][0],
        "images": processed["images"][0],
        "videos": processed["videos"][0],
        "audios": processed["audios"][0]
    }]
    
    batch = collator(features)
    
    print("\nBatch 数据形状:")
    print("- input_ids:", batch["input_ids"].shape)
    print("- attention_mask:", batch["attention_mask"].shape)
    print("- labels:", batch["labels"].shape)
    
    # 5. 验证 IGNORE_INDEX 标记
    print("\n5. 验证 IGNORE_INDEX 标记")
    labels = batch["labels"]
    for i in range(labels.size(0)):
        response_start = (labels[i] != IGNORE_INDEX).nonzero()
        if len(response_start) > 0:
            prompt_len = response_start[0].item()
            print(f"\n样本 {i}:")
            print(f"- Prompt 长度: {prompt_len}")
            print(f"- Response 开始位置: {prompt_len}")
            print(f"- 完整序列长度: {labels.size(1)}")
            
            # 打印完整的标签序列
            print("\n标签序列:")
            print(labels[i].tolist())
            
            # 验证 prompt 部分是否全是 IGNORE_INDEX
            prompt_labels = labels[i, :prompt_len]
            print("\nPrompt 部分标签:")
            print(prompt_labels.tolist())
            assert (prompt_labels == IGNORE_INDEX).all(), f"样本 {i} 的 prompt 部分包含非 IGNORE_INDEX 值"
            
            # 验证 response 部分是否没有 IGNORE_INDEX
            response_labels = labels[i, prompt_len:]
            print("\nResponse 部分标签:")
            print(response_labels.tolist())
            print("\nResponse 部分（去除 padding）:")
            response_labels = response_labels[response_labels != tokenizer.pad_token_id]
            print(response_labels.tolist())
            
            # 打印解码后的文本
            print("\n解码示例:")
            print("完整输入:", tokenizer.decode(batch["input_ids"][i]))
            print("Prompt 部分:", tokenizer.decode(batch["input_ids"][i][:prompt_len]))
            print("Response 部分:", tokenizer.decode(batch["input_ids"][i][prompt_len:]))
            print("\nResponse 标签解码:")
            print(tokenizer.decode(response_labels))

def main():
    """主函数"""
    trace_data_processing()

if __name__ == "__main__":
    main() 