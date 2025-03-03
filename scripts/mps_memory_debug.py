#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MPS内存使用情况监控脚本

此脚本用于监控Apple Silicon的MPS内存使用情况，
并提供诊断信息以帮助优化训练过程中的内存使用。

使用方法：
    python scripts/mps_memory_debug.py --model_path ~/models/Qwen1.5-0.5B

作者: Claude 3.7 (2025)
"""

import os
import sys
import argparse
import time
import math
import torch
import gc
import platform
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

def human_readable_size(size_bytes):
    """将字节转换为人类可读的格式"""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def get_system_info():
    """获取系统信息"""
    print("=" * 50)
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"MPS可用: {torch.backends.mps.is_available()}")
    
    # 获取物理内存信息
    mem = psutil.virtual_memory()
    print(f"系统总内存: {human_readable_size(mem.total)}")
    print(f"系统可用内存: {human_readable_size(mem.available)}")
    print(f"内存使用率: {mem.percent}%")
    print("=" * 50)

def get_mps_memory_info():
    """获取MPS内存信息"""
    if not torch.backends.mps.is_available():
        print("MPS设备不可用")
        return None
    
    # MPS当前不支持直接获取内存统计信息，但我们可以检查环境变量
    print("MPS内存配置:")
    watermark_ratio = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "默认")
    print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {watermark_ratio}")
    
    # 尝试分配内存并观察
    try:
        test_tensor = torch.ones((1024, 1024), device="mps")
        print("成功分配测试张量到MPS设备")
        del test_tensor
        torch.mps.empty_cache()
    except Exception as e:
        print(f"分配测试张量时出错: {e}")
    
    # 清理缓存
    gc.collect()
    torch.mps.empty_cache()
    return True

def measure_model_loading(model_path):
    """测量模型加载时的内存使用情况"""
    print("\n" + "=" * 50)
    print(f"加载模型: {model_path}")
    print("=" * 50)
    
    # 在加载前获取内存状态
    gc.collect()
    torch.mps.empty_cache()
    mem_before = psutil.virtual_memory()
    
    start_time = time.time()
    
    # 首先加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer加载完成")
    
    # 然后加载模型
    print("开始加载模型...")
    try:
        # 尝试将模型加载到MPS设备
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="mps"
        )
        print("模型成功加载到MPS设备")
    except Exception as e:
        print(f"加载模型到MPS设备出错: {e}")
        print("尝试加载到CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu"
        )
        print("模型成功加载到CPU")
    
    end_time = time.time()
    loading_time = end_time - start_time
    
    # 获取加载后的内存状态
    mem_after = psutil.virtual_memory()
    memory_used = mem_before.available - mem_after.available
    
    print(f"\n模型加载时间: {loading_time:.2f}秒")
    print(f"加载模型消耗内存: {human_readable_size(memory_used)}")
    
    # 输出模型信息
    print("\n模型信息:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    print(f"模型类型: {type(model).__name__}")
    
    # 尝试运行一个简单的推理
    print("\n尝试运行推理...")
    try:
        input_text = "你好，我是"
        input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            start_infer_time = time.time()
            output = model.generate(
                **input_ids,
                max_new_tokens=20,
                num_return_sequences=1
            )
            end_infer_time = time.time()
        
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        infer_time = end_infer_time - start_infer_time
        
        print(f"输入: {input_text}")
        print(f"输出: {output_text}")
        print(f"推理时间: {infer_time:.2f}秒")
    except Exception as e:
        print(f"推理过程中出错: {e}")
    
    # 清理资源
    del model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()
    
    return True

def test_lora_memory(model_path):
    """测试LoRA配置下的内存使用情况"""
    from peft import get_peft_model, LoraConfig, TaskType
    
    print("\n" + "=" * 50)
    print("测试不同LoRA配置下的内存使用情况")
    print("=" * 50)
    
    # 加载基础模型到CPU
    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 测试不同的LoRA配置
    lora_ranks = [4, 8, 16, 32]
    
    for rank in lora_ranks:
        print(f"\n测试LoRA rank={rank}的内存使用情况")
        
        # 清理之前的内存
        gc.collect()
        torch.mps.empty_cache()
        mem_before = psutil.virtual_memory()
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        # 应用LoRA
        peft_model = get_peft_model(model, lora_config)
        
        # 尝试移动到MPS设备
        try:
            print("将模型移动到MPS设备...")
            peft_model = peft_model.to("mps")
            
            # 测试一次推理
            input_text = "你好，我是"
            input_ids = tokenizer(input_text, return_tensors="pt").to("mps")
            
            with torch.no_grad():
                output = peft_model.generate(
                    **input_ids,
                    max_new_tokens=10,
                    num_return_sequences=1
                )
            
            # 获取内存使用情况
            mem_after = psutil.virtual_memory()
            memory_used = mem_before.available - mem_after.available
            
            print(f"LoRA rank={rank} 消耗内存: {human_readable_size(memory_used)}")
            
            # 计算可训练参数
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in peft_model.parameters())
            print(f"可训练参数: {trainable_params:,} ({trainable_params/all_params*100:.2f}% 的总参数)")
            
            # 清理资源
            del peft_model
            gc.collect()
            torch.mps.empty_cache()
            
        except Exception as e:
            print(f"使用LoRA rank={rank}时出错: {e}")
    
    # 清理资源
    del model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()
    
    return True

def test_different_seq_lengths(model_path):
    """测试不同序列长度对内存的影响"""
    print("\n" + "=" * 50)
    print("测试不同序列长度对内存的影响")
    print("=" * 50)
    
    # 加载模型到CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 测试不同序列长度
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    for seq_len in seq_lengths:
        print(f"\n测试序列长度 {seq_len} 的内存使用情况")
        
        # 生成测试数据
        test_tokens = torch.randint(0, model.config.vocab_size, (1, seq_len))
        
        # 清理之前的内存
        gc.collect()
        torch.mps.empty_cache()
        mem_before = psutil.virtual_memory()
        
        try:
            # 将模型移动到MPS
            model_mps = model.to("mps")
            test_tokens_mps = test_tokens.to("mps")
            
            # 执行前向传播
            with torch.no_grad():
                start_time = time.time()
                outputs = model_mps(test_tokens_mps)
                end_time = time.time()
            
            # 获取内存使用情况
            mem_after = psutil.virtual_memory()
            memory_used = mem_before.available - mem_after.available
            
            print(f"序列长度 {seq_len} 消耗内存: {human_readable_size(memory_used)}")
            print(f"前向传播时间: {(end_time - start_time):.4f}秒")
            
            # 将模型移回CPU
            model_mps = model_mps.to("cpu")
            del model_mps
            del test_tokens_mps
            gc.collect()
            torch.mps.empty_cache()
            
        except Exception as e:
            print(f"测试序列长度 {seq_len} 时出错: {e}")
    
    # 清理资源
    del model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="MPS内存使用情况监控脚本")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--test_all", action="store_true", help="运行所有测试")
    args = parser.parse_args()
    
    # 显示系统信息
    get_system_info()
    
    # 检查MPS可用性和内存信息
    get_mps_memory_info()
    
    # 如果指定了模型路径，则加载模型并测量内存使用情况
    if args.model_path or args.test_all:
        model_path = args.model_path or "~/models/Qwen1.5-0.5B"
        measure_model_loading(model_path)
        
        # 如果指定了test_all，则运行所有测试
        if args.test_all:
            test_lora_memory(model_path)
            test_different_seq_lengths(model_path)
    
    print("\n" + "=" * 50)
    print("内存优化建议:")
    print("=" * 50)
    print("1. 设置 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 以禁用MPS内存限制")
    print("2. 减小序列长度(MAX_SEQ_LEN)，考虑使用512或更小的值")
    print("3. 减小batch_size和gradient_accumulation_steps")
    print("4. 启用梯度检查点(gradient_checkpointing)以减少激活值内存")
    print("5. 降低LoRA参数rank值，例如从8减少到4")
    print("6. 减少保存的检查点数量，设置save_total_limit=1")
    print("7. 考虑使用更小的模型，如Qwen1.5-0.5B-Chat")
    print("8. 关闭其他内存密集型应用程序")
    print("9. 针对小数据集，减少样本数量，使用max_samples参数")

if __name__ == "__main__":
    main() 