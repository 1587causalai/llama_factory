#!/usr/bin/env python
# coding=utf-8

"""
简单测试脚本，验证betadpo算法是否正确导入
"""

import inspect
from typing import get_type_hints
from llamafactory.train.betadpo import run_betadpo
from llamafactory.hparams import FinetuningArguments

# 测试betadpo模块是否能够正确导入
def test_betadpo_import():
    print("测试导入betadpo模块...")
    assert callable(run_betadpo), "run_betadpo 不是一个可调用函数"
    print("✓ run_betadpo 可以被成功导入")

    # 检查函数参数
    expected_params = ["model_args", "data_args", "training_args", "finetuning_args", "callbacks"]
    params = inspect.signature(run_betadpo).parameters
    
    for param in expected_params:
        assert param in params, f"run_betadpo缺少必要的参数: {param}"
    print(f"✓ run_betadpo 具有正确的参数: {', '.join(expected_params)}")

# 测试betadpo是否已经被集成到tuner模块中
def test_betadpo_in_tuner():
    print("测试betadpo在tuner模块中的集成...")
    
    # 读取tuner.py文件
    with open("src/llamafactory/train/tuner.py", "r") as f:
        src = f.read()
    
    # 检查tuner文件中是否含有betadpo分支
    assert "from .betadpo import run_betadpo" in src, "tuner模块中没有导入run_betadpo"
    assert "elif finetuning_args.stage == \"betadpo\":" in src, "tuner模块中没有betadpo分支"
    assert "run_betadpo(" in src, "tuner模块中没有调用run_betadpo"
    print("✓ betadpo已正确集成到tuner模块")

# 测试betadpo参数是否已经被添加到FinetuningArguments中
def test_betadpo_args():
    print("测试betadpo参数是否已添加到FinetuningArguments...")
    
    # 实例化FinetuningArguments
    args = FinetuningArguments()
    
    # 检查betadpo特定参数
    assert hasattr(args, "beta_strategy"), "FinetuningArguments中缺少beta_strategy参数"
    assert hasattr(args, "beta_min"), "FinetuningArguments中缺少beta_min参数"
    assert hasattr(args, "beta_max"), "FinetuningArguments中缺少beta_max参数"
    print("✓ FinetuningArguments中包含betadpo所需参数")
    
    # 读取finetuning_args.py文件检查stage参数
    with open("src/llamafactory/hparams/finetuning_args.py", "r") as f:
        src = f.read()
    
    assert "Literal[" in src and "\"betadpo\"" in src, "stage参数不支持betadpo"
    print("✓ FinetuningArguments.stage参数支持betadpo")

if __name__ == "__main__":
    print("开始测试betadpo模块...")
    test_betadpo_import()
    test_betadpo_in_tuner()
    test_betadpo_args()
    print("\n✓ 所有测试通过！betadpo模块可以正常使用") 