#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单测试脚本，验证fooDPO算法是否正确导入
"""

import sys
from llamafactory.train.foodpo import run_foodpo
from llamafactory.train.tuner import _training_function


def test_import():
    """
    测试fooDPO模块是否能够正确导入
    """
    print("测试导入fooDPO模块...")
    assert callable(run_foodpo), "run_foodpo 不是一个可调用函数"
    print("✓ run_foodpo 可以被成功导入")
    
    # 检查函数的参数名称是否正确
    import inspect
    params = inspect.signature(run_foodpo).parameters
    expected_params = ['model_args', 'data_args', 'training_args', 'finetuning_args', 'callbacks']
    for param in expected_params:
        assert param in params, f"run_foodpo缺少必要的参数: {param}"
    print(f"✓ run_foodpo 具有正确的参数: {', '.join(expected_params)}")
    
    return True


def test_tuner_integration():
    """
    测试fooDPO是否已经被集成到tuner模块中
    """
    print("测试fooDPO在tuner模块中的集成...")
    
    # 检查tuner文件中是否含有foodpo分支
    import inspect
    src = inspect.getsource(_training_function)
    assert "elif finetuning_args.stage == \"foodpo\":" in src, "tuner模块中没有foodpo分支"
    assert "run_foodpo(" in src, "tuner模块中没有调用run_foodpo"
    print("✓ foodpo已正确集成到tuner模块")
    
    return True


if __name__ == "__main__":
    print("开始测试fooDPO模块...")
    
    try:
        test_import()
        test_tuner_integration()
        print("\n✓ 所有测试通过！fooDPO模块可以正常使用")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        sys.exit(1) 