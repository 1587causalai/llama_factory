#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行DPO训练的Python脚本
等价于运行: llamafactory-cli train examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml
但直接使用LLaMA-Factory内部API，而不是通过子进程调用命令行
"""

import sys
import os
from typing import Dict, Any, Optional

# 导入LLaMA-Factory需要的模块
from llamafactory.train.tuner import run_exp
from llamafactory.extras.misc import get_device_count, is_env_enabled, use_ray


def main():
    """
    使用LLaMA-Factory的内部API运行DPO训练
    """
    print("开始DPO训练...")
    
    # 设置配置文件路径
    config_path = "examples/train_lora/qwen1_5_0_5b_lora_dpo_test.yaml"
    
    try:
        # 保存原始的命令行参数
        original_argv = sys.argv.copy()
        
        # 模拟命令行方式调用，设置sys.argv
        sys.argv = [sys.argv[0]] + [config_path]
        
        # 直接运行训练
        # 这个函数会自动从sys.argv读取配置文件
        run_exp()
        
        # 恢复原始的命令行参数
        sys.argv = original_argv
        
        print("DPO训练完成！")
        return 0
    
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return 1


if __name__ == "__main__":
    sys.exit(main()) 