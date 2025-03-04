#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA-Factory 命令行接口 (CLI) 实现
==================================

本文件实现了LLaMA-Factory框架的命令行接口，作为用户与框架交互的主要入口点。
脚本解析用户输入的命令并调用相应的功能模块执行特定任务。

工作原理:
--------
1. 用户通过`llamafactory-cli <命令> [参数]`形式调用
2. main()函数解析第一个参数(命令)，并根据Command枚举类分发到相应处理函数
3. 对于训练命令(train)，会调用tuner.py中的run_exp函数，根据配置文件执行不同训练流程

支持的主要命令:
-------------
- train: 训练模型（包括SFT、RM、DPO、PPO等多种训练方式）
- eval: 评估模型
- chat: 命令行聊天界面
- export: 合并LoRA适配器并导出模型
- webchat: Web聊天界面
- webui: 启动LlamaBoard界面
- api: 启动OpenAI风格API服务器

DPO训练原理与流程:
---------------
DPO(Direct Preference Optimization)是一种基于人类偏好的训练方法，不同于PPO，它无需训练奖励模型。

DPO训练流程:
1. 命令行输入: llamafactory-cli train <配置文件.yaml>，其中配置指定stage=dpo
2. cli.py将命令转发到tuner.py中的run_exp函数
3. 根据stage=dpo参数，调用train/dpo/workflow.py中的run_dpo函数
4. run_dpo函数执行以下核心步骤:
   - 加载tokenizer和模板
   - 准备成对偏好数据集(包含query、chosen和rejected响应)
   - 加载模型和参考模型(reference model)
   - 初始化CustomDPOTrainer并执行训练
   - 训练过程优化策略模型使其生成的响应更接近人类偏好的响应

DPO算法核心:
- 不直接最大化奖励，而是最小化策略与参考策略之间的KL散度
- 目标函数结合了生成人类偏好响应的概率和保持接近参考模型的平衡
- DPO损失函数: L = -log(σ(β·(r_w(x,y_w) - r_w(x,y_l))))，其中:
  * σ是sigmoid函数
  * β是温度参数(通常为0.1-1)
  * r_w是模型对输入x生成响应y的隐式奖励
  * y_w是人类偏好的响应，y_l是人类不偏好的响应
  
实现特点:
- 使用成对数据(chosen/rejected)进行训练
- 训练更稳定，不需要复杂的PPO强化学习环境
- 效果与PPO相当但训练速度更快，参数调优更简单
"""

import os
import random
import subprocess
import sys
from enum import Enum, unique

from . import launcher
from .api.app import run_api
from .chat.chat_model import run_chat
from .eval.evaluator import run_eval
from .extras import logging
from .extras.env import VERSION, print_env
from .extras.misc import get_device_count, is_env_enabled, use_ray
from .train.tuner import export_model, run_exp
from .webui.interface import run_web_demo, run_web_ui


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli eval -h: evaluate models                        |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to LLaMA Factory, version {VERSION}"
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
    + "-" * 58
)

logger = logging.get_logger(__name__)


@unique
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    ENV = "env"
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    VER = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.API:
        run_api()
    elif command == Command.CHAT:
        run_chat()
    elif command == Command.ENV:
        print_env()
    elif command == Command.EVAL:
        run_eval()
    elif command == Command.EXPORT:
        export_model()
    elif command == Command.TRAIN:
        force_torchrun = is_env_enabled("FORCE_TORCHRUN")
        if force_torchrun or (get_device_count() > 1 and not use_ray()):
            master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
            master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
            logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=os.getenv("NNODES", "1"),
                    node_rank=os.getenv("NODE_RANK", "0"),
                    nproc_per_node=os.getenv("NPROC_PER_NODE", str(get_device_count())),
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split()
            )
            sys.exit(process.returncode)
        else:
            run_exp()
    elif command == Command.WEBDEMO:
        run_web_demo()
    elif command == Command.WEBUI:
        run_web_ui()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
