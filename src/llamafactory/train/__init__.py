#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA-Factory 训练模块
=====================

包含各种训练方法的模块。
"""

from typing import Dict, Type, Optional, Union, get_args, get_origin, Any, Callable

from ..hparams import FinetuningArguments

# 支持的训练方法
try:
    from .dpo import run_dpo
except ImportError:
    run_dpo = None

try:
    from .kto import run_kto
except ImportError:
    run_kto = None

try:
    from .ledpo import run_ledpo
except ImportError:
    run_ledpo = None

try:
    from .ppo import run_ppo
except ImportError:
    run_ppo = None

try:
    from .pt import run_pt
except ImportError:
    run_pt = None

try:
    from .rm import run_rm
except ImportError:
    run_rm = None

try:
    from .sft import run_sft
except ImportError:
    run_sft = None

# 训练方法映射表
_TRAINING_METHODS = {
    "sft": run_sft,
    "rm": run_rm,
    "dpo": run_dpo,
    "kto": run_kto,
    "ppo": run_ppo,
    "pt": run_pt,
    "ledpo": run_ledpo,
}
