#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LEDPO (Learnable Beta DPO) 训练模块
==================================

此模块实现了可学习权重β的DPO训练算法，使模型能够自适应调整loss中的β参数
"""

# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .trainer import LEDPOTrainer
from .workflow import run_ledpo


__all__ = ["run_ledpo"]
