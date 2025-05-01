# 开发总纲：我的定制化偏好对齐算法 (dev 分支)

## 1. 背景与反思：为何重新出发？

经过一段时间在偏好对齐领域的探索，尤其回顾我在 `xdpo` 目录下进行的 LeDPO 和 DiscoDPO 相关实验，我深刻感受到之前的实现方式存在一些问题。主要是**逻辑耦合、设计不够清晰、职责分离做得不好**，导致后续迭代和理解都变得困难，有点"历史负担"的感觉。同时，标准的对齐方法（如 DPO）也存在其固有的局限性，比如奖励表示过于简化、策略权衡不够灵活等。

因此，我决定在 LLaMA Factory 这个强大的框架基础上，**彻底重新开始**。我需要一个全新的起点 (`dev` 分支)，来构建一套真正符合我设想的、**逻辑清晰、模块化、灵活且易于测试**的定制化偏好对齐算法。

## 2. 我的核心需求与设计愿景

这次开发的核心目标，不仅仅是复现或改进算法，更是要**深入贯彻我对偏好对齐底层逻辑的理解**。具体来说，包含以下几个关键点：

*   **概率化奖励 (Disco 逻辑)**: 我认为奖励信号天然带有不确定性，需要用概率分布（如高斯）来建模，而不是简单的标量。这就要求相关的模型架构（通过 **Disco Head**）和损失函数（基于分布比较）都要跟上。
*   **自适应权衡 (动态 Beta / LeDPO 逻辑)**: 平衡先验知识和新反馈的 \(\beta\) 参数必须是动态的、上下文相关的。并且，这个决策逻辑应该发生在**参考模型**评估上下文之后，通过在其上添加 **Beta Head** 输出一个作用于**基准 `pref_beta`** 的调节因子来实现。
*   **架构与使用的彻底解耦**: 这次设计的重中之重！必须能分开控制**模型架构中是否有某个 Head** 和**算法运行时是否使用这个 Head 的输出**。这对于模块化开发、调试和严谨的对比实验至关重要。

经过与 AI 助手的反复讨论和迭代，我们共同明确了实现这些理念的**详细参数设计方案**。这份详细的技术蓝图记录在 [`parameter_design.md`](./parameter_design.md) 中（该文档会随着设计的深入而持续更新），它准确地反映了上述需求，并遵循了解耦原则。

同时，我更宏观的开发需求和设计哲学，也沉淀在了 [`../rules/my_preference_alignment_needs.md`](../rules/my_preference_alignment_needs.md) 文档中。

## 3. 接下来的步骤：结构化实现

有了清晰的设计蓝图和明确的需求，接下来的开发工作将遵循**模块化、分阶段**的方式进行。我计划将整个实现过程分解为一系列更小的、功能独立的开发步骤（可能对应不同的特性分支，待后续规划），确保每一步都目标明确、易于验证。

初步设想的开发顺序大致如下（优先级和具体分支待细化）：

0.  **跑通基础 DPO Baseline** **(已完成)**: 
    *   **目标**: 确保 LLaMA Factory 中标准的 DPO 流程能够在本分支顺利运行，无需任何定制化修改。
    *   **作用**: 建立一个可工作的基准，验证基础环境和流程，为后续定制化开发提供参照。
    *   **成果**: 获得了可运行的标准 DPO 脚本 (`custom/run_dpo_baseline.py`) 和配置文件 (`custom/dpo_baseline.yaml`)，位于 `custom` 目录下。

1.  **实现基础架构变更**: 
    *   **添加 Disco Head**: 实现 `add_disco_head` 参数功能，让模型具备输出分布参数的能力。
    *   **添加 Beta Head**: 实现 `add_beta_head` 参数功能，在参考模型上添加默认实现的 Beta Head。
2.  **实现核心算法逻辑**: 
    *   **Disco 损失逻辑启用**: 实现当 `disco_pref: true` 时，**使用** Disco Head 输出进行损失计算的逻辑。
    *   **动态 Beta 应用启用**: 实现当 `use_dynamic_beta: true` 时，**使用** Beta Head 输出调节 `pref_beta` 的逻辑。
3.  **整合与测试**: 将各个模块整合，进行全面的单元测试、集成测试和效果验证实验。
4.  **文档完善**: 随着开发的进行，持续更新相关文档，记录实现细节和遇到的问题。

具体的每一个开发步骤（或特性分支），我都会遵循 [`../rules/doc_style_pref.md`](../rules/doc_style_pref.md) 中定义的偏好，创建详细的说明文档（存放于 `docs/` 目录下），包含目标、实现思路、涉及的参数和必要的背景回顾。

总之，`dev` 分支的目标就是系统性地、高质量地完成上述开发过程，最终得到一套令人满意的、真正体现我设计思想的定制化对齐工具。 





## 附录







我们虽然是定制对齐算法, 但是目前切入点是 DPO 算法. 在咱们这个标准的 DPO Baseline 总算是跑起来了，虽然中间折腾了不少环境问题，但好歹有了一个干净的起点。这很重要，因为接下来咱们要干的，才是这次重构的核心. 咱们 V7.4 设计里定下来的、**逻辑清晰、模块化**的目的, 解耦旧版本 disco_pref, use_dynamic_beta 对应功能. 

我最不能忍受的就是之前那种"一锅烩"的做法，加个 Head 和用这个 Head 的逻辑搅在一起，太乱了！所以，这次的**核心目标**，就是要**严格执行咱们在 `parameter_design.md` 里反复强调的"架构与使用解耦"原则**。添加 Head 就是添加 Head，用不用它是另一回事，必须分开控制！

那么，按照咱们 `dev_plan.md` 里的计划，接下来要做的就是：

**第一步：先把"骨架"搭起来 (实现基础架构变更)**

*   **只加 Disco Head**: 去改模型加载那部分代码。当 `add_disco_head: true` 的时候，就在当前模型 (\(\pi_{\theta}\) 或 RM 阶段的 \(\pi_{\text{RM}}\)) 上把 Disco Head 加上去，让它有能力输出那个奖励分布的 \(\mu\) 和 \(\sigma^2\)。**关键是**：加了这个 Head，不代表 DPO 损失就自动变了，算法逻辑先不动！
*   **只加 Beta Head**: 同样，改代码。当 `add_beta_head: true` 的时候，就在**参考模型** (\(\pi_{\text{ref}}\)) 上把那个默认实现的 Beta Head 装上，让它能输出那个调节因子 \(s(x)\)。**同样关键**：加了这个 Head，不意味着 DPO 计算 \(\beta\) 的方式就自动变了，算法逻辑也先别碰！

**第二步：再让"灵魂"能选择性附体 (实现核心算法逻辑的启用)**

*   **实现 Disco 损失的"开关"**: 去改 DPO (或者 RM) 的损失计算部分。代码里必须加个判断，**检查 `disco_pref` 这个参数是不是 `true`**。如果是 `true`，并且模型确实有 Disco Head，那好，就调用 Disco Head 的输出，用咱们设计的 erf/NDTR 方式算损失；如果是 `false`，那就必须老老实实还用原来的 Sigmoid 损失，**就算模型上有 Disco Head 也不用它**！
*   **实现动态 Beta 的"开关"**: 类似地，去改 DPO (或者 PPO) 里计算或使用 \(\beta\) 的地方。也加个判断，**检查 `use_dynamic_beta` 是不是 `true`**。如果是 `true`，并且参考模型上有 Beta Head，那就去拿那个调节因子 \(s(x)\)，然后用 `pref_beta * s(x)` 作为实际的 \(\beta_{\text{eff}}(x)\)；如果是 `false`，就直接用 `pref_beta` 这个固定值，**就算参考模型上有 Beta Head 也无视它**！

总而言之，接下来的开发就是要严格按照这个"先搭骨架（只加 Head），再加开关（独立控制使用）"的思路来，把 V7.4 设计的精髓——**解耦**——真正在代码层面落地。这样搞下来，才是我想要的那个干净、灵活、好理解、好测试的定制化对齐功能。


另外请注意: 当前代码的版本已经使用 disco_pref, use_dynamic_beta 实现对应功能, 我只是想要把这个实现改成更好的版本. 所以第1步你一定要搞清楚, 原来的版本是怎么实现的有关细节!!! 你通过对比 origin main 和 upstream main src/llamafactory 差异就能发现如何具体实现的细节.

## 进展记录 (2025-05-02)

今天主要围绕跑通 RM 和 PPO 基线实验展开，为后续定制化开发（DiscoDPO, LeDPO 等）打下基础。

*   **奖励模型 (RM) 基线:**
    *   创建并运行了 `custom/rm_baseline.yaml` 和 `custom/run_rm_baseline.py`。
    *   模型: `/root/models/Qwen3-0.6B`，数据集: `hh_rlhf_en` (成对偏好数据)。
    *   解决了 RM 评估指标未显示的问题 (调整 `max_samples` 和 `eval_steps`)，明确了 `loss` 与 `accuracy` 的区别。
    *   RM 训练成功完成。
*   **PPO 基线:**
    *   创建了 `custom/ppo_baseline.yaml` 和 `custom/run_ppo_baseline.py`。
    *   配置使用已训练的 RM checkpoint (`.../rm_baseline/checkpoint-45`)。
    *   **多次调试解决 PPO 启动问题:**
        *   修正 Data Collator (使用 `transformers.DataCollatorWithPadding`，移除无效参数)。
        *   修正 RM 加载逻辑 (理解 LoRA checkpoint 结构，正确设置 `model_name_or_path` 和 `adapter_name_or_path`)。
        *   修正 PPO 配置 (添加 `reward_model_type: full`)，解决了 PPO Trainer 内部逻辑分支错误。
        *   修正 PPO 数据集配置 (使用 `alpaca_en_demo` 替代 `hh_rlhf_en`)。
    *   PPO 训练目前已成功启动，正在运行中 (存在暂不处理的 `Trainer.tokenizer` DeprecationWarning)。
*   **新文件:** `custom/rm_baseline.yaml`, `custom/run_rm_baseline.py`, `custom/ppo_baseline.yaml`, `custom/run_ppo_baseline.py`。