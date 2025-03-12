# 优化 LearnableBetaDPO 损失函数是否会导致 \(\beta(x)\) 越小越好？

## 背景
我们考虑以下损失函数（LearnableBetaDPO）：

\[
\mathcal{L}_{\text{LearnableBetaDPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta(x; \pi_\theta) \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right] \right) \right]
\]

其中：
- \(\pi_\theta\)：当前策略，\(\pi_{\text{ref}}\)：参考策略。
- \(x\)：输入，\(y_w\)：优选输出，\(y_l\)：劣选输出。
- \(\beta(x; \pi_\theta)\)：可学习参数，依赖于 \(x\) 和 \(\pi_\theta\)。假设 \(\beta(x) > 0\)。 
- \(\sigma(z) = \frac{1}{1 + e^{-z}}\)：sigmoid 函数。
- 目标：最小化 \(\mathcal{L}\)。 

问题：优化此损失函数是否会导致 \(\beta(x)\) 越小越好？

## 核心结论
优化 \(\mathcal{L}_{\text{LearnableBetaDPO}}\) **不会导致 \(\beta(x)\) 越小越好**。具体而言：
1. 若 \(\beta(x) \to 0\)，损失收敛到次优值（例如 \(-\log 2\)），无法有效区分优选和劣选输出，违背 DPO 的目标。
2. \(\beta(x)\) 需要足够大以放大正确偏好（\(\Delta > 0\)），但若过大，可能因错误偏好（\(\Delta < 0\)）导致损失爆炸。
3. \(\beta(x)\) 的最优值趋向一个适中范围，平衡正确和错误偏好的影响，具体值取决于数据集 \(\mathcal{D}\) 中 \(\Delta\) 的分布。

## 证明

### 定义与简化
定义相对偏好差异：

\[
\Delta = \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\]

- \(\Delta > 0\)：\(\pi_\theta\) 更偏好 \(y_w\)（正确）。
- \(\Delta < 0\)：\(\pi_\theta\) 更偏好 \(y_l\)（错误）。

损失函数简化为：

\[
\mathcal{L} = - \mathbb{E} \left[ \log \sigma \left( \beta(x) \cdot \Delta \right) \right]
\]

最小化 \(\mathcal{L}\) 等价于最大化 \(\mathbb{E} \left[ \log \sigma (\beta(x) \cdot \Delta) \right]\)。由于 \(\log \sigma(z)\) 单调递增，等价于最大化 \(\sigma(\beta(x) \cdot \Delta)\) 的期望。

### 分析 \(\beta(x)\) 的影响
设 \(z = \beta(x) \cdot \Delta\)，则 \(\sigma(z) \in (0, 1)\)，\(\log \sigma(z) \in (-\infty, 0)\)。我们分情况分析 \(\beta(x)\) 对损失的贡献。

#### 情况 1：\(\Delta > 0\)（正确偏好）
- \(z = \beta(x) \cdot \Delta > 0\)。
- \(\beta(x)\) 增大，\(z\) 增大，\(\sigma(z) \to 1\)，\(\log \sigma(z) \to 0\)，损失项 \(- \log \sigma(z) \to 0\)（变小）。
- \(\beta(x) \to 0\)，\(z \to 0\)，\(\sigma(0) = 0.5\)，\(\log \sigma(0) = -\log 2 \approx -0.693\)，损失项变为 \(\log 2\)（变大）。

**结论**：\(\beta(x)\) 越大，损失越小；\(\beta(x)\) 越小，损失越大。

#### 情况 2：\(\Delta < 0\)（错误偏好）
- \(z = \beta(x) \cdot \Delta < 0\)。
- \(\beta(x)\) 增大，\(z\) 更负，\(\sigma(z) \to 0\)，\(\log \sigma(z) \to -\infty\)，损失项 \(- \log \sigma(z) \to +\infty\)（爆炸）。
- \(\beta(x) \to 0\)，\(z \to 0\)，\(\sigma(z) \to 0.5\)，损失项 \(\to \log 2\)（有限值）。

**结论**：\(\beta(x)\) 越大，损失越大；\(\beta(x)\) 越小，损失越小。

#### 情况 3：\(\beta(x) \to 0\)（极限情况）
- 无论 \(\Delta > 0\) 或 \(\Delta < 0\)，\(\beta(x) \to 0\) 使 \(z \to 0\)。 
- \(\sigma(0) = 0.5\)，\(\log \sigma(0) = -\log 2\)。 
- 损失变为：

\[
\mathcal{L} \to - \mathbb{E} [-\log 2] = \log 2 \approx 0.693
\]

- 这是一个固定值，模型不再区分 \(y_w\) 和 \(y_l\)，失去优化动力。

### 优化动态
- **\(\beta(x) \to 0\)**：损失固定为 \(\log 2\)，次优，无法进一步降低。
- **\(\beta(x)\) 过大**：若 \(\Delta < 0\) 的样本存在，损失可能趋向无穷，迫使 \(\theta\) 调整使 \(\Delta > 0\)。 
- **平衡**：\(\beta(x)\) 被优化到一个适中值：
  - 足够大以放大 \(\Delta > 0\) 的贡献（降低损失）。
  - 不至于过大以避免 \(\Delta < 0\) 的惩罚（损失爆炸）。

### 数学验证
对 \(\beta(x)\) 求偏导（假设 \(\beta(x)\) 可独立优化）：

\[
\frac{\partial}{\partial \beta(x)} \log \sigma (\beta(x) \cdot \Delta) = \frac{\partial}{\partial z} \log \sigma(z) \cdot \Delta = (1 - \sigma(z)) \cdot \Delta
\]

- \(\Delta > 0\)：梯度正，\(\beta(x)\) 增大。
- \(\Delta < 0\)：梯度负，\(\beta(x)\) 减小。
- 总梯度 \(\mathbb{E}[(1 - \sigma) \cdot \Delta]\) 取决于 \(\Delta\) 的分布，\(\beta(x)\) 会收敛到平衡点。

## 最终结论
\(\beta(x)\) 不会无限制变小，而是趋向一个适中值，具体由 \(\mathcal{D}\) 中 \(\Delta\) 的正负分布决定。优化目标要求 \(\beta(x)\) 在增强正确偏好和避免错误惩罚间取得平衡。




## 实验

对于分类正确的样本 (Δ > 0)：
- beta(x) 增大，损失减小
- beta(x) 减小，损失增大

对于分类错误的样本 (Δ < 0)：
- beta(x) 增大，损失增大
- beta(x) 减小，损失减小

所以理论上，如果我们冻结策略模型参数（不更新模型本身），那么：
1. 对于分类正确的样本，beta(x) 会倾向于增大
2. 对于分类错误的样本，beta(x) 会倾向于减小