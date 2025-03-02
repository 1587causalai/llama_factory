# Prompts



## Raw Prompts

我想新建一个分支, 用于专门对这个项目的深入了解, 当然是带着我的那个研究问题去了解的, 我这个分支的专注点, 是要把 post-training 一次向前传播的细节每一步都搞得清清楚楚, 然后再来定制我自己的算法, 所以我需要把这个架构搞得非常清楚，数据流搞得非常清楚。我们一起来看一看怎么做吧，构思一下怎么行动吧。



 帮我实现一个 fooDPO 算法吧, 嗯，所有的具体和原来的一致, 只是名字不一样, 未接下来可能需要的算法定制做准备, 这个东西的目的是要跑通一次，向前传播计算损失。我们思考一下需要做哪些事情吧?




现在我们要开始定制我们的算法了，先从简单的地方开始吧。标准 DPO Loss 函数：

$$\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right] \right) \right]$$

其中 $\sigma(z) = (1 + e^{-z})^{-1}$ 为 sigmoid 函数，$\mathcal{D}$ 为偏好数据集。

我首先想实现一个 fooDPO 的算法，这个算法是标准 DPO 的变体，它使用 $\beta(x)$ 来替代 $\beta$，其中 $\beta(x)$ 是一个关于输入 $x$ 的函数:

$$\beta(x) =  c \cdot \log \text{PPL}(x)  \cdot \beta$$

where $\text{PPL}(x)$ is the perplexity of the model on the input $x$, and $c$ is a constant. 

你先构思一下我们该怎么实现吧。



第一种修改数据加载器的方式会不会更底层一点, 更好控制, 更好定制, 但是担心和其他方法不兼容. 所以我想想有没有可能快速的计算一个 prompt mask 用于相关计算吗? 
