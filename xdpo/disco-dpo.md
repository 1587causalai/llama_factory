# README

这个算法是基于DPO算法，但是做了一些改进，主要改进点在于：Reward Model 的输出是一个随机变量, 而不是一个确定值。

## 算法原理


$$r(x, y) \sim N(\mu(x, y), \sigma(x, y))$$ 

其中，$r(x, y)$ 是 Reward Model 的输出，$\mu(x, y)$ 和 $\sigma(x, y)$ 是均值和方差。为了简单起见, 我们假设 $\sigma(x, y)$ 是一个常数 1.0。

那么偏好概率的计算方式为:

$$p(y_w > y_l) = \frac{1}{2} \left(1 + \text{erf}\left(\frac{\mu(x, y_w) - \mu(x, y_l)}{\sqrt{2}}\right)\right)$$



