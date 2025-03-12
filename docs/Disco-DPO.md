 The preference of action \(a_1\) over action \(a_2\) in state \(s\), denoted as \(P(a_1 \succ a_2| s)\), is calculated using the cumulative distribution function (CDF) of the standard normal distribution, \(\Phi\), as follows:

\begin{equation}
    P(a_1 \succ a_2| s) = \Phi\left(\frac{\mu(a_1; s) - \mu(a_2; s)}{\sqrt{\sigma^2(a_1; s) + \sigma^2(a_2; s)}}\right)
\end{equation}

Here, \(\Phi\) represents the CDF of the standard normal distribution, and the argument represents the standardized difference in the expected rewards of actions \(a_1\) and \(a_2\) considering their variances, indicating the probability of preferring \(a_1\) over \(a_2\) given the state \(s\).


我们可以设置 方差都是 1.