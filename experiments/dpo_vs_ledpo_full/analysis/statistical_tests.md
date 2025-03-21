# Statistical Significance Tests

This report shows the results of statistical tests comparing DPO and LEDPO metrics.


## Training Metrics

| Metric | p-value | Significant at α=0.05 | Better Model |
|--------|---------|------------------------|-------------|
| accuracy | 0.000001 | Yes | DPO |
| reward_rejected | 0.000000 | Yes | LEDPO |
| reward_margin | 0.000000 | Yes | DPO |
| reward_chosen | 0.000161 | Yes | LEDPO |
| loss | 0.000000 | Yes | DPO |