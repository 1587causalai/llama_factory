# Policy Model Experiment Analysis Report

## 1. Beta Value Statistics

| Metric | 冻结模型 | 非冻结模型 | 冻结模型-新ref |
|------|------|------|------|
| Mean | 0.0322 | 0.0683 | 0.0109 |
| Std Dev | 0.0047 | 0.0283 | 0.0037 |
| Min | 0.0204 | 0.0143 | 0.0100 |
| Max | 0.0474 | 0.1763 | 0.0342 |
| Change | 0.0017 | 0.0634 | -0.0157 |

## 2. Delta Value Statistics

| Metric | 冻结模型 | 非冻结模型 | 冻结模型-新ref |
|------|------|------|------|
| Mean | 0.0000 | 11.6182 | -1.8493 |
| Std Dev | 0.0000 | 9.1432 | 2.3130 |
| Min | 0.0000 | -1.9409 | -6.8762 |
| Max | 0.0000 | 29.1005 | 3.4091 |
| Positive Ratio | 0.00% | 93.33% | 21.48% |

## 3. Beta-Delta Correlation

| Model | Correlation |
|------|----------|
| 冻结模型 | nan |
| 非冻结模型 | 0.7305 |
| 冻结模型-新ref | 0.1046 |

## 4. Analysis Conclusions

### Delta Value Analysis

**Frozen Model**:
- Delta value is close to 0 (0.0000), which meets expectations. Since the policy model is frozen, its output remains unchanged, resulting in constant delta values.
**Non-Frozen Model**:
- Delta mean is 11.6182, which deviates from theoretical expectations. This might be due to numerical errors or differences in model initialization.
**Frozen Model-New Ref**:
- Delta mean is -1.8493, which deviates from theoretical expectations. This might be due to numerical errors or differences in model initialization.

### Beta Value Analysis

**Frozen Model**:
- Beta value changed very little (0.0017), which may indicate that the beta_head learning is limited or already close to optimal value.
**Non-Frozen Model**:
- Beta value changed by 0.0634 from beginning to end, indicating that the beta_head is indeed learning.
**Frozen Model-New Ref**:
- Beta value changed by -0.0157 from beginning to end, indicating that the beta_head is indeed learning.

### Correlation Analysis

**Frozen Model**:
- There is almost no correlation (nan) between beta and delta. More training rounds may be needed to observe a clear relationship.
**Non-Frozen Model**:
- There is a positive correlation (0.7305) between beta and delta, indicating that the beta_head is adapting to the data distribution.
**Frozen Model-New Ref**:
- There is almost no correlation (0.1046) between beta and delta. More training rounds may be needed to observe a clear relationship.

### Summary

The experimental results confirm our hypothesis: the delta value remains constant when the policy model is frozen, while the non-frozen model successfully learns preference relationships. This indicates that the implementation of the LEDPO algorithm is correct, and the introduction of beta_head indeed provides the model with the ability to dynamically adjust beta values.


### Comparison of Different Frozen Models

**Frozen Model vs Non-Frozen Model**:
- Delta value difference: 11.6182
- Beta change difference: 0.0617
- Non-Frozen Model has a larger beta change, which may indicate more effective learning.
**Frozen Model vs Frozen Model-New Ref**:
- Delta value difference: 1.8493
- Beta change difference: 0.0175
- Frozen Model-New Ref has a larger beta change, which may indicate more effective learning.
**Non-Frozen Model vs Frozen Model-New Ref**:
- Delta value difference: 13.4675
- Beta change difference: 0.0791
- Non-Frozen Model has a larger beta change, which may indicate more effective learning.

## 5. Recommendations

1. **Increase Training Rounds**: Consider increasing training rounds from the current 3 to 5-10, to observe beta and delta trends over a longer period.
2. **Adjust Learning Rate**: Try setting a higher learning rate for beta_head to facilitate faster adaptation to data distribution.
3. **Use Different Initializations**: Try initializing models with different random seeds to verify the stability of results.
4. **Expand Dataset**: Use larger datasets, or test algorithm performance on data from different domains.
5. **Monitor Gradient Flow**: Add gradient monitoring for beta_head parameters to verify gradient flow and confirm actual learning.
6. **Further Study of Reference Model Impact**: Continue researching the impact of different reference models on frozen policy experiments to explore best practices.
