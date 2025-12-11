# 03_metrics — Performance Measurement

This directory defines the metric framework for S-DNA validation.

## Metric Matrix

### Signal Quality

| Metric | Formula | Target | Rationale |
|--------|---------|--------|-----------|
| Directional Accuracy | % correct direction | > 55% | Better than coin flip |
| Information Coefficient | Corr(ŷ, y) | > 0.05 | Institutional standard |
| RankIC (Spearman) | Rank correlation | > 0.05 | Robust to outliers |

### Reversal Detection

| Metric | Target | Rationale |
|--------|--------|-----------|
| Precision | > 60% | Minimize false alarms |
| Recall | > 70% | Catch most reversals |
| F1 Score | > 0.65 | Balanced performance |
| MTTD | < 3 bars | Beat SMA lag by 30% |

### Anomaly Detection

| Metric | Target | Rationale |
|--------|--------|-----------|
| AUC-ROC | > 0.80 | Strong discrimination |
| F-Beta (β=2) | > 0.75 | Recall-weighted |
| False Positive Rate | < 10% | Avoid cry-wolf |

### Risk-Adjusted

| Metric | Target | Rationale |
|--------|--------|-----------|
| Sharpe Ratio | > 1.0 | Risk-adjusted returns |
| Max Drawdown | < 20% | Capital preservation |

## Statistical Tests

| Test | Purpose | Threshold |
|------|---------|-----------|
| **Diebold-Mariano** | Model comparison | p < 0.05 |
| **Paired t-test** | Mean difference | p < 0.05 |

## League Table (To Be Populated)

| Model | DA | IC | F1 | MTTD | AUC | Sharpe | Max DD | Rank |
|-------|----|----|----|----- |-----|--------|--------|------|
| S-DNA | - | - | - | - | - | - | - | - |
| Random Walk | 49.4% | - | - | - | - | - | - | - |
| LSTM | - | - | - | - | - | - | - | - |

*Table will be populated after Phase II LSTM training completes.*
