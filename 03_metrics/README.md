# Metrics & League Tables

Performance measurement and comparative rankings.

## Metric Matrix

| Capability | Metric                  | Threshold | Commercial Rationale   |
| ---------- | ----------------------- | --------- | ---------------------- |
| Trend      | Directional Accuracy    | > 55%     | Better than coin flip  |
| Trend      | Information Coefficient | > 0.05    | Institutional standard |
| Trend      | RankIC                  | > 0.05    | Robust to outliers     |
| Reversal   | Precision               | > 60%     | Minimize false alarms  |
| Reversal   | Recall                  | > 70%     | Catch most reversals   |
| Reversal   | F1                      | > 0.65    | Balanced performance   |
| Reversal   | MTTD                    | < 3 bars  | Beat SMA lag by 30%    |
| Anomaly    | AUC-ROC                 | > 0.80    | Strong discrimination  |
| Anomaly    | F-Beta (β=2)            | > 0.75    | Recall-weighted        |
| Anomaly    | FPR                     | < 10%     | Avoid cry-wolf         |
| Risk       | Sharpe Ratio            | > 1.0     | Risk-adjusted returns  |
| Risk       | Max Drawdown            | < 20%     | Capital preservation   |

## League Table Format

Results will be populated after benchmark runs complete.

| Model       | DA  | IC  | F1  | MTTD | AUC | Sharpe | Max DD | Rank |
| ----------- | --- | --- | --- | ---- | --- | ------ | ------ | ---- |
| S-DNA       | -   | -   | -   | -    | -   | -      | -      | -    |
| Random Walk | -   | -   | -   | -    | -   | -      | -      | -    |
| ...         | ... | ... | ... | ...  | ... | ...    | ...    | ...  |
