# Order 5 Research Synthesis — LSTM Baseline

## Agent Contributions

| Agent      | Lines | Key Insights                                                          |
| ---------- | ----- | --------------------------------------------------------------------- |
| **GPT**    | 112   | Focal loss, bidirectional option, regime-specific evaluation          |
| **Sonnet** | 941   | Complete training script, Keras Tuner, Diebold-Mariano implementation |
| **Gemini** | 121   | Log returns for stationarity, AdamW optimizer, mode collapse warning  |

---

## Consensus Architecture

```
Input: Log returns + Volatility (Z-normalized)
Sequence: 60 timesteps
         ↓
Bidirectional LSTM(128) → BatchNorm → Dropout(0.3)
         ↓
Bidirectional LSTM(64) → BatchNorm → Dropout(0.3)
         ↓
Dense(32, relu) → Dropout(0.2)
         ↓
Dense(3, softmax) → Classes: {Bear, Neutral, Bull}
```

---

## Key Decisions from Research

### 1. Features (Gemini insight)

- **Use log returns, NOT raw prices** — ensures stationarity
- Volatility column already provided
- Optional: RSI, lagged labels

### 2. Regularization (All agents)

- Dropout: 0.3 LSTM, 0.2 Dense
- L2 weight decay: 1e-4
- Early stopping patience: 10-15 epochs

### 3. Class Imbalance (Sonnet + GPT)

```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0,1,2], y=y_train)
```

### 4. Diebold-Mariano Test (Sonnet implementation)

```python
def diebold_mariano_test(actual, pred1, pred2):
    e1 = actual - pred1  # LSTM errors
    e2 = actual - pred2  # ARIMA errors
    d = e1**2 - e2**2
    DM_stat = np.mean(d) / np.sqrt(np.var(d) / len(d))
    p_value = 1 - stats.norm.cdf(DM_stat)
    return DM_stat, p_value
```

### 5. Walk-Forward (GPT + Gemini)

- Expanding window, retrain monthly (30 days)
- Aggregate out-of-sample predictions chronologically

### 6. Failure Diagnostics (All)

- If Train Acc >> Val Acc → Overfitting → Increase dropout
- If both low → Underfitting → Add capacity
- Check for mode collapse (only predicting "0")
- Regime-specific analysis (high vs low vol)

---

## Success Criteria

| Metric                  | Target                          |
| ----------------------- | ------------------------------- |
| RMSE                    | < 4.54 (SPY), < 1322 (BTC)      |
| Directional Accuracy    | > 52% (significantly above 50%) |
| Diebold-Mariano p-value | < 0.05 vs ARIMA                 |

---

## Implementation Priority

1. ✓ Log returns as primary feature
2. ✓ Bidirectional LSTM architecture
3. ✓ Class weights for imbalance
4. ✓ Diebold-Mariano test
5. ○ Walk-forward validation (future enhancement)
