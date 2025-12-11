# S-DNA Order 4: Benchmarks Research Report (Gemini)

_Generated: 2024-12-10_
_Status: COMPLETE_

---

## Executive Summary

Before deploying complex neural networks (LSTM, Transformer), we must establish a **performance floor** using statistical baselines:

1. **Random Walk** — Tests Efficient Market Hypothesis
2. **SMA(20)** — Tests trend-following/smoothing
3. **ARIMA(5,1,0)** — Tests linear autocorrelation

If sophisticated models can't beat these, the complexity is unjustified.

---

## Theoretical Framework

### Random Walk (Null Hypothesis)

$$\hat{P}_{t+1} = P_t$$

The simplest baseline. If this wins, the market is efficient and price history contains no predictive signal.

### Simple Moving Average (N=20)

$$\hat{P}_{t+1} = \frac{1}{N}\sum_{i=0}^{N-1} P_{t-i}$$

Low-pass filter. Introduces ~10 day lag. Will fail in volatile trends but may anchor in mean-reverting markets.

### ARIMA(5,1,0)

- **AR(5)**: Uses 5 lags (one trading week)
- **I(1)**: Differences data for stationarity
- **MA(0)**: Pure autoregressive on differenced data

Tests: "Do price changes correlate with previous changes?"

---

## Critical Implementation Detail

### ARIMA: Filtered One-Step-Ahead (NOT Static Forecast)

**Wrong approach**: Fit on train, then forecast N_test steps ahead → Error compounds

**Correct approach**: Use `model_fit.apply(test)`

- Parameters (φ) frozen from training
- Kalman Filter updates state using actual test observations
- Each prediction uses real previous close, not previous prediction

```python
# Fit on train
model = ARIMA(train, order=(5,1,0))
fitted = model.fit()

# Apply to test (filtered, not static)
new_results = fitted.apply(test)
predictions = new_results.predict(start=0, end=len(test)-1)
```

---

## Data Handling

### SPY (Daily)

- 80/20 chronological split (NO shuffle)
- Forward-fill gaps
- Use Adjusted Close if available

### BTC (1-min → 1-hour resample)

- `df.resample('1H').ohlc()` reduces 1M+ points to ~17K
- Preserves intra-day structure
- Makes ARIMA computationally tractable

---

## Evaluation Metrics

| Metric                   | Formula             | Purpose                               |
| ------------------------ | ------------------- | ------------------------------------- |
| **RMSE**                 | √(Σ(y-ŷ)²/n)        | Penalizes large errors (blow-up risk) |
| **MAE**                  | Σ\|y-ŷ\|/n          | Linear error (outlier robust)         |
| **Directional Accuracy** | % correct direction | Trading utility (50% = random)        |

---

## Output Schema

### predictions\_[asset].csv

| Column          | Description             |
| --------------- | ----------------------- |
| Date            | Timestamp               |
| Actual_Close    | Observed price          |
| Pred_RandomWalk | P(t-1)                  |
| Pred_SMA        | Mean of t-20...t-1      |
| Pred_ARIMA      | Filtered one-step-ahead |

### metrics_summary.csv

| Asset | Model       | RMSE | MAE | Directional_Accuracy |
| ----- | ----------- | ---- | --- | -------------------- |
| SPY   | Random Walk | ...  | ... | ...                  |
| SPY   | SMA         | ...  | ... | ...                  |
| SPY   | ARIMA       | ...  | ... | ...                  |
| BTC   | Random Walk | ...  | ... | ...                  |
| ...   | ...         | ...  | ... | ...                  |

---

## Expected Outcomes

1. **Random Walk will be hard to beat on RMSE** — SPY is highly efficient
2. **SMA will have worst RMSE** — Lag causes massive errors in trends
3. **ARIMA is the true test** — If it beats Random Walk, linear predictability exists

### BTC Caveat

ARIMA assumes constant variance. BTC's volatility clustering will cause fat-tailed residuals, setting stage for GARCH models later.

---

_End of Research_
