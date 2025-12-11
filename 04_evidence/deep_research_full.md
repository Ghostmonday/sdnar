# S-DNA Deep Learning Research — Full Platform Output

_Source: Google Platform Deep Research_
_Status: COMPREHENSIVE — Use as primary reference_

---

## Executive Summary

The **S-DNA Baseline LSTM** beats Random Walk by:

1. **Volatility-Adaptive Scaling** — homoskedastic inputs
2. **Directional-MSE Loss** — penalizes wrong signs
3. **Purged Walk-Forward Validation** — eliminates data leakage

---

## Key Insight #1: Volatility Normalization (S-DNA Core)

**The critical transformation:**
$$\tilde{x}_t = \frac{r_t}{\sigma_t}$$

This converts heteroskedastic returns to homoskedastic:

- 5% crash in 2008 (High Vol) → -2.0 scaled
- 0.5% drop in 2017 (Low Vol) → -2.0 scaled

**LSTM sees them as structurally identical "downward shocks"**

---

## Key Insight #2: Directional-MSE Loss

Standard MSE doesn't care about sign. Wrong direction = wrong trade.

$$\mathcal{L} = \text{MSE}(y, \hat{y}) + \lambda \cdot \text{SignPenalty}(y, \hat{y})$$

Where:
$$\text{SignPenalty} = \text{ReLU}(- \text{sign}(y \cdot \hat{y})) \cdot |y - \hat{y}|^2$$

- Same sign → no penalty
- Different sign → squared error added
- λ = 0.5 to 1.0

---

## Key Insight #3: Purged Walk-Forward Validation

**Problem:** Triple-Barrier labels at time t contain info from t+10 (barrier horizon).

**Solution:** Purge samples at train/test boundary equal to max barrier horizon.

```
Fold 1:
  Train: Jan 2000 - Dec 2005
  Purge: Remove Jan 2006 (10 trading days)
  Test: Feb 2006 - Dec 2006
```

---

## Architecture Specification

| Component       | Specification           | Rationale                            |
| --------------- | ----------------------- | ------------------------------------ |
| Input           | (Batch, 60, 6)          | 60-day lookback                      |
| Noise Injection | GaussianNoise(std=0.01) | Prevents overfitting to exact values |
| LSTM Layer 1    | Bidirectional(128)      | High capacity                        |
| Regularization  | SpatialDropout1D(0.2)   | Drops entire feature channels        |
| LSTM Layer 2    | LSTM(64)                | Bottleneck compression               |
| Normalization   | BatchNormalization      | Stabilizes gradients                 |
| Dense           | Dense(32, swish)        | Swish > ReLU for regression          |
| Output          | Dense(1, linear)        | Volatility-adjusted return           |

---

## Training Protocol

### Optimizer

- **AdamW** with weight_decay=1e-4
- **Cyclical Learning Rates**: cycle between 1e-5 and 1e-3

### Sample Weighting (Inverse Variance)

$$w_i = \frac{1}{\sigma_i}$$

Down-weights 2008 crash, up-weights 2017 calm. Forces learning of structure, not outliers.

---

## Failure Diagnostics

### 1. The "Lag-1" Failure

Model learned $\hat{y}_{t+1} ≈ y_t$ (just copying previous value).

**Diagnosis:** Autocorrelation between predictions and y\_{t-1} > 0.95
**Remedy:** Increase Dropout to 0.4-0.5, increase GaussianNoise

### 2. Regime Overfitting

Test RMSE is 10x Train RMSE during 2020.

**Cause:** Volatility scaling not applied correctly
**Remedy:** Verify all features divided by σ_t

### 3. Directional Collapse

RMSE is good but DirAcc ≈ 50%. Model predicts small values clustered near 0.

**Remedy:** Increase λ in Directional-MSE loss

---

## Diebold-Mariano Implementation

```python
def dm_test(actual, pred_lstm, pred_rw):
    e_lstm = (actual - pred_lstm)**2
    e_rw = (actual - pred_rw)**2
    d = e_rw - e_lstm
    mean_d = np.mean(d)
    var_d = np.var(d)
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    return dm_stat  # > 1.96 = LSTM significantly better
```

---

## Feature Engineering Protocol

1. **Log-Returns:** $r_t = \ln(P_t / P_{t-1})$
2. **Volatility Normalize:** Divide by σ_t
3. **Winsorize:** Clip at ±5σ
4. **Target:** return_realized (bounded by barrier)

---

_End of Deep Research_
