# S-DNA LSTM Research — Platform Deep Research Output

_Source: Google Platform Deep Research (preliminary)_
_Status: Contains useful material_

---

## Key Architecture Decision: Dual-Mode

> "To beat the divergent baselines (RMSE for regression, Directional Accuracy for classification), a single model is insufficient."

**Mode A (Classifier)**: Targets Directional Accuracy > 52%
**Mode B (Regressor)**: Targets RMSE < 4.54 by predicting Log Returns

---

## Critical Insight: Predict Log Returns, Not Price

Price is non-stationary. Predicting raw price causes the model to learn the trivial solution $P_t ≈ P_{t-1}$.

**Solution:**

1. Predict log return: $\hat{r}_t$
2. Reconstruct price: $\hat{P}_t = P_{t-1} \times e^{\hat{r}_t}$
3. Calculate RMSE on reconstructed price

---

## Failure Diagnostics

### The "Lag Trap"

If prediction looks like actual chart shifted right by 1 day → model is useless (just learned persistence).

**Fix:** Force model to predict returns/differences, never raw values.

### Mode Collapse

If accuracy ≈ 33% (or equals most frequent class ratio) → model just predicts "Neutral" every time.

**Fix:** Increase class_weights penalty for majority class.

### Regime Overfitting

If fails during 2020 crash → model hasn't learned volatility clustering.

**Fix:** Feed VIX or Historical Volatility as direct input.

---

## Diebold-Mariano Test (Ready-to-Use)

```python
from scipy import stats
import numpy as np

def diebold_mariano_test(actual, pred1, pred2, h=1):
    """
    actual: True values
    pred1: Model 1 predictions (LSTM)
    pred2: Model 2 predictions (Baseline)
    h: forecast horizon (1 for next-day)
    """
    e1 = np.array(actual) - np.array(pred1)
    e2 = np.array(actual) - np.array(pred2)
    d = e1**2 - e2**2  # Squared error differential

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    # DM Statistic
    dm_stat = d_mean / np.sqrt(d_var / len(d))

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

# Usage:
# dm_stat, p_value = diebold_mariano_test(y_actual, lstm_preds, baseline_preds)
# if p_value < 0.05: print("LSTM significantly better!")
```

---

## Dual-Head Model Skeleton

```python
def build_lstm_consensus(seq_len, n_features, output_type='classification'):
    input_layer = Input(shape=(seq_len, n_features))

    # Layer 1: Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Layer 2: Bidirectional LSTM
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Dense Head
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)

    if output_type == 'classification':
        output = Dense(3, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
    else:  # regression
        output = Dense(1, activation='linear')(x)
        loss = 'mse'

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=AdamW(learning_rate=1e-4), loss=loss)
    return model
```

---

## S-DNA Source Code Verdict

> **YES, include it.** The "Signal Forge" likely contains pre-calculated features (Z-scores, sentiment aggregates). Deep Learning models are feature-hungry; feeding raw OHLCV is suboptimal when proprietary signals exist.

---

_End of preliminary research extraction_
