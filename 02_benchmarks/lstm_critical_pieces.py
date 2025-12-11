"""
S-DNA LSTM — Critical Implementation Pieces

1. Diebold-Mariano Test with HAC (Newey-West) adjustment
2. Optimized Sliding Window Generator (numpy stride_tricks)
3. Dual-Head LSTM with justified loss weighting
4. Regime-based failure diagnostics
"""

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 1. ROBUST DIEBOLD-MARIANO TEST (HAC / Newey-West Adjusted)
# ============================================================

def newey_west_variance(d, max_lag=None):
    """
    Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) variance.
    
    Corrects for autocorrelation in the loss differential series.
    Standard DM test assumes no autocorrelation - financial data violates this.
    """
    n = len(d)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
    
    d_centered = d - np.mean(d)
    gamma_0 = np.sum(d_centered ** 2) / n
    
    weighted_cov = 0
    for k in range(1, max_lag + 1):
        weight = 1 - k / (max_lag + 1)  # Bartlett kernel
        gamma_k = np.sum(d_centered[k:] * d_centered[:-k]) / n
        weighted_cov += 2 * weight * gamma_k
    
    return (gamma_0 + weighted_cov) / n


def diebold_mariano_robust(actual, pred_model, pred_baseline, one_sided=True):
    """
    Robust Diebold-Mariano test with Newey-West HAC standard errors.
    For financial time series where errors are autocorrelated.
    """
    actual = np.asarray(actual).flatten()
    pred_model = np.asarray(pred_model).flatten()
    pred_baseline = np.asarray(pred_baseline).flatten()
    
    e_model = (actual - pred_model) ** 2
    e_baseline = (actual - pred_baseline) ** 2
    d = e_baseline - e_model  # positive = model better
    
    d_mean = np.mean(d)
    var_d = newey_west_variance(d)
    dm_stat = d_mean / np.sqrt(var_d) if var_d > 0 else 0
    
    p_value = 1 - stats.norm.cdf(dm_stat) if one_sided else 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {
        'dm_statistic': round(dm_stat, 4),
        'p_value': round(p_value, 6),
        'significant': p_value < 0.05,
        'interpretation': 'LSTM BETTER' if (p_value < 0.05 and dm_stat > 0) else 'No significant diff'
    }


# ============================================================
# 2. OPTIMIZED SLIDING WINDOW (numpy stride_tricks)
# ============================================================

def create_sequences_optimized(data, targets, seq_len):
    """Memory-efficient sliding window using numpy stride_tricks."""
    from numpy.lib.stride_tricks import sliding_window_view
    
    n_samples, n_features = data.shape
    X_view = sliding_window_view(data, (seq_len, n_features))
    X = X_view.squeeze(axis=1)
    y = targets[seq_len - 1:]
    
    min_len = min(len(X), len(y))
    return X[:min_len].copy(), y[:min_len].copy()


def walk_forward_generator(data, targets, seq_len, initial_ratio=0.6, step=30):
    """Generator for Walk-Forward validation (expanding window)."""
    n = len(data)
    train_end = int(n * initial_ratio)
    fold = 0
    
    while train_end + step <= n:
        X_tr, y_tr = create_sequences_optimized(data[:train_end], targets[:train_end], seq_len)
        test_end = min(train_end + step, n)
        X_te, y_te = create_sequences_optimized(data[train_end-seq_len:test_end], targets[train_end-seq_len:test_end], seq_len)
        
        yield X_tr, y_tr, X_te, y_te, {'fold': fold, 'train': len(X_tr), 'test': len(X_te)}
        train_end += step
        fold += 1


# ============================================================
# 3. DUAL-HEAD LSTM (Loss Weighting: clf=1.0, reg=0.3)
# ============================================================
# Justification: Auxiliary regression regularizes; too high = wrong objective, 0.3 is standard

def build_dual_head_lstm(seq_len, n_features, clf_weight=1.0, reg_weight=0.3):
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import AdamW
    
    inputs = Input(shape=(seq_len, n_features))
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    shared = Dense(32, activation='relu')(x)
    shared = Dropout(0.2)(shared)
    
    clf_out = Dense(3, activation='softmax', name='clf')(Dense(16, activation='relu')(shared))
    reg_out = Dense(1, activation='linear', name='reg')(Dense(16, activation='relu')(shared))
    
    model = Model(inputs, [clf_out, reg_out])
    model.compile(optimizer=AdamW(1e-4, weight_decay=1e-5),
                  loss={'clf': 'sparse_categorical_crossentropy', 'reg': 'mse'},
                  loss_weights={'clf': clf_weight, 'reg': reg_weight},
                  metrics={'clf': ['accuracy'], 'reg': ['mae']})
    return model


# ============================================================
# 4. REGIME DIAGNOSTICS
# ============================================================

def regime_diagnostics(y_true, y_pred, regimes, metadata=None):
    from sklearn.metrics import accuracy_score
    
    results = []
    for r in np.unique(regimes):
        mask = regimes == r
        if mask.sum() < 20:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        collapse = len(np.unique(y_pred[mask])) == 1
        results.append({'regime': r, 'n': mask.sum(), 'acc': round(acc, 4), 'collapse': collapse})
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df
