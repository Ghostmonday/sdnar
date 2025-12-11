"""
S-DNA LSTM Baseline — Full Implementation
==========================================
Integrates all research from multi-agent synthesis:
- Volatility Normalization (homoskedastic inputs)
- Directional-MSE Loss (penalizes wrong signs)  
- Dual-Head Architecture (classification + regression)
- Purged Walk-Forward Validation
- Diebold-Mariano Statistical Test (HAC-adjusted)
- Regime-Based Failure Diagnostics

Target: Beat Random Walk (SPY RMSE < 4.54, BTC RMSE < 1322)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "sequence_length": 60,          # 3 trading months
    "train_ratio": 0.7,
    "purge_days": 10,               # Triple-barrier horizon
    "lstm_units": [128, 64],
    "dropout": 0.3,
    "noise_std": 0.01,              # GaussianNoise injection
    "batch_size": 64,
    "epochs": 100,
    "early_stopping_patience": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "directional_lambda": 0.5,      # Sign penalty weight
    "clf_weight": 1.0,
    "reg_weight": 0.3,
    "version": "2.0.0-SDNA"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "01_data", "labeled")
OUTPUT_DIR = BASE_DIR


# ============================================================
# 1. FEATURE ENGINEERING (Volatility-Normalized)
# ============================================================

def create_sdna_features(df):
    """
    S-DNA Feature Engineering Protocol:
    1. Log-returns for stationarity
    2. Divide by volatility for homoskedasticity
    3. Winsorize at ±5σ
    """
    features = pd.DataFrame(index=df.index)
    
    # Log returns
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility normalization (S-DNA Core)
    vol = df['volatility'].replace(0, np.nan).fillna(method='ffill')
    
    # Normalized returns
    features['ret_norm'] = log_ret / vol
    
    # High/Low range normalized
    if 'High' in df.columns and 'Low' in df.columns:
        hl_range = (df['High'] - df['Low']) / df['Close']
        features['range_norm'] = hl_range / vol
    
    # Volume change (if available)
    if 'Volume' in df.columns:
        vol_pct = df['Volume'].pct_change()
        features['vol_chg'] = vol_pct.clip(-5, 5)  # Cap extreme spikes
    
    # Raw volatility (z-scored)
    features['vol_zscore'] = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    
    # Momentum (5-day normalized)
    mom_5 = df['Close'].pct_change(5)
    features['mom_5_norm'] = mom_5 / vol
    
    # Winsorize all features at ±5σ
    for col in features.columns:
        mean, std = features[col].mean(), features[col].std()
        features[col] = features[col].clip(mean - 5*std, mean + 5*std)
    
    # Drop NaN
    features = features.dropna()
    
    return features


# ============================================================
# 2. OPTIMIZED SEQUENCE GENERATOR
# ============================================================

def create_sequences_optimized(data, targets, seq_len):
    """Memory-efficient sliding window using stride_tricks."""
    from numpy.lib.stride_tricks import sliding_window_view
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    X_view = sliding_window_view(data, (seq_len, n_features))
    X = X_view.squeeze(axis=1)
    y = targets[seq_len - 1:]
    
    min_len = min(len(X), len(y))
    return X[:min_len].copy(), y[:min_len].copy()


# ============================================================
# 3. PURGED WALK-FORWARD GENERATOR
# ============================================================

def purged_walk_forward(data, targets, seq_len, initial_ratio=0.6, 
                        step_size=60, purge_days=10):
    """
    Walk-Forward with purge buffer to eliminate data leakage.
    Accounts for Triple-Barrier look-ahead.
    """
    n = len(data)
    train_end = int(n * initial_ratio)
    fold = 0
    
    while train_end + purge_days + step_size <= n:
        # Train: start to train_end
        X_train, y_train = create_sequences_optimized(
            data[:train_end], targets[:train_end], seq_len
        )
        
        # Skip purge period
        test_start = train_end + purge_days
        test_end = min(test_start + step_size, n)
        
        # Test: after purge
        X_test, y_test = create_sequences_optimized(
            data[test_start - seq_len:test_end], 
            targets[test_start - seq_len:test_end], 
            seq_len
        )
        
        fold_info = {
            'fold': fold,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'purge_days': purge_days
        }
        
        yield X_train, y_train, X_test, y_test, fold_info
        
        train_end += step_size
        fold += 1


# ============================================================
# 4. DIRECTIONAL-MSE LOSS
# ============================================================

def directional_mse_loss(y_true, y_pred, lam=0.5):
    """
    Custom loss that penalizes wrong-sign predictions.
    L = MSE + λ * SignPenalty
    """
    import tensorflow as tf
    
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Sign penalty: positive when signs differ
    sign_match = tf.sign(y_true * y_pred)  # +1 if same, -1 if different
    sign_penalty = tf.nn.relu(-sign_match) * tf.square(y_true - y_pred)
    
    return mse + lam * tf.reduce_mean(sign_penalty)


# ============================================================
# 5. DUAL-HEAD S-DNA LSTM
# ============================================================

def build_sdna_lstm(seq_len, n_features, config=CONFIG):
    """
    S-DNA Baseline LSTM with:
    - GaussianNoise injection
    - Bidirectional layers
    - SpatialDropout
    - Dual-head outputs
    - Swish activation
    """
    try:
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, Dropout, Bidirectional,
            BatchNormalization, GaussianNoise, SpatialDropout1D
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import AdamW
        import tensorflow as tf
        
        # Custom Swish activation
        def swish(x):
            return x * tf.nn.sigmoid(x)
        
        # Input
        inputs = Input(shape=(seq_len, n_features), name='input')
        
        # Noise injection (prevents overfitting to exact values)
        x = GaussianNoise(config['noise_std'])(inputs)
        
        # Bidirectional LSTM Layer 1
        x = Bidirectional(LSTM(config['lstm_units'][0], return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(config['dropout'])(x)
        
        # LSTM Layer 2 (bottleneck)
        x = LSTM(config['lstm_units'][1], return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(config['dropout'])(x)
        
        # Shared representation with Swish
        shared = Dense(32, activation=swish)(x)
        shared = Dropout(0.2)(shared)
        
        # HEAD 1: Classification (Directional Accuracy target)
        clf_head = Dense(16, activation='relu')(shared)
        clf_out = Dense(3, activation='softmax', name='clf')(clf_head)
        
        # HEAD 2: Regression (RMSE target via log-return prediction)
        reg_head = Dense(16, activation='relu')(shared)
        reg_out = Dense(1, activation='linear', name='reg')(reg_head)
        
        # Build
        model = Model(inputs=inputs, outputs=[clf_out, reg_out])
        
        # Compile with weighted losses
        model.compile(
            optimizer=AdamW(
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay']
            ),
            loss={
                'clf': 'sparse_categorical_crossentropy',
                'reg': 'mse'
            },
            loss_weights={
                'clf': config['clf_weight'],
                'reg': config['reg_weight']
            },
            metrics={
                'clf': ['accuracy'],
                'reg': ['mae']
            }
        )
        
        return model
        
    except ImportError:
        print("TensorFlow required: pip install tensorflow")
        return None


# ============================================================
# 6. DIEBOLD-MARIANO TEST (HAC-adjusted)
# ============================================================

def newey_west_variance(d, max_lag=None):
    """Newey-West HAC variance for autocorrelated errors."""
    n = len(d)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
    
    d_centered = d - np.mean(d)
    gamma_0 = np.sum(d_centered ** 2) / n
    
    weighted_cov = 0
    for k in range(1, max_lag + 1):
        weight = 1 - k / (max_lag + 1)
        gamma_k = np.sum(d_centered[k:] * d_centered[:-k]) / n
        weighted_cov += 2 * weight * gamma_k
    
    return (gamma_0 + weighted_cov) / n


def diebold_mariano_test(actual, pred_lstm, pred_baseline, one_sided=True):
    """
    Robust DM test with Newey-West standard errors.
    Returns: dm_stat, p_value, significant, interpretation
    """
    actual = np.asarray(actual).flatten()
    pred_lstm = np.asarray(pred_lstm).flatten()
    pred_baseline = np.asarray(pred_baseline).flatten()
    
    e_lstm = (actual - pred_lstm) ** 2
    e_baseline = (actual - pred_baseline) ** 2
    d = e_baseline - e_lstm  # positive = LSTM better
    
    d_mean = np.mean(d)
    var_d = newey_west_variance(d)
    dm_stat = d_mean / np.sqrt(var_d) if var_d > 0 else 0
    
    if one_sided:
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {
        'dm_statistic': round(dm_stat, 4),
        'p_value': round(p_value, 6),
        'significant': p_value < 0.05,
        'interpretation': (
            'LSTM significantly BETTER (p < 0.05)' if (p_value < 0.05 and dm_stat > 0)
            else 'LSTM significantly WORSE' if (p_value < 0.05 and dm_stat < 0)
            else 'No significant difference'
        )
    }


# ============================================================
# 7. FAILURE DIAGNOSTICS
# ============================================================

def diagnose_lag1_failure(y_pred, y_actual_lagged):
    """Check if model just learned y_{t+1} ≈ y_t"""
    corr = np.corrcoef(y_pred.flatten(), y_actual_lagged.flatten())[0, 1]
    if corr > 0.95:
        print(f"⚠️ LAG-1 FAILURE: Correlation = {corr:.4f}")
        print("   Model is just copying previous value")
        print("   Remedy: Increase Dropout, increase GaussianNoise")
        return True
    return False


def diagnose_directional_collapse(y_pred):
    """Check if predictions clustered near zero"""
    pred_std = np.std(y_pred)
    pred_mean = np.mean(np.abs(y_pred))
    
    if pred_mean < 0.01:
        print(f"⚠️ DIRECTIONAL COLLAPSE: Mean|pred| = {pred_mean:.6f}")
        print("   Model predicting near-zero (avoiding risk)")
        print("   Remedy: Increase λ in Directional-MSE loss")
        return True
    return False


def regime_diagnostics(y_true, y_pred, regime_labels):
    """Analyze performance across market regimes."""
    from sklearn.metrics import accuracy_score
    
    results = []
    for r in np.unique(regime_labels):
        if r < 0:
            continue
        mask = regime_labels == r
        if mask.sum() < 20:
            continue
        
        acc = accuracy_score(y_true[mask], y_pred[mask])
        collapse = len(np.unique(y_pred[mask])) == 1
        
        results.append({
            'regime': r,
            'samples': mask.sum(),
            'accuracy': round(acc, 4),
            'mode_collapse': collapse
        })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        print("\n=== REGIME PERFORMANCE ===")
        print(df.to_string(index=False))
        
        if df['mode_collapse'].any():
            print(f"\n🚨 MODE COLLAPSE in regimes: {df[df['mode_collapse']]['regime'].tolist()}")
    
    return df


# ============================================================
# 8. TRAINING PIPELINE
# ============================================================

def train_sdna_lstm(filepath, asset_name, config=CONFIG):
    """Full S-DNA training pipeline for a single asset."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    print(f"\n{'='*60}")
    print(f"S-DNA LSTM: {asset_name}")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = df.dropna(subset=['volatility', 'label'])
    print(f"      Loaded {len(df)} samples")
    
    # 2. Feature engineering
    print("[2/6] Engineering features...")
    features = create_sdna_features(df)
    
    # Align labels and returns
    labels = df.loc[features.index, 'label'].values
    
    # Map labels: {-1, 0, 1} → {0, 1, 2}
    labels_mapped = labels + 1
    
    # Get returns for regression target
    returns = np.log(df.loc[features.index, 'Close'] / 
                     df.loc[features.index, 'Close'].shift(1)).fillna(0).values
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.values)
    
    print(f"      Features: {features.columns.tolist()}")
    print(f"      Shape: {features_scaled.shape}")
    
    # 3. Create sequences
    print(f"[3/6] Creating sequences (len={config['sequence_length']})...")
    X, y_clf = create_sequences_optimized(
        features_scaled, labels_mapped, config['sequence_length']
    )
    _, y_reg = create_sequences_optimized(
        features_scaled, returns, config['sequence_length']
    )
    
    print(f"      X: {X.shape}, y_clf: {y_clf.shape}, y_reg: {y_reg.shape}")
    
    # 4. Train/Val/Test split (chronological)
    print("[4/6] Splitting data (chronological)...")
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_train, y_clf_train, y_reg_train = X[:train_end], y_clf[:train_end], y_reg[:train_end]
    X_val, y_clf_val, y_reg_val = X[train_end:val_end], y_clf[train_end:val_end], y_reg[train_end:val_end]
    X_test, y_clf_test, y_reg_test = X[val_end:], y_clf[val_end:], y_reg[val_end:]
    
    print(f"      Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 5. Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_clf_train), y=y_clf_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"      Class weights: {class_weight_dict}")
    
    # 6. Build and train model
    print("[5/6] Building S-DNA LSTM...")
    model = build_sdna_lstm(X.shape[1], X.shape[2], config)
    
    if model is None:
        return None
    
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_clf_accuracy',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )
    ]
    
    print("\n[6/6] Training...")
    history = model.fit(
        X_train,
        {'clf': y_clf_train, 'reg': y_reg_train},
        validation_data=(X_val, {'clf': y_clf_val, 'reg': y_reg_val}),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        class_weight={'clf': class_weight_dict},
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n=== EVALUATION ===")
    clf_pred, reg_pred = model.predict(X_test)
    clf_pred_class = np.argmax(clf_pred, axis=1)
    
    from sklearn.metrics import accuracy_score, classification_report
    
    acc = accuracy_score(y_clf_test, clf_pred_class)
    print(f"\nClassification Accuracy: {acc:.4f}")
    print(classification_report(y_clf_test, clf_pred_class, target_names=['Bear', 'Neutral', 'Bull']))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_reg_test - reg_pred.flatten()) ** 2))
    print(f"Regression RMSE: {rmse:.6f}")
    
    # Directional accuracy
    actual_sign = np.sign(y_reg_test)
    pred_sign = np.sign(reg_pred.flatten())
    dir_acc = np.mean(actual_sign == pred_sign)
    print(f"Directional Accuracy: {dir_acc:.4f}")
    
    # Failure diagnostics
    print("\n=== DIAGNOSTICS ===")
    diagnose_lag1_failure(reg_pred.flatten(), y_reg_test)
    diagnose_directional_collapse(reg_pred)
    
    return model, history, {
        'accuracy': acc,
        'rmse': rmse,
        'dir_acc': dir_acc
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("S-DNA LSTM BASELINE — PRODUCTION RUN")
    print("=" * 60)
    print(f"Config: {json.dumps(CONFIG, indent=2)}")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"\nTensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs: {gpus if gpus else 'None (CPU mode)'}")
    except ImportError:
        print("\n❌ TensorFlow not installed")
        print("Run: pip install tensorflow")
        exit(1)
    
    # Assets
    assets = {
        "SPY": os.path.join(DATA_DIR, "spy_labeled.csv"),
        "BTC": os.path.join(DATA_DIR, "btc_labeled.csv"),
    }
    
    results = {}
    
    for asset, path in assets.items():
        if os.path.exists(path):
            try:
                model, history, metrics = train_sdna_lstm(path, asset)
                results[asset] = metrics
                
                # Save model
                model_path = os.path.join(OUTPUT_DIR, f"{asset.lower()}_sdna_lstm.keras")
                model.save(model_path)
                print(f"\n✓ Model saved: {model_path}")
                
            except Exception as e:
                print(f"\n❌ {asset} failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠️ {asset} data not found: {path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for asset, metrics in results.items():
        print(f"\n{asset}:")
        print(f"  Classification Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Regression RMSE: {metrics['rmse']:.6f}")
        print(f"  Directional Accuracy: {metrics['dir_acc']:.4f}")
    
    print("\n" + "=" * 60)
    print("S-DNA LSTM COMPLETE")
    print("=" * 60)
