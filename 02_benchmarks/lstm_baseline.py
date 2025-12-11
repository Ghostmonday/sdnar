"""
S-DNA Order 5: LSTM Baseline Model
Skeleton Implementation (to be enhanced with research insights)

Goal: Beat Random Walk baseline (RMSE: SPY=4.54, BTC=1322)
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION (Placeholder - will refine with research)
# ============================================================

CONFIG = {
    "sequence_length": 60,      # Lookback window
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "lstm_units": [64, 32],     # Two LSTM layers
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "learning_rate": 0.001,
    "version": "1.0.0"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "01_data", "labeled")
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("S-DNA LSTM BASELINE MODEL")
print("=" * 60)
print(f"Config: {json.dumps(CONFIG, indent=2)}")


# ============================================================
# DATA LOADING
# ============================================================

def load_labeled_data(filepath, asset_name):
    """Load labeled data from Triple-Barrier pipeline."""
    print(f"\n  Loading {asset_name}...")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Expected columns: Close, volatility, label, barrier_hit, return_realized
    required_cols = ['Close', 'volatility', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Drop rows with NaN labels
    df = df.dropna(subset=['label'])
    
    print(f"    Loaded {len(df)} rows")
    print(f"    Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


# ============================================================
# FEATURE ENGINEERING (Enhanced per multi-agent research)
# ============================================================

def create_features(df):
    """
    Create features for LSTM input.
    
    Per Research Synthesis:
    - Use LOG RETURNS (not prices) for stationarity
    - Include volatility
    - Z-normalize all features
    """
    features = pd.DataFrame(index=df.index)
    
    # Log returns (Gemini insight: stationarity critical)
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (already provided)
    features['volatility'] = df['volatility']
    
    # Additional features (GPT + Sonnet insights)
    # RSI-like momentum indicator
    features['returns'] = df['Close'].pct_change()
    features['momentum_5'] = df['Close'].pct_change(5)  # 5-day momentum
    
    # Volatility ratio (current vs rolling avg)
    features['vol_ratio'] = df['volatility'] / df['volatility'].rolling(20).mean()
    
    # Drop first rows with NaN
    features = features.dropna()
    
    return features


def create_sequences(features, labels, sequence_length):
    """
    Convert features and labels into sequences for LSTM.
    
    Input shape: (samples, sequence_length, n_features)
    Output shape: (samples,) for classification or (samples, 1) for regression
    """
    X, y = [], []
    
    feature_values = features.values
    label_values = labels.values
    
    for i in range(sequence_length, len(feature_values)):
        # X: previous sequence_length timesteps
        X.append(feature_values[i - sequence_length:i])
        
        # y: label at current timestep (what we're predicting)
        y.append(label_values[i])
    
    return np.array(X), np.array(y)


# ============================================================
# DATA SPLITTING (Chronological)
# ============================================================

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Chronological split (no shuffle - prevents look-ahead bias).
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================
# MODEL BUILDING
# ============================================================

def build_lstm_model(input_shape, n_classes=3, config=CONFIG):
    """
    Build LSTM model for classification.
    
    Architecture:
    - LSTM layers with dropout
    - Dense output with softmax
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=config['lstm_units'][0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(config['dropout']))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(
            units=config['lstm_units'][1],
            return_sequences=False
        ))
        model.add(Dropout(config['dropout']))
        model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except ImportError:
        print("TensorFlow not installed. Run: pip install tensorflow")
        return None


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_data, val_data, config=CONFIG):
    """
    Train LSTM with early stopping.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_data, asset_name):
    """
    Evaluate LSTM on test set.
    
    Metrics:
    - Accuracy
    - Per-class precision/recall/F1
    - Directional accuracy
    """
    from sklearn.metrics import classification_report, accuracy_score
    
    X_test, y_test = test_data
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Remap labels: original {-1, 0, 1} → model {0, 1, 2}
    # This mapping happens in preprocessing
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n  {asset_name} Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    return {
        "accuracy": accuracy,
        "classification_report": report
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_lstm_pipeline(filepath, asset_name):
    """Run full LSTM pipeline on a single asset."""
    
    # 1. Load data
    df = load_labeled_data(filepath, asset_name)
    
    # 2. Create features
    features = create_features(df)
    
    # Align labels with features (after dropping first row)
    labels = df['label'].iloc[1:].copy()
    
    # Remap labels: {-1, 0, 1} → {0, 1, 2} for sparse categorical
    label_map = {-1: 0, 0: 1, 1: 2}
    labels = labels.map(label_map)
    
    # Drop any remaining NaN
    valid_idx = ~(features.isna().any(axis=1) | labels.isna())
    features = features[valid_idx]
    labels = labels[valid_idx]
    
    # 3. Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
    
    # 4. Create sequences
    print(f"\n  Creating sequences (length={CONFIG['sequence_length']})...")
    X, y = create_sequences(features_scaled, labels, CONFIG['sequence_length'])
    print(f"    Shape: X={X.shape}, y={y.shape}")
    
    # 5. Split
    train_data, val_data, test_data = train_val_test_split(X, y)
    
    # 6. Build model
    print("\n  Building LSTM model...")
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, n_features)
    model = build_lstm_model(input_shape, n_classes=3)
    
    if model is None:
        return None
    
    model.summary()
    
    # 7. Train
    print("\n  Training...")
    history = train_model(model, train_data, val_data)
    
    # 8. Evaluate
    results = evaluate_model(model, test_data, asset_name)
    
    return model, results, history


if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("NOTE: This is a skeleton implementation.")
    print("Will be refined with research insights from parallel agents.")
    print("=" * 60)
    
    # Check for TensorFlow
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")
        print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("\nTensorFlow not installed.")
        print("Run: pip install tensorflow")
        exit(1)
    
    # Assets
    assets = {
        "SPY": os.path.join(DATA_DIR, "spy_labeled.csv"),
        "BTC": os.path.join(DATA_DIR, "btc_labeled.csv"),
    }
    
    all_results = {}
    
    for asset_name, filepath in assets.items():
        if os.path.exists(filepath):
            print(f"\n{'='*60}")
            print(f"ASSET: {asset_name}")
            print(f"{'='*60}")
            
            try:
                model, results, history = run_lstm_pipeline(filepath, asset_name)
                all_results[asset_name] = results
                
                # Save model
                model_path = os.path.join(OUTPUT_DIR, f"{asset_name.lower()}_lstm_model.keras")
                model.save(model_path)
                print(f"\n  Model saved to {model_path}")
                
            except Exception as e:
                print(f"  [ERROR] {asset_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("LSTM RESULTS SUMMARY")
    print("=" * 60)
    
    for asset, results in all_results.items():
        print(f"\n{asset}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
    
    print("\n" + "=" * 60)
    print("SKELETON COMPLETE - Awaiting research refinements")
    print("=" * 60)
