# LSTM Baseline Research for S-DNA Validation Framework

**Research Date:** December 10, 2025  
**Objective:** Design and implement LSTM models to outperform current baseline performance (SPY Random Walk RMSE=4.54, BTC Random Walk RMSE=1322)

---

## 1. ARCHITECTURE SPECIFICATION

### 1.1 Input Features

**Recommended Feature Set:**

- **Close Price (scaled):** Primary predictive feature
- **Returns:** `(Close_t - Close_{t-1}) / Close_{t-1}`
- **Volatility:** Already computed in datasets (rolling standard deviation)
- **Lagged Labels:** Previous 3-5 labels to capture label persistence/momentum
- **Technical Indicators:**
  - **RSI (Relative Strength Index):** Overbought/oversold conditions
  - **Price Distance from SMA(20):** `(Close - SMA_20) / SMA_20`
  - **Volume (if available):** Market activity indicator

**Rationale:** The Triple-Barrier Method labels are derived from future price movements relative to volatility. Including volatility, returns, and momentum features will help the LSTM learn the underlying patterns that drive these labels.

### 1.2 Sequence Length

**Recommended:** **60 timesteps** (approximately 3 months of trading days)

**Alternatives to Test:**

- **20 timesteps:** Shorter-term patterns, faster training
- **120 timesteps:** Longer-term dependencies, higher computational cost

**Rationale:**

- Triple-Barrier Method uses 10-day vertical barrier
- 60 days provides ~6 barrier cycles for pattern recognition
- Balance between capturing sufficient context and avoiding vanishing gradients

### 1.3 Network Structure

**Recommended Architecture:**

```python
Layer 1: Bidirectional LSTM (128 units, return_sequences=True)
Layer 2: Dropout (0.3)
Layer 3: Bidirectional LSTM (64 units, return_sequences=False)
Layer 4: Dropout (0.3)
Layer 5: Dense (32 units, activation='relu')
Layer 6: Dropout (0.2)
Layer 7: Output Layer (see below)
```

**Architecture Rationale:**

- **Bidirectional LSTMs:** Capture both past and future context within sequences
- **Stacked design:** First layer extracts low-level temporal patterns, second layer learns higher-level abstractions
- **Decreasing units:** 128→64→32 prevents overfitting while maintaining expressiveness
- **Dropout layers:** Combat overfitting on limited financial data

### 1.4 Output Layer Design

**Decision: Dual-Head Architecture**

Given the data contains both discrete labels (1/-1/0) and continuous returns, implement **two output heads:**

**Head 1 - Classification (Primary):**

```python
Dense(3, activation='softmax')  # Predict label probabilities for {-1, 0, 1}
```

**Head 2 - Regression (Auxiliary):**

```python
Dense(1, activation='linear')  # Predict actual return
```

**Alternative (Simpler):**
If focusing solely on directional prediction:

```python
Dense(3, activation='softmax')  # 3-class classification
```

**Rationale:**

- Classification head directly optimizes for directional accuracy
- Regression head provides additional supervision signal
- Multi-task learning often improves generalization

### 1.5 Regularization Strategy

**Comprehensive Regularization:**

1. **Dropout:** 0.3 after LSTM layers, 0.2 after dense layers
2. **L2 Regularization:** `kernel_regularizer=l2(0.001)` on Dense layers
3. **Recurrent Dropout:** `recurrent_dropout=0.2` within LSTM layers
4. **Batch Normalization:** After each LSTM layer (before dropout)
5. **Early Stopping:** Monitor validation loss with patience=15 epochs
6. **Learning Rate Scheduling:** ReduceLROnPlateau (factor=0.5, patience=7)

**Rationale:** Financial data is notoriously noisy and prone to overfitting. Aggressive regularization is essential.

---

## 2. TRAINING PROTOCOL

### 2.1 Train/Validation/Test Split

**Walk-Forward Strategy (Recommended):**

```
SPY Data (6,274 rows):
├── Train: First 70% (4,392 samples) → 2000-01-03 to 2017-06-15
├── Validation: Next 15% (941 samples) → 2017-06-16 to 2020-03-24
└── Test: Final 15% (941 samples) → 2020-03-25 to 2024-12-31

BTC Data (3,737 rows):
├── Train: First 70% (2,616 samples) → 2014-09-17 to 2021-01-20
├── Validation: Next 15% (560 samples) → 2021-01-21 to 2022-04-15
└── Test: Final 15% (561 samples) → 2022-04-16 to 2024-12-31
```

**Critical Considerations:**

- **NO SHUFFLING:** Preserve temporal order to prevent data leakage
- **Validation for hyperparameter tuning:** Learning rate, dropout, sequence length
- **Test set remains untouched** until final evaluation

### 2.2 Handling Class Imbalance

**Label Distribution Analysis Required:**

```python
# Check balance across {-1, 0, 1}
label_counts = df['label'].value_counts()
```

**If Imbalanced (likely):**

1. **Class Weights:** Inverse frequency weighting

   ```python
   from sklearn.utils.class_weight import compute_class_weight
   class_weights = compute_class_weight('balanced',
                                         classes=np.unique(labels),
                                         y=labels)
   ```

2. **SMOTE for Time Series (SMOGN):** Synthetic oversampling while respecting temporal order
3. **Focal Loss (Alternative):** Dynamic weighting based on prediction difficulty

### 2.3 Optimizer Configuration

**Primary Choice: Adam Optimizer**

```python
optimizer = Adam(learning_rate=0.001,  # Start conservatively
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-07,
                 clipnorm=1.0)  # Gradient clipping for stability
```

**Learning Rate Schedule:**

```python
ReduceLROnPlateau(monitor='val_loss',
                  factor=0.5,
                  patience=7,
                  min_lr=1e-7,
                  verbose=1)
```

**Alternative Optimizers to Test:**

- **AdamW:** Better generalization via decoupled weight decay
- **RMSprop:** Effective for RNNs, adaptive learning rates

### 2.4 Loss Function

**For Classification Head:**

```python
# Weighted Categorical Crossentropy
loss_classification = 'categorical_crossentropy'
# Apply class_weights during model.fit()
```

**For Multi-Task (Classification + Regression):**

```python
losses = {
    'classification_output': 'categorical_crossentropy',
    'regression_output': 'mse'
}
loss_weights = {
    'classification_output': 1.0,  # Primary task
    'regression_output': 0.3       # Auxiliary task
}
```

**Alternative (For Imbalanced Data):**

```python
# Focal Loss implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = alpha * tf.pow(1 - p_t, gamma)
        return focal_weight * ce
    return loss_fn
```

### 2.5 Training Hyperparameters

**Batch Size:**

- **Start with 64** (balance between gradient stability and memory)
- **Test:** 32, 128 (smaller captures finer patterns, larger stabilizes gradients)

**Epochs:**

- **100 epochs** with early stopping (patience=15)
- Likely to converge earlier (~40-60 epochs)

**Early Stopping:**

```python
EarlyStopping(monitor='val_loss',
              patience=15,
              restore_best_weights=True,
              verbose=1)
```

**Model Checkpointing:**

```python
ModelCheckpoint('best_model_spy.h5',
                monitor='val_dir_acc',  # Custom metric
                save_best_only=True,
                mode='max')
```

---

## 3. WALK-FORWARD VALIDATION

### 3.1 Implementation Strategy

**Expanding Window Approach (Recommended for Limited Data):**

```python
# Pseudo-code for walk-forward validation
initial_train_size = int(0.7 * len(data))
step_size = 30  # Re-train every 30 days

predictions = []
for i in range(initial_train_size, len(data) - step_size, step_size):
    # Expanding window: always use data from start
    train_data = data[0:i]
    test_data = data[i:i+step_size]

    # Train model
    model = build_lstm()
    model.fit(train_data, ...)

    # Predict next 30 days
    preds = model.predict(test_data)
    predictions.extend(preds)
```

**Alternative: Rolling Window (Fixed Training Size):**

```python
# Use last N samples for training (e.g., N = 2000)
train_data = data[i-2000:i]
```

**Rationale:**

- Expanding window leverages all historical data (better for limited datasets)
- Rolling window adapts faster to regime changes
- **Recommendation:** Start with expanding window

### 3.2 Retrain Frequency

**Options:**

1. **Monthly (20-30 trading days):** Balance between adaptation and computational cost
2. **Quarterly (60 trading days):** Fewer retrains, captures longer-term patterns
3. **Event-Driven:** Retrain when volatility regime changes (detected via VIX threshold)

**Recommended:** **Monthly retraining** with performance monitoring

### 3.3 Aggregating Out-of-Sample Predictions

**Process:**

1. Store predictions from each walk-forward fold
2. Concatenate chronologically
3. Align with true labels from test set
4. Compute metrics on aggregated predictions

**Avoid:** Averaging predictions across multiple models (loses directional clarity)

---

## 4. EVALUATION FRAMEWORK

### 4.1 Classification Metrics (Primary)

**Per-Class Metrics:**

```python
from sklearn.metrics import classification_report, confusion_matrix

# Per-class precision, recall, F1
print(classification_report(y_true, y_pred,
                           target_names=['Down (-1)', 'Neutral (0)', 'Up (1)']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

**Key Metrics:**

- **Precision (Class 1 & -1):** Of predicted up/down, how many correct?
- **Recall (Class 1 & -1):** Of actual up/down, how many detected?
- **F1-Score:** Harmonic mean (important when classes imbalanced)
- **Macro F1:** Average F1 across classes (weights all classes equally)

**Multi-Class AUC:**

```python
from sklearn.metrics import roc_auc_score

# One-vs-Rest AUC
auc_ovr = roc_auc_score(y_true_one_hot, y_pred_prob,
                        multi_class='ovr', average='macro')
```

### 4.2 Regression Metrics (If Using Regression Head)

**Primary Metrics:**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_true_returns, y_pred_returns))
mae = mean_absolute_error(y_true_returns, y_pred_returns)
```

**Directional Accuracy (Critical for Trading):**

```python
def directional_accuracy(y_true, y_pred):
    """Percentage of correctly predicted directions"""
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
```

**Comparison to Baseline:**

- **Target:** RMSE < 4.54 (SPY) and < 1322 (BTC)
- **Directional Accuracy:** > 50% (significantly better than random)

### 4.3 Statistical Significance Testing

**Diebold-Mariano Test:**

```python
from scipy import stats

def diebold_mariano_test(actual, pred1, pred2, h=1):
    """
    Test if forecast errors from pred1 are significantly
    different from pred2

    H0: pred1 and pred2 have equal forecast accuracy
    H1: pred1 is more accurate than pred2
    """
    e1 = actual - pred1  # LSTM errors
    e2 = actual - pred2  # ARIMA errors

    d = e1**2 - e2**2  # Loss differential

    # Compute DM statistic
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    n = len(d)

    DM_stat = mean_d / np.sqrt((1/n) * var_d)

    # One-sided test: Is LSTM significantly better?
    p_value = 1 - stats.norm.cdf(DM_stat)

    return DM_stat, p_value

# Usage
dm_stat, p_val = diebold_mariano_test(actual=y_test,
                                       pred1=lstm_preds,
                                       pred2=arima_preds)

if p_val < 0.05:
    print("LSTM significantly outperforms ARIMA (p < 0.05)")
```

**Interpretation:**

- **p < 0.05:** LSTM is statistically significantly better than ARIMA
- **p >= 0.05:** No significant difference (cannot reject H0)

### 4.4 Comprehensive Evaluation Report

**Metrics to Report:**

| Metric                | SPY (LSTM) | SPY (Baseline) | BTC (LSTM) | BTC (Baseline) |
| --------------------- | ---------- | -------------- | ---------- | -------------- |
| RMSE                  | ?          | 4.54 (RW)      | ?          | 1322 (RW)      |
| MAE                   | ?          | 3.29 (RW)      | ?          | 824.56 (RW)    |
| Dir. Acc (%)          | ?          | 49.44 (RW)     | ?          | 46.05 (RW)     |
| F1-Macro              | ?          | -              | ?          | -              |
| AUC-OVR               | ?          | -              | ?          | -              |
| DM p-value (vs ARIMA) | ?          | -              | ?          | -              |

**Success Criteria:**

1. ✅ RMSE < Baseline
2. ✅ Directional Accuracy > 52% (meaningfully above random)
3. ✅ DM Test p-value < 0.05 vs ARIMA

---

## 5. IMPLEMENTATION GUIDANCE

### 5.1 Framework Recommendation

**Primary: TensorFlow/Keras**

**Rationale:**

- **Keras Sequential API:** Rapid prototyping
- **Keras Functional API:** Easy multi-output models
- **TensorBoard:** Built-in training visualization
- **Mature LSTM implementations:** Optimized CuDNN kernels

**Alternative: PyTorch**

- More flexible for custom architectures
- Better for research experimentation
- Steeper learning curve

**Recommendation:** **Start with Keras** for speed, migrate to PyTorch if custom layers needed

### 5.2 Time Series Data Generator

**Critical Pattern: Sliding Window Generator**

```python
import numpy as np

def create_sequences(data, labels, sequence_length=60):
    """
    Create LSTM input sequences

    Args:
        data: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
        sequence_length: Lookback window

    Returns:
        X: (n_sequences, sequence_length, n_features)
        y: (n_sequences, n_outputs)
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        # Input: sequence of past 60 timesteps
        X.append(data[i:i+sequence_length])

        # Target: label at timestep i+sequence_length
        y.append(labels[i+sequence_length])

    return np.array(X), np.array(y)

# Usage
from sklearn.preprocessing import StandardScaler

# Scale features (CRITICAL for LSTMs)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, y_train = create_sequences(features_scaled, labels, 60)
```

**Key Considerations:**

1. **Feature Scaling:** StandardScaler or MinMaxScaler (LSTMs sensitive to scale)
2. **Label Encoding:** One-hot encode labels for classification
3. **Fit scaler on training data only:** Prevent data leakage

### 5.3 GPU Requirements

**Estimated Training Time (per asset):**

- **CPU (8-core):** ~4-6 hours per experiment
- **GPU (NVIDIA RTX 3060):** ~20-30 minutes per experiment
- **Cloud GPU (A100):** ~5-10 minutes per experiment

**Memory Requirements:**

- **Sequence Length 60, Batch 64:** ~2-4 GB GPU RAM
- **Larger batches/sequences:** 8-16 GB GPU RAM

**Recommendation:**

- **Free Option:** Google Colab (15GB GPU, time-limited)
- **Cloud Option:** AWS SageMaker, GCP AI Platform (~$1-3/hour)
- **Local:** NVIDIA GTX 1660 or better (6GB+ VRAM)

### 5.4 Complete Training Script Outline

```python
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                      Bidirectional, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# 1. Load Data
df = pd.read_csv('spy_labeled.csv')

# 2. Feature Engineering
df['returns'] = df['Close'].pct_change()
df['rsi'] = compute_rsi(df['Close'], 14)  # Implement RSI
df = df.dropna()

features = df[['Close', 'returns', 'volatility', 'rsi']].values
labels = df['label'].values

# 3. Train/Val/Test Split (temporal)
train_size = int(0.7 * len(features))
val_size = int(0.15 * len(features))

train_X, train_y = features[:train_size], labels[:train_size]
val_X, val_y = features[train_size:train_size+val_size], labels[train_size:train_size+val_size]
test_X, test_y = features[train_size+val_size:], labels[train_size+val_size:]

# 4. Scale Features
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)
test_X_scaled = scaler.transform(test_X)

# 5. Create Sequences
SEQ_LENGTH = 60
X_train, y_train = create_sequences(train_X_scaled, train_y, SEQ_LENGTH)
X_val, y_val = create_sequences(val_X_scaled, val_y, SEQ_LENGTH)
X_test, y_test = create_sequences(test_X_scaled, test_y, SEQ_LENGTH)

# 6. One-Hot Encode Labels
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train + 1, num_classes=3)  # Map {-1,0,1} to {0,1,2}
y_val_cat = to_categorical(y_val + 1, num_classes=3)
y_test_cat = to_categorical(y_test + 1, num_classes=3)

# 7. Compute Class Weights
class_weights = compute_class_weight('balanced',
                                     classes=np.array([0, 1, 2]),
                                     y=y_train + 1)
class_weight_dict = {i: class_weights[i] for i in range(3)}

# 8. Build Model
def build_lstm(input_shape, n_classes=3):
    inputs = Input(shape=input_shape)

    # LSTM Layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Dense Layers
    x = Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)

    # Output
    outputs = Dense(n_classes, activation='softmax', name='classification')(x)

    model = Model(inputs, outputs)
    return model

model = build_lstm(input_shape=(SEQ_LENGTH, X_train.shape[2]), n_classes=3)

# 9. Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 10. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7),
    ModelCheckpoint('best_model_spy.h5', monitor='val_accuracy',
                    save_best_only=True, mode='max')
]

# 11. Train
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# 12. Evaluate
test_preds = model.predict(X_test)
test_pred_classes = np.argmax(test_preds, axis=1) - 1  # Map back to {-1,0,1}

from sklearn.metrics import classification_report
print(classification_report(y_test, test_pred_classes))
```

---

## 6. FAILURE DIAGNOSTICS

### 6.1 If LSTM Doesn't Beat Random Walk

**Diagnostic Checklist:**

#### **1. Check for Overfitting**

**Symptoms:**

- High training accuracy (>80%), low validation accuracy (<55%)
- Training loss decreases, validation loss increases

**Solutions:**

- ✅ Increase dropout rates (0.4-0.5)
- ✅ Add more L2 regularization
- ✅ Reduce model complexity (fewer LSTM units)
- ✅ Increase training data (if possible)
- ✅ Use simpler architecture (single LSTM layer)

#### **2. Check for Underfitting**

**Symptoms:**

- Both training and validation accuracy low (<50%)
- Loss plateaus early

**Solutions:**

- ✅ Increase model capacity (more LSTM units, add layers)
- ✅ Decrease regularization
- ✅ Train longer (remove early stopping temporarily)
- ✅ Add more features (technical indicators)
- ✅ Increase sequence length

#### **3. Data Quality Issues**

**Potential Problems:**

- **Data leakage:** Future information in features
- **Scaling errors:** Fit scaler on test data
- **Label imbalance:** 90% neutral labels → model predicts neutral always

**Solutions:**

- ✅ Verify train/test split is chronological
- ✅ Check scaler fit only on training data
- ✅ Analyze label distribution, apply SMOTE or class weights
- ✅ Filter out neutral labels (focus on {-1, 1} only)

#### **4. Hyperparameter Tuning**

**Key Parameters to Tune:**

- Sequence length: [20, 40, 60, 90, 120]
- LSTM units: [32, 64, 128, 256]
- Dropout: [0.2, 0.3, 0.4, 0.5]
- Learning rate: [0.0001, 0.001, 0.01]
- Batch size: [32, 64, 128]

**Use:** Keras Tuner or Optuna for automated search

```python
import keras_tuner as kt

def build_model_tuner(hp):
    model = keras.Sequential()

    # Tunable LSTM units
    model.add(LSTM(units=hp.Int('units_1', 32, 256, step=32),
                   return_sequences=True,
                   input_shape=(SEQ_LENGTH, N_FEATURES)))
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('units_2', 32, 128, step=32)))
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))

    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.BayesianOptimization(
    build_model_tuner,
    objective='val_accuracy',
    max_trials=20
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val))
```

### 6.2 Overfitting Indicators

**Track These Metrics:**

1. **Train-Val Loss Gap:** If gap > 0.5, overfitting likely
2. **Train-Val Accuracy Gap:** If gap > 10%, overfitting
3. **Learning Curves:** Plot training/validation loss over epochs

**Visualization:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.show()
```

### 6.3 Regime-Specific Performance Analysis

**Hypothesis:** LSTM may perform differently in high/low volatility regimes

**Analysis:**

```python
# Split test set by volatility regime
median_vol = df['volatility'].median()

high_vol_idx = df[df['volatility'] > median_vol].index
low_vol_idx = df[df['volatility'] <= median_vol].index

# Evaluate separately
from sklearn.metrics import accuracy_score

high_vol_acc = accuracy_score(y_test[high_vol_idx], preds[high_vol_idx])
low_vol_acc = accuracy_score(y_test[low_vol_idx], preds[low_vol_idx])

print(f"High Volatility Accuracy: {high_vol_acc:.2%}")
print(f"Low Volatility Accuracy: {low_vol_acc:.2%}")
```

**Expected:**

- High volatility → Easier to predict (clearer trends)
- Low volatility → Harder (more noise)

**Action:**

- If one regime significantly worse → Train separate models per regime
- Use volatility as switching criterion

### 6.4 Alternative Architectures to Try

If standard LSTM fails:

**1. GRU (Gated Recurrent Unit):**

- Simpler than LSTM, fewer parameters
- Often performs similarly, faster training

**2. CNN-LSTM Hybrid:**

```python
# CNN extracts local patterns, LSTM captures sequences
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64),
    Dense(3, activation='softmax')
])
```

**3. Attention Mechanisms:**

- Transformer-based (requires more data)
- Self-attention LSTM (focus on important timesteps)

**4. Ensemble:**

- Train 5 LSTM models with different random seeds
- Average predictions → Reduces variance

### 6.5 When to Abandon LSTM Approach

**Red Flags:**

1. After extensive tuning, validation accuracy stuck at ~50% (random)
2. Diebold-Mariano test: p > 0.5 (LSTM worse than ARIMA)
3. Directional accuracy < 48% on test set

**Alternative Approaches:**

- **XGBoost/LightGBM:** May handle noisy financial data better
- **Ensemble of shallow models:** Random Forest + SVM + LSTM
- **Feature engineering focus:** Better features > complex model
- **Simpler baselines:** Improve SMA strategy with adaptive parameters

---

## 7. EXPECTED SUCCESS CRITERIA

### 7.1 Minimum Viable Performance

| Metric                   | SPY Target | BTC Target |
| ------------------------ | ---------- | ---------- |
| **RMSE**                 | < 4.54     | < 1322     |
| **Directional Accuracy** | > 52%      | > 50%      |
| **F1-Macro**             | > 0.40     | > 0.35     |
| **DM Test (vs ARIMA)**   | p < 0.10   | p < 0.10   |

### 7.2 Stretch Goals

| Metric                           | SPY Stretch | BTC Stretch |
| -------------------------------- | ----------- | ----------- |
| **RMSE**                         | < 4.0       | < 1200      |
| **Directional Accuracy**         | > 55%       | > 53%       |
| **Sharpe Ratio (if backtested)** | > 1.0       | > 0.8       |

### 7.3 Timeline Estimate

**Phase 1: Setup & Initial Training (Week 1)**

- Data preprocessing: 1 day
- Feature engineering: 1 day
- First LSTM model: 1 day
- Baseline evaluation: 0.5 day

**Phase 2: Optimization (Week 2)**

- Hyperparameter tuning: 2-3 days
- Architecture experiments: 2 days

**Phase 3: Validation (Week 3)**

- Walk-forward validation: 2 days
- Statistical testing: 1 day
- Final documentation: 1 day

**Total:** ~2-3 weeks for comprehensive research

---

## 8. RECOMMENDED NEXT STEPS

1. ✅ **Implement baseline LSTM** (single bi-LSTM layer, 60 timesteps)
2. ✅ **Validate data pipeline** (no leakage, proper scaling)
3. ✅ **Compute class distribution** (address imbalance if needed)
4. ✅ **Train on SPY first** (larger dataset, faster iteration)
5. ✅ **Hyperparameter sweep** (Keras Tuner with 20 trials)
6. ✅ **Walk-forward validation** (expanding window, monthly retrain)
7. ✅ **Statistical testing** (Diebold-Mariano vs ARIMA)
8. ✅ **Replicate on BTC** (transfer insights from SPY)
9. ✅ **Ensemble if time permits** (average 3-5 models)
10. ✅ **Document findings** (even if LSTM fails, insights valuable)

---

## 9. KEY REFERENCES

**Academic Papers:**

1. Fischer & Krauss (2018): "Deep Learning for Financial Prediction"
2. Sezer et al. (2020): "Financial Time Series Forecasting with Deep Learning"
3. Jiang (2021): "Applications of Deep Learning in Stock Market Prediction"

**Technical Resources:**

1. TensorFlow Time Series Tutorial: https://www.tensorflow.org/tutorials/structured_data/time_series
2. Keras LSTM Documentation: https://keras.io/api/layers/recurrent_layers/lstm/
3. Financial ML (Marcos López de Prado): Chapter on Meta-Labeling

**Implementation Examples:**

1. Keras Time Series Example: https://github.com/keras-team/keras-io/blob/master/examples/timeseries/
2. PyTorch Financial LSTM: https://github.com/pytorch/examples/tree/master/time_sequence_prediction

---

## CONCLUSION

LSTM models offer a powerful framework for capturing temporal dependencies in financial time series labeled via the Triple-Barrier Method. Success hinges on:

1. **Careful data handling** (no leakage, proper scaling)
2. **Aggressive regularization** (financial data is noisy)
3. **Class imbalance mitigation** (likely skewed toward neutral)
4. **Rigorous validation** (walk-forward, not random split)
5. **Statistical rigor** (DM test for significance)

**Expected Outcome:** With proper implementation, LSTM should achieve 52-55% directional accuracy on SPY and 50-53% on BTC, representing a meaningful improvement over random walk and providing a solid foundation for the S-DNA validation framework.

If LSTM underperforms, the extensive diagnostic framework provided will identify root causes and guide alternative approaches (XGBoost, ensemble methods, or enhanced feature engineering).
