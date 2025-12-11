"""
S-DNA Benchmark Baselines
Implements: Random Walk, SMA, ARIMA
Per Gemini's Spec - Order 4

Metrics: RMSE, MAE, Directional Accuracy
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "train_ratio": 0.8,
    "sma_window": 20,
    "arima_order": (5, 1, 0),  # Conservative ARIMA(5,1,0)
    "version": "1.0.0"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "01_data")
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("S-DNA BENCHMARK BASELINES")
print("=" * 60)
print(f"Config: {CONFIG}")


# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath, asset_name):
    """Load and clean price data."""
    print(f"\n  Loading {asset_name}...")
    
    df = pd.read_csv(filepath)
    
    # Handle yfinance format
    if df.iloc[0, 0] == 'Ticker' or str(df.iloc[0, 0]).startswith('Date'):
        df = pd.read_csv(filepath, skiprows=2)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    else:
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    df = df.sort_index()  # Chronological order
    
    print(f"    Loaded {len(df)} rows: {df.index.min().date()} to {df.index.max().date()}")
    return df


def train_test_split(df, train_ratio=0.8):
    """Chronological train/test split (NO shuffle - prevents look-ahead bias)."""
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"    Train: {len(train)} rows, Test: {len(test)} rows")
    return train, test


# ============================================================
# BASELINE MODELS
# ============================================================

def random_walk_forecast(train, test):
    """
    Random Walk: Predict P(t+1) = P(t)
    The simplest baseline - tomorrow's price equals today's price.
    """
    # Shift by 1: use previous day's close as prediction
    predictions = test['Close'].shift(1)
    
    # First value has no previous - use last training value
    predictions.iloc[0] = train['Close'].iloc[-1]
    
    return predictions


def sma_forecast(train, test, window=20):
    """
    Simple Moving Average: Predict P(t+1) = mean(P(t-window+1:t))
    Uses rolling window mean.
    """
    # Combine train and test for continuous rolling window
    combined = pd.concat([train['Close'].iloc[-window:], test['Close']])
    
    # Calculate SMA
    sma = combined.rolling(window=window).mean()
    
    # Get predictions for test period only
    # Prediction at t is SMA of t-1 (shifted by 1)
    predictions = sma.shift(1).loc[test.index]
    
    return predictions


def arima_forecast(train, test, order=(5, 1, 0)):
    """
    ARIMA: Autoregressive Integrated Moving Average
    
    Uses FILTERED ONE-STEP-AHEAD approach (per Gemini research):
    - Fit parameters on train set
    - Apply() to test set with frozen parameters
    - Kalman Filter updates state using actual observations
    - NOT static multi-step forecast (which compounds error)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        print("    Fitting ARIMA (this may take a moment)...")
        
        # Fit on training data
        model = ARIMA(train['Close'], order=order)
        fitted = model.fit()
        
        # CRITICAL: Use apply() for filtered one-step-ahead
        # This freezes parameters but updates Kalman state with real observations
        try:
            # Try apply() method (preferred)
            new_results = fitted.apply(test['Close'])
            # Get one-step-ahead predictions
            predictions = new_results.fittedvalues
            predictions = pd.Series(predictions.values, index=test.index)
        except AttributeError:
            # Fallback: extend and predict
            print("    Using fallback prediction method...")
            history = list(train['Close'].values)
            predictions_list = []
            
            for t in range(len(test)):
                # Predict next value
                model_step = ARIMA(history, order=order)
                model_step_fit = model_step.fit()
                yhat = model_step_fit.forecast(steps=1)[0]
                predictions_list.append(yhat)
                
                # Add actual observation to history (not prediction)
                history.append(test['Close'].iloc[t])
            
            predictions = pd.Series(predictions_list, index=test.index)
        
        return predictions
        
    except Exception as e:
        print(f"    ARIMA failed: {e}")
        # Fallback to random walk
        return random_walk_forecast(train, test)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(actual, predicted, model_name):
    """
    Compute RMSE, MAE, and Directional Accuracy.
    """
    # Drop NaN pairs
    mask = ~(actual.isna() | predicted.isna())
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {"model": model_name, "rmse": np.nan, "mae": np.nan, "dir_acc": np.nan}
    
    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # MAE
    mae = np.mean(np.abs(actual - predicted))
    
    # Directional Accuracy
    # Did we predict the direction of price change correctly?
    actual_direction = np.sign(actual.diff().dropna())
    predicted_direction = np.sign(predicted.diff().dropna())
    
    # Align indices
    common_idx = actual_direction.index.intersection(predicted_direction.index)
    dir_acc = (actual_direction.loc[common_idx] == predicted_direction.loc[common_idx]).mean()
    
    return {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "dir_acc": round(dir_acc * 100, 2)
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_benchmarks(filepath, asset_name):
    """Run all baselines on a single asset."""
    
    # Load data
    df = load_data(filepath, asset_name)
    
    # Split
    train, test = train_test_split(df, CONFIG['train_ratio'])
    actual = test['Close']
    
    results = []
    predictions_all = {}
    
    # 1. Random Walk
    print(f"\n  [1/3] Random Walk...")
    rw_pred = random_walk_forecast(train, test)
    rw_metrics = compute_metrics(actual, rw_pred, "Random Walk")
    results.append(rw_metrics)
    predictions_all['random_walk'] = rw_pred
    print(f"    RMSE={rw_metrics['rmse']}, MAE={rw_metrics['mae']}, DirAcc={rw_metrics['dir_acc']}%")
    
    # 2. SMA
    print(f"\n  [2/3] SMA({CONFIG['sma_window']})...")
    sma_pred = sma_forecast(train, test, CONFIG['sma_window'])
    sma_metrics = compute_metrics(actual, sma_pred, f"SMA({CONFIG['sma_window']})")
    results.append(sma_metrics)
    predictions_all['sma'] = sma_pred
    print(f"    RMSE={sma_metrics['rmse']}, MAE={sma_metrics['mae']}, DirAcc={sma_metrics['dir_acc']}%")
    
    # 3. ARIMA
    print(f"\n  [3/3] ARIMA{CONFIG['arima_order']}...")
    arima_pred = arima_forecast(train, test, CONFIG['arima_order'])
    arima_metrics = compute_metrics(actual, arima_pred, f"ARIMA{CONFIG['arima_order']}")
    results.append(arima_metrics)
    predictions_all['arima'] = arima_pred
    print(f"    RMSE={arima_metrics['rmse']}, MAE={arima_metrics['mae']}, DirAcc={arima_metrics['dir_acc']}%")
    
    return pd.DataFrame(results), predictions_all, test


if __name__ == "__main__":
    
    assets = {
        "SPY": os.path.join(DATA_DIR, "equities", "spy_daily.csv"),
        "BTC": os.path.join(DATA_DIR, "crypto", "btc_usd_full.csv"),
    }
    
    all_results = []
    
    print("\n[BENCHMARK PIPELINE]")
    
    for asset_name, filepath in assets.items():
        if os.path.exists(filepath):
            print(f"\n{'='*60}")
            print(f"ASSET: {asset_name}")
            print(f"{'='*60}")
            
            try:
                results_df, predictions, test = run_benchmarks(filepath, asset_name)
                results_df['asset'] = asset_name
                all_results.append(results_df)
                
                # Save predictions
                pred_df = test[['Close']].copy()
                pred_df.columns = ['Actual']
                for model, preds in predictions.items():
                    pred_df[model] = preds
                
                pred_path = os.path.join(OUTPUT_DIR, f"{asset_name.lower()}_predictions.csv")
                pred_df.to_csv(pred_path)
                print(f"\n  Saved predictions to {pred_path}")
                
            except Exception as e:
                print(f"  [ERROR] {asset_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  [SKIP] {asset_name}: File not found")
    
    # Aggregate results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns
        final_results = final_results[['asset', 'model', 'rmse', 'mae', 'dir_acc']]
        
        # Save
        results_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
        final_results.to_csv(results_path, index=False)
        
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(final_results.to_string(index=False))
        print(f"\nSaved to {results_path}")
    
    print("\n" + "=" * 60)
    print("BASELINE PIPELINE COMPLETE")
    print("=" * 60)
