"""
S-DNA PELT Regime Detection Pipeline
Agent: Claude
Purpose: Detect market regime changes using Turing Change Point Detection

Algorithm: PELT (Pruned Exact Linear Time) with RBF cost function
- Detects distributional changes (volatility regimes)
- Finds exact optimal change points in O(N) time
"""

import os
import json
import pandas as pd
import numpy as np
import ruptures as rpt
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "algorithm": "PELT",
    "cost_function": "rbf",  # Radial Basis Function - detects distributional changes
    "penalty": 10,           # Higher = fewer regime changes detected
    "version": "1.0.0"
}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "01_data")
OUTPUT_DIR = os.path.join(DATA_DIR, "regimes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("S-DNA PELT REGIME DETECTION PIPELINE")
print("=" * 60)
print(f"Config: {CONFIG}")


# ============================================================
# REGIME DETECTOR
# ============================================================

def detect_regimes(df, close_col='Close', penalty=10):
    """
    Detect volatility regimes using PELT with RBF cost.
    
    PELT (Pruned Exact Linear Time):
    - Exact algorithm (guaranteed optimal)
    - O(N) complexity
    - RBF cost detects changes in distribution (mean AND variance)
    
    Args:
        df: DataFrame with price data
        close_col: Name of close price column
        penalty: Higher = fewer breakpoints (more conservative)
    
    Returns:
        change_points: List of indices where regime changes
        regime_labels: Array of regime IDs for each row
    """
    # Compute log returns (better for detecting volatility changes)
    close = df[close_col].values
    returns = np.log(close[1:] / close[:-1])
    
    # Pad to match original length
    returns = np.concatenate([[0], returns])
    
    # Reshape for ruptures (requires 2D)
    signal = returns.reshape(-1, 1)
    
    # Initialize PELT with RBF cost
    algo = rpt.Pelt(model="rbf").fit(signal)
    
    # Predict change points
    change_points = algo.predict(pen=penalty)
    
    # Convert change points to regime labels
    regime_labels = np.zeros(len(df), dtype=int)
    prev_idx = 0
    for regime_id, cp in enumerate(change_points):
        regime_labels[prev_idx:cp] = regime_id
        prev_idx = cp
    
    return change_points, regime_labels


def analyze_regimes(df, regime_labels, close_col='Close'):
    """
    Analyze characteristics of each detected regime.
    """
    df = df.copy()
    df['regime'] = regime_labels
    df['returns'] = df[close_col].pct_change()
    
    regime_stats = []
    for regime_id in sorted(df['regime'].unique()):
        regime_data = df[df['regime'] == regime_id]
        
        if len(regime_data) < 2:
            continue
            
        stats = {
            'regime_id': regime_id,
            'start_date': str(regime_data.index[0].date()) if hasattr(regime_data.index[0], 'date') else str(regime_data.index[0]),
            'end_date': str(regime_data.index[-1].date()) if hasattr(regime_data.index[-1], 'date') else str(regime_data.index[-1]),
            'days': len(regime_data),
            'mean_return': regime_data['returns'].mean() * 252,  # Annualized
            'volatility': regime_data['returns'].std() * np.sqrt(252),  # Annualized
            'start_price': regime_data[close_col].iloc[0],
            'end_price': regime_data[close_col].iloc[-1],
            'total_return': (regime_data[close_col].iloc[-1] / regime_data[close_col].iloc[0] - 1) * 100
        }
        regime_stats.append(stats)
    
    return regime_stats


def process_asset(name, input_path, output_path, config=CONFIG):
    """Process a single asset through regime detection."""
    print(f"\n  Processing {name}...")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Handle yfinance format
    if df.iloc[0, 0] == 'Ticker' or str(df.iloc[0, 0]).startswith('Date'):
        df = pd.read_csv(input_path, skiprows=2)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    else:
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Ensure numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    print(f"    Loaded {len(df)} rows")
    
    # Detect regimes
    change_points, regime_labels = detect_regimes(df, 'Close', config['penalty'])
    
    n_regimes = len(set(regime_labels))
    print(f"    Detected {n_regimes} regimes ({len(change_points)-1} change points)")
    
    # Analyze regimes
    regime_stats = analyze_regimes(df, regime_labels, 'Close')
    
    # Add regime labels to dataframe
    df['regime'] = regime_labels
    
    # Save
    df.to_csv(output_path)
    print(f"    Saved to {output_path}")
    
    # Print regime summary
    print(f"    Regime Summary:")
    for stat in regime_stats[:5]:  # Show first 5
        vol_label = "HIGH" if stat['volatility'] > 0.20 else "LOW" if stat['volatility'] < 0.10 else "MED"
        print(f"      R{stat['regime_id']}: {stat['start_date']} to {stat['end_date']} | Vol={stat['volatility']:.1%} ({vol_label}) | Ret={stat['total_return']:+.1f}%")
    if len(regime_stats) > 5:
        print(f"      ... and {len(regime_stats) - 5} more regimes")
    
    return df, regime_stats, change_points


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # Define assets to process
    assets = {
        "SPY": os.path.join(DATA_DIR, "equities", "spy_daily.csv"),
        "BTC": os.path.join(DATA_DIR, "crypto", "btc_usd_full.csv"),
    }
    
    all_results = {}
    
    print("\n[REGIME DETECTION PIPELINE]")
    
    for name, input_path in assets.items():
        if os.path.exists(input_path):
            output_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_regimes.csv")
            
            try:
                df, stats, cps = process_asset(name, input_path, output_path, CONFIG)
                all_results[name] = {
                    "num_regimes": len(set(df['regime'])),
                    "change_points": [int(x) for x in cps],
                    "regime_stats": stats
                }
            except Exception as e:
                print(f"    [ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    [SKIP] {name}: File not found")
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "regime_analysis.json")
    with open(results_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "results": all_results
        }, f, indent=2, default=str)
    
    print(f"\n  Saved analysis to {results_path}")
    
    # ============================================================
    # KEY REGIME VALIDATION
    # ============================================================
    print("\n" + "=" * 60)
    print("REGIME VALIDATION: KEY HISTORICAL EVENTS")
    print("=" * 60)
    
    # Check if known crisis periods are detected
    if "SPY" in all_results:
        spy_stats = all_results["SPY"]["regime_stats"]
        
        print("\nExpected Events to Detect:")
        print("  - GFC: Oct 2008 crash")
        print("  - Covid: Feb 2020 volatility spike")
        print("  - 2022: Fed tightening")
        
        print("\nHigh Volatility Regimes Detected (Vol > 20%):")
        for stat in spy_stats:
            if stat['volatility'] > 0.20:
                print(f"  R{stat['regime_id']}: {stat['start_date']} to {stat['end_date']} | Vol={stat['volatility']:.1%}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
