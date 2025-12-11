"""
S-DNA Triple-Barrier Labeling Pipeline
Agent: Claude (Parallel Track)
Purpose: Generate training labels for trend/reversal prediction

Methodology:
- Triple-Barrier Method (Lopez de Prado)
- Dynamic volatility-scaled barriers
- Path-dependent labels (not fixed-time horizon)
"""

import os
import hashlib
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "volatility_span": 20,      # EWM span for volatility estimation
    "barrier_width": 2.0,       # Multiplier for barriers (in std devs)
    "vertical_barrier": 10,     # Days until timeout
    "version": "1.0.0"
}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "01_data")
OUTPUT_DIR = os.path.join(DATA_DIR, "labeled")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("S-DNA TRIPLE-BARRIER LABELING PIPELINE")
print("=" * 60)
print(f"Config: {CONFIG}")

# ============================================================
# STEP 1: VOLATILITY ESTIMATOR
# ============================================================

def get_daily_volatility(close_prices, span=20):
    """
    Compute exponentially weighted moving average of returns volatility.
    
    This is the standard approach from Lopez de Prado's work.
    EWM gives more weight to recent observations, adapting to regime changes.
    
    Args:
        close_prices: Series of adjusted close prices
        span: EWM span (half-life = span / (span + 1))
    
    Returns:
        Series of daily volatility estimates
    """
    # Compute log returns (more stable than percentage returns)
    returns = np.log(close_prices / close_prices.shift(1))
    
    # EWM standard deviation
    volatility = returns.ewm(span=span).std()
    
    return volatility


# ============================================================
# STEP 2: TRIPLE-BARRIER LABELER
# ============================================================

def apply_triple_barrier(df, close_col='Close', config=CONFIG):
    """
    Apply Triple-Barrier Method to generate labels.
    
    For each row (entry point), we check which barrier is hit first:
    - Upper barrier: Profit take (+M*sigma)
    - Lower barrier: Stop loss (-M*sigma)
    - Vertical barrier: Time expiry (T days)
    
    Labels:
        1  = Upper barrier hit first (bullish)
        -1 = Lower barrier hit first (bearish)
        0  = Timeout (vertical barrier hit)
    
    Args:
        df: DataFrame with price data
        close_col: Name of close price column
        config: Configuration dict
    
    Returns:
        DataFrame with labels added
    """
    close = df[close_col]
    
    # Compute volatility
    volatility = get_daily_volatility(close, span=config['volatility_span'])
    
    # Initialize label columns
    labels = []
    barrier_hit = []
    returns_realized = []
    
    M = config['barrier_width']
    T = config['vertical_barrier']
    
    for i in range(len(df)):
        if i + T >= len(df):
            # Near end of data, can't look forward enough
            labels.append(np.nan)
            barrier_hit.append('insufficient_data')
            returns_realized.append(np.nan)
            continue
        
        entry_price = close.iloc[i]
        vol = volatility.iloc[i]
        
        if pd.isna(vol) or vol == 0:
            labels.append(np.nan)
            barrier_hit.append('no_volatility')
            returns_realized.append(np.nan)
            continue
        
        # Define barriers
        upper = entry_price * (1 + M * vol)
        lower = entry_price * (1 - M * vol)
        
        # Get forward price path
        future_prices = close.iloc[i+1 : i+1+T]
        
        # Check which barrier is hit first
        upper_hit_idx = None
        lower_hit_idx = None
        
        for j, price in enumerate(future_prices):
            if upper_hit_idx is None and price >= upper:
                upper_hit_idx = j
            if lower_hit_idx is None and price <= lower:
                lower_hit_idx = j
        
        # Determine label
        if upper_hit_idx is not None and (lower_hit_idx is None or upper_hit_idx < lower_hit_idx):
            labels.append(1)
            barrier_hit.append('upper')
            returns_realized.append((upper - entry_price) / entry_price)
        elif lower_hit_idx is not None and (upper_hit_idx is None or lower_hit_idx < upper_hit_idx):
            labels.append(-1)
            barrier_hit.append('lower')
            returns_realized.append((lower - entry_price) / entry_price)
        else:
            # Vertical barrier (timeout)
            final_price = future_prices.iloc[-1] if len(future_prices) > 0 else entry_price
            ret = (final_price - entry_price) / entry_price
            labels.append(0)
            barrier_hit.append('vertical')
            returns_realized.append(ret)
    
    # Add to dataframe
    df = df.copy()
    df['volatility'] = volatility
    df['label'] = labels
    df['barrier_hit'] = barrier_hit
    df['return_realized'] = returns_realized
    
    return df


# ============================================================
# STEP 3: HASH GENERATION FOR REPRODUCIBILITY
# ============================================================

def compute_hash(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_manifest(input_files, config, output_files):
    """Create reproducibility manifest with hashes."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "inputs": {},
        "outputs": {}
    }
    
    for name, path in input_files.items():
        if os.path.exists(path):
            manifest["inputs"][name] = {
                "path": path,
                "hash": compute_hash(path)
            }
    
    for name, path in output_files.items():
        if os.path.exists(path):
            manifest["outputs"][name] = {
                "path": path,
                "hash": compute_hash(path)
            }
    
    return manifest


# ============================================================
# STEP 4: PROCESS ALL ASSETS
# ============================================================

def process_asset(name, input_path, output_path):
    """Process a single asset through the labeling pipeline."""
    print(f"\n  Processing {name}...")
    
    # Load data - handle yfinance multi-index headers
    df = pd.read_csv(input_path)
    
    # yfinance saves with row 0 as ticker name, row 1 as "Date" header
    # Skip these rows and re-parse
    if df.iloc[0, 0] == 'Ticker' or str(df.iloc[0, 0]).startswith('Date'):
        # Skip bad rows
        df = pd.read_csv(input_path, skiprows=2)
        # First column is the date
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        # Standard format
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Ensure numeric types
    for col in ['Close', 'High', 'Low', 'Open']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Drop any rows with NaN in Close
    df = df.dropna(subset=['Close'])
    
    print(f"    Loaded {len(df)} rows, Close range: {df['Close'].min():.2f} to {df['Close'].max():.2f}")
    
    # Apply Triple-Barrier
    df_labeled = apply_triple_barrier(df, close_col='Close', config=CONFIG)
    
    # Save
    df_labeled.to_csv(output_path)
    print(f"    Saved to {output_path}")
    
    # Compute label distribution
    label_counts = df_labeled['label'].value_counts(dropna=False)
    total_valid = label_counts.get(1, 0) + label_counts.get(-1, 0) + label_counts.get(0, 0)
    
    if total_valid > 0:
        pct_bull = label_counts.get(1, 0) / total_valid * 100
        pct_bear = label_counts.get(-1, 0) / total_valid * 100
        pct_neutral = label_counts.get(0, 0) / total_valid * 100
        print(f"    Labels: Bull={pct_bull:.1f}%, Bear={pct_bear:.1f}%, Neutral={pct_neutral:.1f}%")
    
    return df_labeled, label_counts


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # Define assets to process
    assets = {
        "SPY": os.path.join(DATA_DIR, "equities", "spy_daily.csv"),
        "QQQ": os.path.join(DATA_DIR, "equities", "qqq_daily.csv"),
        "GLD": os.path.join(DATA_DIR, "equities", "gld_daily.csv"),
        "BTC": os.path.join(DATA_DIR, "crypto", "btc_usd_full.csv"),
    }
    
    input_files = {}
    output_files = {}
    all_distributions = {}
    
    print("\n[LABELING PIPELINE]")
    
    for name, input_path in assets.items():
        if os.path.exists(input_path):
            output_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_labeled.csv")
            
            try:
                df_labeled, dist = process_asset(name, input_path, output_path)
                input_files[name] = input_path
                output_files[name] = output_path
                all_distributions[name] = {
                    "bull": int(dist.get(1, 0)),
                    "bear": int(dist.get(-1, 0)),
                    "neutral": int(dist.get(0, 0)),
                    "invalid": int(dist.get(np.nan, 0)) if np.nan in dist.index else 0
                }
            except Exception as e:
                print(f"    [ERROR] {name}: {e}")
        else:
            print(f"    [SKIP] {name}: File not found")
    
    # Create manifest
    print("\n[CREATING MANIFEST]")
    manifest = create_manifest(input_files, CONFIG, output_files)
    manifest["label_distributions"] = all_distributions
    
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest to {manifest_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("LABELING PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"\nConfig Hash: {manifest['config_hash'][:16]}...")
    print(f"\nLabel Distributions:")
    for name, dist in all_distributions.items():
        total = dist['bull'] + dist['bear'] + dist['neutral']
        if total > 0:
            print(f"  {name}: Bull={dist['bull']/total*100:.1f}%, Bear={dist['bear']/total*100:.1f}%, Neutral={dist['neutral']/total*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
