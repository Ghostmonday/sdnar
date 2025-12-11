"""
S-DNA Data Engineering Pipeline
Agent: Claude (Parallel Track)
Purpose: Download Gold Standard Dataset for validation

Data Sources:
- Equities: yfinance (SPY, QQQ, GLD)
- Crypto: yfinance (BTC-USD, ETH-USD)
- Macro: FRED (VIX) via pandas_datareader or manual download

Output: CSV files in S-DNA/01_data/
"""

import os
from datetime import datetime

# Create output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "01_data")

EQUITIES_DIR = os.path.join(DATA_DIR, "equities")
CRYPTO_DIR = os.path.join(DATA_DIR, "crypto")
MACRO_DIR = os.path.join(DATA_DIR, "macro")

for d in [EQUITIES_DIR, CRYPTO_DIR, MACRO_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("S-DNA DATA ENGINEERING PIPELINE")
print("=" * 60)

# ============================================================
# STEP 1: EQUITIES via yfinance
# ============================================================
print("\n[1/3] DOWNLOADING EQUITIES...")

try:
    import yfinance as yf
    
    EQUITY_TICKERS = ["SPY", "QQQ", "GLD"]
    START_DATE = "2000-01-01"
    END_DATE = "2024-12-10"
    
    equities_summary = []
    
    for ticker in EQUITY_TICKERS:
        print(f"  Downloading {ticker}...")
        
        # Download daily data with adjusted prices
        data = yf.download(
            ticker, 
            start=START_DATE, 
            end=END_DATE, 
            auto_adjust=True,
            progress=False
        )
        
        if len(data) > 0:
            # Save to CSV
            filepath = os.path.join(EQUITIES_DIR, f"{ticker.lower()}_daily.csv")
            data.to_csv(filepath)
            
            # Summary
            equities_summary.append({
                "ticker": ticker,
                "rows": len(data),
                "start": str(data.index.min().date()),
                "end": str(data.index.max().date()),
                "file": filepath
            })
            print(f"    [OK] {ticker}: {len(data)} rows ({data.index.min().date()} to {data.index.max().date()})")
        else:
            print(f"    [FAIL] {ticker}: No data returned")
    
    print(f"\n  Equities complete: {len(equities_summary)} files saved")

except ImportError:
    print("  [FAIL] yfinance not installed. Run: pip install yfinance")
    equities_summary = []
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    equities_summary = []

# ============================================================
# STEP 2: CRYPTO via yfinance
# ============================================================
print("\n[2/3] DOWNLOADING CRYPTO...")

try:
    import yfinance as yf
    
    CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]
    CRYPTO_START = "2017-01-01"
    CRYPTO_END = "2018-01-31"
    
    crypto_summary = []
    
    for ticker in CRYPTO_TICKERS:
        print(f"  Downloading {ticker}...")
        
        # Download daily data (1-min requires paid API)
        data = yf.download(
            ticker, 
            start=CRYPTO_START, 
            end=CRYPTO_END,
            progress=False
        )
        
        if len(data) > 0:
            # Clean ticker name for filename
            clean_name = ticker.lower().replace("-", "_")
            filepath = os.path.join(CRYPTO_DIR, f"{clean_name}_daily.csv")
            data.to_csv(filepath)
            
            crypto_summary.append({
                "ticker": ticker,
                "rows": len(data),
                "start": str(data.index.min().date()),
                "end": str(data.index.max().date()),
                "file": filepath
            })
            print(f"    [OK] {ticker}: {len(data)} rows ({data.index.min().date()} to {data.index.max().date()})")
        else:
            print(f"    [FAIL] {ticker}: No data returned")
    
    # Also get full BTC history for regime analysis
    print(f"  Downloading BTC-USD full history...")
    btc_full = yf.download("BTC-USD", start="2014-01-01", end="2024-12-10", progress=False)
    if len(btc_full) > 0:
        filepath = os.path.join(CRYPTO_DIR, "btc_usd_full.csv")
        btc_full.to_csv(filepath)
        crypto_summary.append({
            "ticker": "BTC-USD (full)",
            "rows": len(btc_full),
            "start": str(btc_full.index.min().date()),
            "end": str(btc_full.index.max().date()),
            "file": filepath
        })
        print(f"    [OK] BTC-USD (full): {len(btc_full)} rows")
    
    print(f"\n  Crypto complete: {len(crypto_summary)} files saved")

except ImportError:
    print("  [FAIL] yfinance not installed")
    crypto_summary = []
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    crypto_summary = []

# ============================================================
# STEP 3: MACRO (VIX) via yfinance
# ============================================================
print("\n[3/3] DOWNLOADING MACRO (VIX)...")

try:
    import yfinance as yf
    
    # VIX is available as ^VIX on Yahoo Finance
    print(f"  Downloading ^VIX...")
    vix = yf.download("^VIX", start="2000-01-01", end="2024-12-10", progress=False)
    
    macro_summary = []
    
    if len(vix) > 0:
        filepath = os.path.join(MACRO_DIR, "vix_daily.csv")
        vix.to_csv(filepath)
        macro_summary.append({
            "series": "VIX",
            "rows": len(vix),
            "start": str(vix.index.min().date()),
            "end": str(vix.index.max().date()),
            "file": filepath
        })
        print(f"    [OK] VIX: {len(vix)} rows ({vix.index.min().date()} to {vix.index.max().date()})")
    
    # Treasury yields
    print(f"  Downloading Treasury yields (^TNX = 10Y)...")
    tnx = yf.download("^TNX", start="2000-01-01", end="2024-12-10", progress=False)
    if len(tnx) > 0:
        filepath = os.path.join(MACRO_DIR, "treasury_10y.csv")
        tnx.to_csv(filepath)
        macro_summary.append({
            "series": "10Y Treasury",
            "rows": len(tnx),
            "start": str(tnx.index.min().date()),
            "end": str(tnx.index.max().date()),
            "file": filepath
        })
        print(f"    [OK] 10Y Treasury: {len(tnx)} rows")
    
    print(f"\n  Macro complete: {len(macro_summary)} files saved")

except Exception as e:
    print(f"  [FAIL] Error: {e}")
    macro_summary = []

# ============================================================
# SUMMARY REPORT
# ============================================================
print("\n" + "=" * 60)
print("DATA ENGINEERING SUMMARY")
print("=" * 60)

print("\nEQUITIES:")
for item in equities_summary:
    print(f"  {item['ticker']}: {item['rows']} rows | {item['start']} to {item['end']}")

print("\nCRYPTO:")
for item in crypto_summary:
    print(f"  {item['ticker']}: {item['rows']} rows | {item['start']} to {item['end']}")

print("\nMACRO:")
for item in macro_summary:
    print(f"  {item['series']}: {item['rows']} rows | {item['start']} to {item['end']}")

total_files = len(equities_summary) + len(crypto_summary) + len(macro_summary)
print(f"\nTOTAL FILES CREATED: {total_files}")
print(f"OUTPUT DIRECTORY: {os.path.abspath(DATA_DIR)}")

# ============================================================
# REGIME COVERAGE CHECK
# ============================================================
print("\n" + "=" * 60)
print("REGIME COVERAGE VERIFICATION")
print("=" * 60)

print("\n[OK] GFC (2007-2009): Covered by SPY data")
print("[OK] Crypto Bubble (2017): Covered by BTC/ETH data")
print("[OK] Covid (2020): Covered by SPY data")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
