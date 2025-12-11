# 01_data — Gold Standard Dataset

This directory contains the labeled financial time series data used for S-DNA validation.

## Contents

```
01_data/
├── equities/       # Daily OHLCV data
│   ├── spy_daily.csv   # S&P 500 ETF (2000-2024, 6,274 rows)
│   ├── qqq_daily.csv   # Nasdaq 100 ETF (2000-2024, 6,274 rows)
│   └── gld_daily.csv   # Gold ETF (2004-2024, 5,048 rows)
│
├── crypto/         # Cryptocurrency data
│   ├── btc_usd_full.csv   # Bitcoin (2017-2024, 3,737 rows)
│   ├── btc_usd_daily.csv  # Bitcoin daily subset
│   └── eth_usd_daily.csv  # Ethereum daily
│
├── macro/          # Macroeconomic indicators
│   ├── vix_daily.csv      # CBOE Volatility Index
│   └── treasury_10y.csv   # 10-Year Treasury Yield
│
├── labeled/        # Triple-Barrier labeled datasets
│   ├── spy_labeled.csv    # Labels: {-1: bear, 0: neutral, 1: bull}
│   ├── qqq_labeled.csv
│   ├── gld_labeled.csv
│   ├── btc_labeled.csv
│   └── manifest.json      # SHA-256 hashes for reproducibility
│
└── regimes/        # PELT-detected market regimes
    ├── spy_regimes.csv    # Regime labels per row
    ├── btc_regimes.csv
    └── regime_analysis.json  # Regime statistics
```

## Label Distribution

| Asset | Bull (+1) | Neutral (0) | Bear (-1) |
|-------|-----------|-------------|-----------|
| SPY   | 2,735     | 1,369       | 2,158     |
| QQQ   | 2,776     | 1,254       | 2,232     |
| GLD   | 2,238     | 1,086       | 1,712     |
| BTC   | 1,507     | 1,175       | 1,043     |

## Labeling Configuration

- **Volatility Span:** 20 days (EWM)
- **Barrier Width:** ±2σ
- **Vertical Barrier:** 10 days
- **Config Hash:** `14bdc7cf...`

## Data Sources

- **Equities/Macro:** Yahoo Finance via `yfinance`
- **Crypto:** Yahoo Finance (daily), CoinAPI archives (minute-level historical)

## Regime Coverage

The dataset spans three major market stress periods:
1. **2007-2009 GFC** — Global Financial Crisis
2. **2017-2018 Crypto Bubble** — Bitcoin parabolic rise and crash
3. **2020 COVID Crash** — Pandemic-induced volatility spike
