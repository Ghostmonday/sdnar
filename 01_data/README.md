# Data Directory

Labeled Gold Standard Dataset storage.

## Subdirectories

- `equities/` — Stock market data (S&P 500, major indices)
- `forex/` — Currency pairs (major and emerging)
- `crypto/` — Cryptocurrency data (BTC, ETH, etc.)

## Data Requirements

- OHLCV format
- Minimum 15 years history (2008-2024)
- Include VIX and macro markers where available

## Labeling

All data files must have corresponding `*_labels.csv` with Triple-Barrier classifications.
