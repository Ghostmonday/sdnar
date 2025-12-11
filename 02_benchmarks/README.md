# Benchmark Models

Baseline implementations for comparative analysis.

## Tier Structure

### Tier 1: Naive (Sanity Check)

- `naive/random_walk.py` — Predict next = current
- `naive/buy_and_hold.py` — Passive accumulation

### Tier 2: Technical (Competitors)

- `technical/sma_crossover.py` — 50/200 SMA
- `technical/rsi_divergence.py` — RSI-based reversals
- `technical/bollinger_atr.py` — Volatility bands

### Tier 3: Advanced (State-of-Art)

- `econometric/arima.py` — ARIMA forecasting
- `econometric/garch.py` — Volatility modeling
- `deep_learning/lstm.py` — LSTM baseline
- `deep_learning/transformer.py` — Transformer baseline

## Requirements

- All models use fixed seeds
- Results stored as hash-stamped ZIPs
- Model cards required for each
