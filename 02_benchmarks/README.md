# 02_benchmarks — Baseline Models

This directory contains baseline model implementations for comparative analysis against S-DNA.

## Benchmark Tiers

| Tier | Models | Purpose |
|------|--------|---------|
| **1 — Naive** | Random Walk | Sanity check (EMH null hypothesis) |
| **2 — Technical** | SMA(20) | Traditional trend-following |
| **3 — Econometric** | ARIMA(5,1,0) | Statistical autocorrelation |
| **4 — Deep Learning** | LSTM | State-of-the-art neural baseline |

## Files

| File | Description |
|------|-------------|
| `run_baselines.py` | Execute Tier 1-3 baselines on all assets |
| `lstm_baseline.py` | Basic LSTM implementation (398 lines) |
| `sdna_lstm_full.py` | Full S-DNA LSTM with research insights (587 lines) |
| `lstm_critical_pieces.py` | Key code snippets for reference |
| `baseline_results.csv` | Tier 1-3 results (RMSE, MAE, DirAcc) |
| `spy_predictions.csv` | SPY model predictions (1,257 rows) |
| `btc_predictions.csv` | BTC model predictions |

## Current Results

| Asset | Model | RMSE | Dir. Accuracy |
|-------|-------|------|---------------|
| SPY | Random Walk | **4.54** | 49.4% |
| SPY | SMA(20) | 11.60 | 53.9% |
| SPY | ARIMA(5,1,0) | 9.37 | 48.7% |
| BTC | Random Walk | **1,322** | 46.1% |
| BTC | SMA(20) | 3,709 | 51.0% |
| BTC | ARIMA(5,1,0) | 1,453 | 46.1% |

## Success Criteria

S-DNA must beat Random Walk with:
- **RMSE:** Lower than baseline
- **Diebold-Mariano p-value:** < 0.05 (statistically significant)
- **Directional Accuracy:** > 52% (significantly above chance)

## Usage

```bash
# Run all baselines
python run_baselines.py

# Train LSTM
python lstm_baseline.py

# Full S-DNA LSTM (with volatility normalization, directional loss)
python sdna_lstm_full.py
```
