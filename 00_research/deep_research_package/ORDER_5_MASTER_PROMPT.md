# S-DNA Order 5: Deep Learning Research Mission

## CONTEXT — READ ALL FILES IN THIS FOLDER

You have access to a complete research package for the S-DNA validation framework.

### What This Project Is

S-DNA is an institutional crypto sentiment engine that aggregates signals from 12+ sources (Binance, Bybit, OKX, Reddit, 4chan, Telegram) using Volume-Weighted Z-Score filtering. It claims Sharpe 3.0+ and Max DD < 12%.

### What We've Built So Far

1. **Data Pipeline** — Equities (SPY, QQQ, GLD) + Crypto (BTC, ETH) + Macro (VIX, Treasury)
2. **Labeling** — Triple-Barrier Method with ±2σ volatility-adjusted barriers
3. **Regime Detection** — PELT algorithm identified 8 regimes in SPY (GFC correctly flagged)
4. **Baselines** — Random Walk, SMA, ARIMA benchmarks established

### Current Baseline Performance (THESE ARE THE NUMBERS TO BEAT)

```
SPY:
- Random Walk RMSE: 4.54 (best)
- SMA(20) RMSE: 11.60
- ARIMA(5,1,0) RMSE: 9.37
- Random Walk DirAcc: 49.44%

BTC:
- Random Walk RMSE: 1322 (best)
- SMA(20) RMSE: 3709
- ARIMA(5,1,0) RMSE: 1453
- Random Walk DirAcc: 46.05%
```

---

## YOUR MISSION

Produce a comprehensive research document for implementing LSTM Deep Learning models that can **beat these baselines**.

### Required Deliverables

1. **ARCHITECTURE SPECIFICATION**

   - Input features (stationary? lagged? technical indicators?)
   - Sequence length (lookback window)
   - Network structure (layers, units, bidirectional?)
   - Output layer (classification vs regression)
   - Regularization strategy

2. **TRAINING PROTOCOL**

   - Train/Val/Test split methodology
   - Walk-forward validation approach
   - Class imbalance handling
   - Optimizer, learning rate schedule
   - Loss function selection

3. **EVALUATION FRAMEWORK**

   - Classification: Precision, Recall, F1, AUC-ROC
   - Regression: RMSE, MAE, Directional Accuracy
   - Statistical significance: Diebold-Mariano test (p < 0.05)

4. **FAILURE DIAGNOSTICS**

   - What to check if LSTM doesn't beat Random Walk
   - Overfitting indicators
   - Regime-specific performance analysis

5. **IMPLEMENTATION GUIDANCE**
   - Complete code implementation (TensorFlow/Keras or PyTorch)
   - GPU requirements
   - Expected training time

---

## FILES IN THIS PACKAGE

### Data Files

- `spy_labeled.csv` — 6,274 rows with label, volatility, barrier info
- `btc_labeled.csv` — 3,737 rows with same structure
- `spy_regimes.csv` — Regime labels from PELT detection
- `btc_regimes.csv` — Same for BTC
- `manifest.json` — SHA-256 hashes for reproducibility

### Research Reports (Previous Phases)

- `research_output.md` — Phase I findings
- `data_engineering_gemini_report.md` — Data pipeline research
- `labeling_gemini_report.md` — Triple-Barrier research
- `benchmarks_gemini_research.md` — Baseline research

### Implementation Scripts

- `labeling_pipeline.py` — Triple-Barrier implementation
- `regime_detection.py` — PELT regime detection
- `run_baselines.py` — Random Walk, SMA, ARIMA
- `lstm_baseline.py` — Current LSTM skeleton (needs enhancement)

### Preliminary LSTM Research (Quick Outputs)

- `lstm_research_sonnet.md` — 941 lines, most detailed
- `lstm_research_gpt.md` — 112 lines
- `lstm_research_gemini.md` — 121 lines

---

## QUESTION FOR RESEARCHER

Should I also include the S-DNA source code (Signal Forge engine)?

The repo is at: https://github.com/GhostMonday/S-DNA

Including it would let you understand:

- How signals are generated
- What features S-DNA produces
- How to integrate LSTM with S-DNA outputs

---

## OUTPUT FORMAT

Produce a detailed technical document with:

- Mathematical formulas where applicable
- Code implementations (complete, not snippets)
- Citations to academic literature
- Expected outcomes and success criteria
