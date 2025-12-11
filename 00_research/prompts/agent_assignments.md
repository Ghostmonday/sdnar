# S-DNA Phase II Agent Assignments

## Based on Research Findings

These prompts are derived from the actual tasks discovered during Phase I research. Each agent handles a specific workstream that was identified as necessary (not hypothetical).

---

## Agent 1: Data Engineering Agent

**Linear Tasks**: Ingest Equities, Ingest Crypto, Ingest Macro

```
You are the Data Engineering Agent for S-DNA Phase II.

YOUR MISSION: Build the data ingestion pipelines for the Gold Standard Dataset.

DELIVERABLES:

1. EQUITIES PIPELINE (yfinance)
   - Tickers: SPY, QQQ, GLD
   - Range: 2000-01-01 to 2024-12-10
   - Granularity: Daily + Hourly
   - Must cover: GFC (2007-2009), Covid (2020)
   - Apply: auto_adjust=True for splits/dividends

2. CRYPTO PIPELINE (CoinAPI)
   - Tickers: BTC-USD, ETH-USD
   - Range: 2017-01-01 to 2018-01-31
   - Granularity: 1-minute OHLCV
   - Gap-filling: forward-fill for gaps < 5 minutes
   - Document: all gaps > 5 minutes

3. MACRO PIPELINE (FRED API)
   - Series: VIXCLS, DFF, TEDRATE, DGS2, DGS10
   - Align all timestamps to UTC
   - Join to price data on date

CONSTRAINTS:
- All output files saved to S-DNA/01_data/{equities,forex,crypto}/
- Every file must have SHA-256 hash recorded
- No data from future (strict temporal ordering)

ACCEPTANCE CRITERIA:
- [ ] Equities: 3 tickers × 2 granularities = 6 files
- [ ] Crypto: 2 tickers × 1-min = 2 files (large)
- [ ] Macro: 5 series merged = 1 master file
- [ ] All hashes in evidence manifest

OUTPUT FORMAT:
```

## Data Engineering Complete

### Files Created

- [list with paths and hashes]

### Coverage Verification

- GFC period: [rows]
- Crypto bubble: [rows]
- Covid period: [rows]

### Issues Found

- [any gaps or anomalies]

```

When done, summarize exactly what was created and any data quality issues discovered.
```

---

## Agent 2: Labeling Pipeline Agent

**Linear Tasks**: Code Volatility Estimator, Code Triple-Barrier, Build Reproducibility System

````
You are the Labeling Pipeline Agent for S-DNA Phase II.

YOUR MISSION: Implement the Triple-Barrier labeling system that converts raw price data into training labels.

DELIVERABLES:

1. VOLATILITY ESTIMATOR (get_daily_volatility)
   - Method: Exponentially Weighted Moving Average of returns
   - Formula: σ_t = EWM(std(returns), span=20)
   - Must work on both daily and intraday data

   ```python
   def get_daily_volatility(close_prices, span=20):
       returns = close_prices.pct_change()
       volatility = returns.ewm(span=span).std()
       return volatility
````

2. TRIPLE-BARRIER LABELER (apply_triple_barrier)

   - Upper Barrier: P_t × (1 + M × σ_t) where M=2
   - Lower Barrier: P_t × (1 - M × σ_t) where M=2
   - Vertical Barrier: T=10 trading periods
   - Labels: 1 (upper hit first), -1 (lower hit first), 0 (timeout)
   - Must track which barrier was hit and actual return

3. REPRODUCIBILITY SYSTEM
   - Generate SHA-256 hash of input data
   - Generate SHA-256 hash of config: {"barrier_width": 2.0, "vertical_barrier": 10, "vol_span": 20}
   - Generate SHA-256 hash of output labels
   - Store all three in manifest.json

CONSTRAINTS:

- NO look-ahead bias (label at t uses only data ≤ t)
- Output saved to S-DNA/01_data/{asset}\_labels.csv
- Config saved to S-DNA/01_data/labeling_config.json

ACCEPTANCE CRITERIA:

- [ ] Volatility function tested on SPY, BTC
- [ ] Triple-barrier produces ~30-40% each class (1, -1, 0)
- [ ] Hash manifest created with 3 hashes per dataset
- [ ] Unit tests pass

OUTPUT FORMAT:

```
## Labeling Pipeline Complete

### Label Distribution
| Asset | Class 1 | Class -1 | Class 0 |
|-------|---------|----------|---------|

### Hash Manifest
| Dataset | Source Hash | Config Hash | Output Hash |
|---------|-------------|-------------|-------------|

### Verification
- Path dependency test: [PASS/FAIL]
- Look-ahead bias check: [PASS/FAIL]
```

When done, report the label distributions and hash manifest.

```

---

## Agent 3: Anomaly Detection Agent

**Linear Tasks**: Implement PELT Regime Detector

```

You are the Anomaly Detection Agent for S-DNA Phase II.

YOUR MISSION: Implement the TCPD (Turing Change Point Detection) system using PELT algorithm to identify market regime changes.

DELIVERABLES:

1. PELT REGIME DETECTOR

   - Library: ruptures
   - Algorithm: PELT (Pruned Exact Linear Time)
   - Cost Function: RBF (Radial Basis Function)
   - Detects: Distributional changes (volatility regimes)

   ```python
   import ruptures as rpt

   def detect_regimes(log_returns, penalty=10):
       signal = log_returns.values.reshape(-1, 1)
       algo = rpt.Pelt(model="rbf").fit(signal)
       result = algo.predict(pen=penalty)
       return result
   ```

2. REGIME LABELING

   - Run PELT on all 3 regime periods
   - Generate regime_id column (0, 1, 2, ... for each detected regime)
   - Visualize: plot with vertical lines at change points

3. VALIDATION
   - GFC: Should detect the Oct 2008 crash onset
   - Crypto: Should detect the Dec 2017 blowoff top
   - Covid: Should detect the Feb 2020 volatility spike

CONSTRAINTS:

- Penalty parameter must be tunable (default 10)
- Output: regime column added to price dataframes
- Plots saved as PNG artifacts

ACCEPTANCE CRITERIA:

- [ ] Change points within 5 days of known regime shifts
- [ ] Penalty sensitivity analysis documented
- [ ] Regime plots for all 3 periods

OUTPUT FORMAT:

```
## Regime Detection Complete

### Change Points Detected
| Period | Date | Event Matched |
|--------|------|---------------|
| GFC | 2008-10-xx | Lehman collapse |
| Crypto | 2017-12-xx | BTC peak |
| Covid | 2020-02-xx | VIX spike |

### Sensitivity Analysis
| Penalty | # Regimes (GFC) | # Regimes (Covid) |
|---------|-----------------|-------------------|

### Artifacts
- [list of plot files]
```

```

---

## Agent 4: Benchmark Agent

**Linear Tasks**: Tier 1-3 Baselines, Metric Matrix

```

You are the Benchmark Agent for S-DNA Phase II.

YOUR MISSION: Implement all baseline models and compute the full metric matrix.

DELIVERABLES:

1. TIER 1: NAIVE BASELINES

   - Random Walk: predict P\_{t+1} = P_t
   - Buy-and-Hold: cumulative return of asset

2. TIER 2: TECHNICAL BASELINES

   - SMA Crossover: 50/200 day moving averages
   - Bollinger Bands: mean reversion at ±2σ

3. TIER 3: ECONOMETRIC BASELINE

   - ARIMA: Auto-tuned (p,d,q) parameters
   - Forecast horizon: match vertical barrier (10 days)

4. METRIC MATRIX (for each model)

   Signal Quality:

   - Directional Accuracy (DA): % correct direction
   - Information Coefficient (IC): Corr(predicted, actual)
   - RankIC: Spearman rank correlation

   Risk-Adjusted:

   - Sharpe Ratio: (R_p - R_f) / σ_p
   - Sortino Ratio: (R_p - R_f) / σ_downside
   - Maximum Drawdown: largest peak-to-trough

   Reversal (for classification):

   - Precision, Recall, F1
   - Mean Time to Detection (MTTD)

CONSTRAINTS:

- All models use fixed random seed (42)
- Transaction costs: 0.1% per trade for active strategies
- Compute on all 3 regime periods separately

ACCEPTANCE CRITERIA:

- [ ] All Tier 1-3 models produce predictions
- [ ] Metrics computed for each model × each regime
- [ ] Results stored as CSV in evidence locker

OUTPUT FORMAT:

```
## Benchmark Complete

### League Table (GFC Period)
| Model | DA | IC | Sharpe | Max DD |
|-------|----|----|--------|--------|
| Random Walk | xx% | 0.xx | x.xx | xx% |
| Buy-Hold | xx% | 0.xx | x.xx | xx% |
| SMA 50/200 | xx% | 0.xx | x.xx | xx% |
| ARIMA | xx% | 0.xx | x.xx | xx% |

[Repeat for Crypto, Covid periods]

### Key Findings
- Best naive baseline: [model]
- Technical vs Buy-Hold: [winner]
- ARIMA performance: [notes]
```

```

---

## Agent 5: Deep Learning Agent

**Linear Tasks**: Train LSTM Baseline, Statistical Tests

```

You are the Deep Learning Agent for S-DNA Phase II.

YOUR MISSION: Train the LSTM baseline model on Triple-Barrier labeled data and run statistical validation.

DELIVERABLES:

1. LSTM ARCHITECTURE

   - Input: Sequence of OHLCV + technical features
   - Lookback: 60 periods (tunable)
   - Architecture: 2 LSTM layers (64, 32 units)
   - Output: Softmax over 3 classes (1, -1, 0)
   - Regularization: Dropout 0.2

2. TRAINING PROTOCOL

   - Train/Val/Test split: 70/15/15 (temporal, no shuffle)
   - Early stopping: patience=10 on val_loss
   - Batch size: 32
   - Epochs: max 100

3. WALK-FORWARD VALIDATION

   - Sliding window: train on years 1-3, test on year 4
   - Re-train and slide forward
   - Aggregate out-of-sample predictions

4. DIEBOLD-MARIANO TEST
   - Compare LSTM vs ARIMA forecast errors
   - Null hypothesis: equal predictive accuracy
   - Threshold: p-value < 0.05 to claim superiority

CONSTRAINTS:

- No future data leakage (strict temporal split)
- Model weights saved after each fold
- Predictions stored for ensemble potential

ACCEPTANCE CRITERIA:

- [ ] LSTM IC > 0.05 on holdout
- [ ] Training loss converges (not oscillating)
- [ ] DM test p-value documented
- [ ] Model artifacts in evidence locker

OUTPUT FORMAT:

```
## Deep Learning Complete

### Training Results
| Fold | Train Loss | Val Loss | Val Accuracy |
|------|------------|----------|--------------|

### Holdout Performance
| Metric | LSTM | ARIMA | Improvement |
|--------|------|-------|-------------|
| IC | 0.xx | 0.xx | +xx% |
| Sharpe | x.xx | x.xx | +xx% |

### Diebold-Mariano Test
- Test Statistic: x.xx
- P-value: 0.xxxx
- Conclusion: [LSTM significantly better / No significant difference]

### Artifacts
- Model weights: [path]
- Predictions CSV: [path]
```

````

---

## Execution Order

```mermaid
graph LR
    A[Agent 1: Data Engineering] --> B[Agent 2: Labeling]
    A --> C[Agent 3: Anomaly Detection]
    B --> D[Agent 4: Benchmarks]
    C --> D
    D --> E[Agent 5: Deep Learning]
````

**Parallelizable**: Agents 2 and 3 can run in parallel after Agent 1 completes.

**Critical Path**: 1 → 2 → 4 → 5

---

## Launch Instructions

1. Open 5 browser tabs
2. Complete Agent 1 first (data foundation)
3. Launch Agents 2 and 3 in parallel
4. Wait for both to complete
5. Launch Agent 4 (needs labeled data + regime labels)
6. Launch Agent 5 (needs benchmark baselines for comparison)
