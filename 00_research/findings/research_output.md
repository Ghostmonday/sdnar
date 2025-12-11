# S-DNA Phase I Research Report

## Establishing the Labeled Gold Standard Dataset and Algorithmic Baselines

_Generated: 2024-12-10_
_Source: Gemini Research Agent_

---

## 1. Executive Summary

The initiation of Phase I for the S-DNA predictive analytics engine marks a critical transition from theoretical design to empirical validation. The primary objective of this phase is the construction of a "Labeled Gold Standard Dataset"—a rigorous, statistically sanitized, and regime-diverse repository of financial time series that will serve as the ground truth for training the S-DNA model.

### Key Findings

| Finding                                        | Implication                                  |
| ---------------------------------------------- | -------------------------------------------- |
| Fixed-time horizon labeling is flawed          | **Triple-Barrier Method (TBM)** is mandatory |
| Markets exhibit distinct regimes               | **TCPD with PELT/RBF** for regime detection  |
| LSTM outperforms ARIMA by 84-87%               | Deep learning baseline is competitive        |
| Transformers have marginal advantage over LSTM | Cost-benefit analysis required               |

### Three Regime Archetypes for Training

1. **2007-2009 GFC** — Systemic credit failure, correlation breakdown
2. **2017 Crypto Bubble** — Speculative mania, parabolic advance
3. **2020 Covid Pandemic** — Exogenous shock, V-shape recovery

---

## 2. Data Acquisition Strategy

### 2.1 Global Financial Crisis (2007-2009)

| Asset           | Source   | Rationale                         |
| --------------- | -------- | --------------------------------- |
| SPY             | yfinance | S&P 500 proxy, -57% drawdown      |
| VIX             | FRED     | Fear index, volatility clustering |
| TED Spread      | FRED     | Interbank credit risk             |
| Treasury 2Y/10Y | FRED     | Yield curve inversion signal      |

### 2.2 Crypto Bubble (2017)

| Asset   | Source             | Granularity    |
| ------- | ------------------ | -------------- |
| BTC-USD | CoinAPI Flat Files | 1-minute OHLCV |
| ETH-USD | CoinAPI Flat Files | 1-minute OHLCV |

**Note**: Gap-filling protocol required for exchange downtime.

### 2.3 Covid Pandemic (2020)

| Asset   | Source   | Rationale               |
| ------- | -------- | ----------------------- |
| QQQ     | yfinance | Tech/stay-at-home trade |
| XLE     | yfinance | Energy sector collapse  |
| WTI Oil | yfinance | Negative price anomaly  |
| VIX     | FRED     | Spike rivaling 2008     |

---

## 3. Triple-Barrier Labeling Method

### 3.1 Why Fixed-Time Horizon Fails

**Scenario**: Buy at $100. Price drops to $80 at t+5 (stop-loss triggered), recovers to $110 at t+10.

- **Fixed-Time Label**: WIN (1) ❌
- **Reality**: LOSS (stopped out) ✅

### 3.2 Barrier Definitions

| Barrier                | Formula               | Parameter   |
| ---------------------- | --------------------- | ----------- |
| Upper (Profit Take)    | `P_t × (1 + M × σ_t)` | M = 2       |
| Lower (Stop Loss)      | `P_t × (1 - M × σ_t)` | M = 2       |
| Vertical (Time Expiry) | `t + T`               | T = 10 days |

### 3.3 Volatility Estimation

```python
def get_daily_volatility(close_prices, span0=20):
    returns = close_prices.pct_change()
    volatility = returns.ewm(span=span0).std()
    return volatility
```

### 3.4 Reproducibility Protocol

Every labeled dataset must include:

- **Source Hash**: SHA-256 of raw input CSV
- **Config Hash**: SHA-256 of parameters JSON
- **Output Hash**: SHA-256 of labeled output

---

## 4. Anomaly Detection: TCPD Protocol

### 4.1 Algorithm Selection

| Algorithm | Speed      | Accuracy               | Decision |
| --------- | ---------- | ---------------------- | -------- |
| BinSeg    | O(N log N) | Approximate            | ❌       |
| **PELT**  | O(N)       | Exact (global optimum) | ✅       |

### 4.2 Cost Function

- **Cost L2**: Detects mean shifts
- **Cost RBF**: Detects distributional changes (volatility regimes) ✅

### 4.3 Implementation

```python
import ruptures as rpt

def detect_regimes(log_returns, penalty=10):
    signal = log_returns.values.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=penalty)
    return result
```

---

## 5. Benchmark League Table

### Tier Structure

| Tier              | Models                      | Purpose              |
| ----------------- | --------------------------- | -------------------- |
| 1 - Naive         | Random Walk, Buy-and-Hold   | Sanity check         |
| 2 - Technical     | SMA 50/200, Bollinger Bands | Trader's logic       |
| 3 - Econometric   | ARIMA, GARCH                | Statistical standard |
| 4 - Deep Learning | LSTM, Transformer           | State-of-the-art     |

### Literature Benchmarks

| Comparison          | Result                  | Source |
| ------------------- | ----------------------- | ------ |
| LSTM vs ARIMA       | 84-87% error reduction  | [1]    |
| Transformer vs LSTM | Marginal advantage      | [3]    |
| SMA vs Buy-Hold     | Lower Sharpe (whipsaws) | [25]   |

---

## 6. Metric Matrix

### Signal Quality

| Metric                       | Formula          | Target             |
| ---------------------------- | ---------------- | ------------------ |
| Information Coefficient (IC) | Corr(ŷ, y)       | > 0.05             |
| RankIC (Spearman)            | Rank correlation | Robust to outliers |

### Risk-Adjusted

| Metric           | Formula        | Target |
| ---------------- | -------------- | ------ |
| Sharpe Ratio     | (Rp - Rf) / σp | > 1.5  |
| Maximum Drawdown | Peak-to-trough | < 20%  |

### Statistical Significance

| Test            | Purpose          | Threshold |
| --------------- | ---------------- | --------- |
| Diebold-Mariano | Model comparison | p < 0.05  |

---

## 7. Implementation Roadmap (Phase II Tasks)

### 7.1 Data Engineering Tasks

- [ ] **Ingest yfinance**: SPY, QQQ, GLD daily/hourly (2000-2024)
- [ ] **Ingest CoinAPI**: BTC-USD 1-minute flat files (2017)
- [ ] **Ingest Macro**: FRED API for VIX, Fed Funds Rate

### 7.2 Labeling Pipeline Tasks

- [ ] **Code Volatility Estimator**: EWM std (span=20)
- [ ] **Code Triple Barrier**: M=2, T=10 parameters
- [ ] **Reproducibility System**: SHA-256 hash generation

### 7.3 Benchmark Execution Tasks

- [ ] **Baseline Generation**: Tier 1/2/3 on all regimes
- [ ] **Deep Learning Training**: LSTM, Transformer on Gold Standard
- [ ] **Evaluation**: IC, RankIC, Sharpe, DM-Test
- [ ] **Reporting**: League Table visualization

---

## References

1. CoinAPI OHLCV Flat Files
2. Triple-Barrier Method (Lopez de Prado)
3. Turing Change Point Detection & Ruptures Library
4. LSTM vs ARIMA Benchmarks
5. Information Coefficient (IC) & RankIC
6. Diebold-Mariano Statistical Tests
7. yfinance Data Access
8. Sharpe Ratio Benchmarks

---

_Signed,_
_Lead Quantitative Researcher_
_S-DNA Project_
