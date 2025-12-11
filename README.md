# S-DNA: Sentiment-Driven Neural Analytics

> **Predictive analytics engine for trend detection, reversal prediction, and anomaly identification in financial markets.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Phase I Complete](https://img.shields.io/badge/Status-Phase%20I%20Complete-green.svg)]()

---

## 🎯 Project Overview

S-DNA is a validation framework for testing predictive signals against institutional-grade baselines. The goal: prove (or disprove) that the S-DNA signal beats random walk, technical indicators, and econometric models with statistical significance.

### Claims Under Validation
- **Directional Accuracy** > 55% (better than coin flip)
- **Sharpe Ratio** > 1.0 (risk-adjusted returns)
- **MTTD** < 3 bars (faster reversal detection than SMA)

---

## 📊 Baseline Results (Phase I)

| Asset | Model | RMSE | Directional Acc |
|-------|-------|------|-----------------|
| SPY | Random Walk | **4.54** | 49.4% |
| SPY | SMA(20) | 11.60 | 53.9% |
| SPY | ARIMA(5,1,0) | 9.37 | 48.7% |
| BTC | Random Walk | **1,322** | 46.1% |
| BTC | SMA(20) | 3,709 | 51.0% |
| BTC | ARIMA(5,1,0) | 1,453 | 46.1% |

**Target:** Beat Random Walk RMSE with p < 0.05 (Diebold-Mariano test)

---

## 🏗️ Architecture

```
S-DNA/
├── 00_research/        # Research prompts, scripts, methodology
│   ├── scripts/        # Data ingestion, labeling, regime detection
│   ├── prompts/        # Multi-agent research coordination
│   └── deep_research_package/  # Research synthesis & findings
│
├── 01_data/            # Gold Standard Dataset
│   ├── equities/       # SPY, QQQ, GLD (2000-2024)
│   ├── crypto/         # BTC, ETH (2017-2024)
│   ├── macro/          # VIX, Treasury yields
│   ├── labeled/        # Triple-Barrier labeled datasets
│   └── regimes/        # PELT-detected market regimes
│
├── 02_benchmarks/      # Baseline implementations
│   ├── run_baselines.py
│   ├── lstm_baseline.py
│   └── sdna_lstm_full.py
│
├── 03_metrics/         # Metric definitions & league tables
├── 04_evidence/        # Hash-stamped research artifacts
├── 05_validation/      # SR 11-7 compliance dossier
└── SentimentDNA-Linear/  # Project management (Linear import)
```

---

## 🔬 Methodology

### Triple-Barrier Labeling
Path-dependent labels using volatility-adjusted barriers:
- **Upper Barrier:** P₀ × (1 + 2σ) → Profit take
- **Lower Barrier:** P₀ × (1 - 2σ) → Stop loss  
- **Vertical Barrier:** 10 days → Time expiry

### Regime Detection
PELT algorithm with RBF cost function detects distributional changes:
- 8 regimes identified in SPY (2000-2024)
- GFC (2007-2009), COVID crash (2020) correctly flagged

### LSTM Architecture
```
Input: (batch, 60, 6) — 60-day lookback, 6 features
    ↓
Bidirectional LSTM(128) + BatchNorm + Dropout(0.3)
    ↓
Bidirectional LSTM(64) + BatchNorm + Dropout(0.3)
    ↓
Dense(32, swish) + Dropout(0.2)
    ↓
Output: Dense(3, softmax) → {Bear, Neutral, Bull}
```

**Key Innovations:**
- Volatility-normalized inputs (homoskedastic)
- Directional-MSE loss (penalizes wrong signs)
- Purged walk-forward validation (no data leakage)

---

## 📈 Data Coverage

| Asset | Period | Rows | Regimes Covered |
|-------|--------|------|-----------------|
| SPY | 2000-2024 | 6,274 | GFC, Recovery, COVID |
| QQQ | 2000-2024 | 6,274 | Dot-com, GFC, COVID |
| GLD | 2004-2024 | 5,048 | Gold bull run |
| BTC | 2017-2024 | 3,737 | Crypto bubble, bear, recovery |
| VIX | 2000-2024 | 6,274 | All volatility regimes |

---

## 🚀 Quick Start

```bash
# Clone (private repo - requires access)
git clone https://github.com/Ghostmonday/sdnar.git
cd sdnar

# Install dependencies
pip install -r requirements.txt

# Run baselines
python 02_benchmarks/run_baselines.py

# Train LSTM
python 02_benchmarks/lstm_baseline.py
```

---

## 📋 Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **I** | ✅ Complete | Research, data acquisition, baseline benchmarks |
| **II** | 🔄 Next | LSTM training, hyperparameter tuning |
| **III** | ⏳ Planned | Walk-forward validation, regime-specific testing |
| **IV** | ⏳ Planned | Stress testing, blind holdout evaluation |
| **V** | ⏳ Planned | SR 11-7 validation dossier, production readiness |

---

## 📚 Research Artifacts

All research is documented in `/04_evidence/`:
- `deep_research_full.md` — LSTM architecture synthesis
- `data_engineering_gemini_report.md` — Data pipeline documentation
- `labeling_gemini_report.md` — Triple-Barrier methodology
- `order5_research_synthesis.md` — Multi-agent consensus

---

## 🔐 Governance

- All datasets include SHA-256 hashes for reproducibility
- Fixed random seeds throughout pipeline
- Evidence locker for audit trail
- SR 11-7 compliant validation framework

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Built with multi-agent AI research coordination (GPT, Claude, Gemini)*
