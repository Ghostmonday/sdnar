# 📌 Master Research Prompt — Phase I Kickoff

## Objective

Conduct a comprehensive research run to establish the "Labeled Gold Standard Dataset" and baseline benchmarks for S-DNA, a predictive analytics engine for trend detection, reversal prediction, and anomaly identification.

---

## Scope

### 1. Data Collection

- Gather diverse historical datasets across Equities, Forex, and Crypto.
- Ensure coverage of multiple regimes:
  - 2007–2009 Global Financial Crisis
  - 2017 Crypto Bubble
  - 2020 Covid Pandemic
- Include OHLCV data, volatility indices (VIX), and macroeconomic markers.

### 2. Labeling Standards

- Implement **Triple-Barrier labeling** for trend/reversal classification:
  - Upper Barrier (Profit Take): +2σ of volatility
  - Lower Barrier (Stop Loss): -2σ of volatility
  - Vertical Barrier (Time Expiry): 10 trading days
- Define anomalies using **Turing Change Point Detection (TCPD)** criteria (mean/variance shifts).
- Document labeling scripts with hash-stamped reproducibility.

### 3. Benchmark Baselines

Collect performance metrics for:

| Tier          | Models                                                    |
| ------------- | --------------------------------------------------------- |
| Naive         | Random Walk, Buy-and-Hold                                 |
| Technical     | 50/200 SMA crossover, RSI divergence, Bollinger Bands/ATR |
| Econometric   | ARIMA, GARCH                                              |
| Deep Learning | LSTM, Transformer/TimeGPT                                 |

### 4. Metric Matrix

| Capability    | Metrics                                                         |
| ------------- | --------------------------------------------------------------- |
| Trend         | Directional Accuracy (DA), Information Coefficient (IC), RankIC |
| Reversal      | Precision, Recall, F1, Mean Time to Detection (MTTD)            |
| Anomaly       | AUC-ROC, F-Beta (Recall-weighted), False Positive Rate (FPR)    |
| Risk-Adjusted | Sharpe Ratio, Sortino Ratio, Maximum Drawdown                   |

---

## Deliverables

1. **Consolidated Dataset** — Labeled ground truth across assets and regimes
2. **League Table** — Baseline benchmark results across all metrics
3. **Methodology Documentation** — Labeling specs, anomaly definitions, reproducibility checks
4. **Evidence Artifacts** — Plots, CSVs, PDFs stored in Evidence Locker (hash-stamped)

---

## Checkpoint

Completion of the **Benchmark Performance Report** (Phase II, Milestone 2.1) with:

- Statistically significant comparisons (Diebold-Mariano test, p-value < 0.05)
- S-DNA performance vs. all Tier 1/2/3 baselines

---

## How to Use This Prompt

### Option A: Single Research Agent

Run this entire prompt through one powerful model (Claude, GPT-4, Gemini) with code execution capabilities. The agent will:

1. Research available data sources
2. Document labeling methodology
3. Compile benchmark literature
4. Generate the task list for Phase II

### Option B: Parallel Research Agents

Split into focused prompts:

- **Data Agent**: Scope 1 only (data collection)
- **Labeling Agent**: Scope 2 only (labeling standards)
- **Benchmark Agent**: Scope 3 only (baseline research)
- **Metrics Agent**: Scope 4 only (metric specifications)

### Recommended Model Selection

| Task Type           | Best Model    | Why                                |
| ------------------- | ------------- | ---------------------------------- |
| Literature research | Claude/GPT-4  | Strong reasoning, citation quality |
| Code implementation | Claude/Gemini | Code execution, debugging          |
| Data pipeline       | Gemini/GPT-4  | API access, structured output      |
| Documentation       | Any           | All perform well                   |

---

## Output Format

After running this research, the agent should produce:

```
## Research Findings Summary

### Data Sources Identified
- [List of available datasets with URLs/APIs]

### Labeling Implementation Plan
- [Specific parameters, code snippets]

### Benchmark Literature
- [Published results to compare against]

### Tasks Discovered
- [Concrete work items that need doing]
- [Dependencies between tasks]
- [Estimated effort per task]
```

This output becomes the input for Linear task creation.
