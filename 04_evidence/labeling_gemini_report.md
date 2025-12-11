# S-DNA Phase II: Triple-Barrier Labeling Report (Gemini Track)

_Generated: 2024-12-10_
_Agent: Gemini_
_Status: COMPLETE_

---

## Executive Summary

Implementation of Triple-Barrier Method (TBM) labeling pipeline with:

- EWM volatility estimator (span=20)
- Dynamic barriers at ±2σ
- Vertical barrier at 10 days
- SHA-256 reproducibility manifests

---

## Critical Finding: Expected Label Distribution

| Asset | Label -1 (Stop) | Label 0 (Neutral) | Label 1 (Profit) |
| ----- | --------------- | ----------------- | ---------------- |
| SPY   | ~10%            | **~80%**          | ~10%             |
| BTC   | ~15%            | **~65%**          | ~20%             |

### Gemini's Reasoning:

> "SPY is highly mean-reverting. Prices often stay within the ±2σ bands for 10 days. The high prevalence of '0' labels indicates that a Meta-Labeling approach is necessary."

> "The dataset is heavily imbalanced towards Class 0. Standard classifiers trained on this data will default to predicting '0' to maximize accuracy."

---

## Theoretical Framework

### Why Fixed-Time Horizons Fail:

1. **Ignores path** — A position may suffer catastrophic drawdown then recover
2. **Ignores heteroscedasticity** — 1% in low-vol is signal; 1% in high-vol is noise

### Triple-Barrier Geometry:

- **Upper Barrier**: $B_{upper} = P_{t0} \times (1 + 2 \times \sigma_t)$
- **Lower Barrier**: $B_{lower} = P_{t0} \times (1 - 2 \times \sigma_t)$
- **Vertical Barrier**: $t_0 + 10$ days

### First-Touch Logic:

```
τ = min{t : P_t ≥ B_upper OR P_t ≤ B_lower} ∪ {t_0 + T}

Label =
  1  if P_τ ≥ B_upper (Profit Take)
  -1 if P_τ ≤ B_lower (Stop Loss)
  0  if τ = t_0 + T (Time Expiration)
```

---

## Data Quality Audit

### SPY:

- Adjusted Close imperative (dividends/splits)
- "10 days" = 10 trading bars (not calendar days)
- Mean-reverting: expects ~80% neutral labels

### BTC:

- 24/7 trading, missing rows = lost information
- Fat tails: ±2σ may be ±10% during volatile periods
- Momentum: expects asymmetric bull bias (~20% vs ~15%)

---

## Volatility Estimator Specification

```python
α = 2 / (span + 1) = 2/21 ≈ 0.0952

σ²_t = (1-α) × σ²_{t-1} + α × (r_t - r̄)²
```

### Expected Ranges:

- SPY: Daily σ from 0.006 to 0.02 (annualized 10-60%)
- BTC: Daily σ from 0.02 to 0.10 (extraordinarily high)

---

## Implementation Notes

### Class Imbalance Warning:

> "To make this data usable for S-DNA Phase II, downstream training must employ SMOTE or sample weighting (weighting Class 1 and -1 significantly higher than Class 0)."

### Reproducibility:

- SHA-256 hash of serialized DataFrame
- Enables audit trail and version control
- Detects retroactive data vendor updates

---

## Action Items

1. **Ingest**: Execute pipeline on raw CSVs
2. **Verify**: Check SHA-256 hashes against baseline
3. **Balance**: Apply sample weights to counteract Class 0 imbalance

---

_End of Report_
_Labeling Pipeline Agent - Gemini Track_
