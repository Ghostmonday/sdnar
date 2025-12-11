# Evidence Locker

This directory contains hash-stamped artifacts for audit purposes.

## Artifact Naming Convention

```
[DATE]_[TYPE]_[DESCRIPTION]_[HASH].ext

Examples:
2024-12-10_DATA_equities_labeling_a1b2c3.csv
2024-12-10_PLOT_benchmark_comparison_d4e5f6.png
2024-12-10_REPORT_phase1_summary_g7h8i9.pdf
```

## Hash Verification

All artifacts include SHA-256 checksums recorded in the manifest below.

## Manifest

| Date       | Artifact                         | Type     | SHA-256 (first 8)                  |
| ---------- | -------------------------------- | -------- | ---------------------------------- |
| 2025-12-10 | spy_labeled.csv                  | DATA     | bff6b9d5 (full in labeled/manifest)|
| 2025-12-10 | qqq_labeled.csv                  | DATA     | f8b2235f (full in labeled/manifest)|
| 2025-12-10 | gld_labeled.csv                  | DATA     | 47af2db2 (full in labeled/manifest)|
| 2025-12-10 | btc_labeled.csv                  | DATA     | 410de677 (full in labeled/manifest)|
| 2025-12-10 | regime_analysis.json             | ANALYSIS | See 01_data/regimes/               |
| 2025-12-10 | baseline_results.csv             | METRICS  | See 02_benchmarks/                 |
| 2025-12-10 | deep_research_full.md            | REPORT   | Present in this folder             |
| 2025-12-10 | data_engineering_gemini_report.md| REPORT   | Present in this folder             |
| 2025-12-10 | labeling_gemini_report.md        | REPORT   | Present in this folder             |
| 2025-12-10 | order5_research_synthesis.md     | REPORT   | Present in this folder             |

## Hash Sources

- **Labeled Data**: Full SHA-256 hashes in `01_data/labeled/manifest.json`
- **Config Hash**: `14bdc7cfdf9fc70b8dd2ef9e4e6376fa7efb4908afcd52c2da0eef0dbeee6177`

---

_Phase I artifacts logged. Config: volatility_span=20, barrier_width=2.0, vertical_barrier=10_
