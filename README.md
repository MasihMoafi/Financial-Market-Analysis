---
name: gated-transformer-fx-signals
type: experimental ML research pipeline for EUR/USD signal prediction and backtest evaluation
---

# Gated Transformer FX Signals

![architecture](docs/architecture.png)

**A classifier score is not the same thing as a profitable trading system.**

This repository contains my experimental EUR/USD signal-prediction pipeline built around a dual-tower Gated Transformer Network (GTN), plus data preparation and backtest/evaluation code. The strongest deep-learning run recorded here reached a composite score of **0.36**; a LightGBM baseline reached **0.37** in its best run, but that result was not stable across repeated runs.

The unresolved problem is the important one: offline classification performance has not yet been shown to translate reliably into live trading P&L.

## Quick start

The repository does **not** include the full raw market dataset, so there is no honest one-command reproduction from a fresh clone.

Clone the code:

```bash
git clone https://github.com/MasihMoafi/gated-transformer-fx-signals.git
cd gated-transformer-fx-signals
```

Install the Python dependencies used by the GTN script, including a PyTorch build appropriate for your hardware:

```bash
python -m pip install numpy pandas scikit-learn scipy matplotlib seaborn packaging joblib
```

Then run the GTN against a compatible time-ordered CSV containing the expected OHLC/feature columns and a `signal` target:

```bash
python model/gtn_dual_tower.py \
  --train_file /path/to/eurusd_signals.csv \
  --output_dir output_gtn
```

Expected result: the script temporally splits the supplied data, trains the GTN, logs validation/test metrics, and writes run artifacts under the output directory.

A clean-room reproducibility package with a redistributable dataset or download script is still missing.

## The problem

A market classifier can look good while still being useless after execution costs, class imbalance, regime changes, entry timing, and TP/SL behavior are accounted for.

This project therefore evaluates more than accuracy. The experiments moved from fixed-horizon labels toward TP/SL-hit-first labels, corrected entry timing to the next-candle open, and compared several modeling approaches against a metric that penalizes one-sided buy/sell precision.

## Results currently recorded

| Model | Composite score | What the repository can honestly claim |
| --- | ---: | --- |
| LightGBM baseline | **0.37** best run | Best single recorded score, but stochastic; typical runs were reported around 0.20–0.25. |
| **GTN dual-tower Transformer** | **0.36** | Best recorded deep-learning result in this repository. |
| CatBoost + Sharpe optimization | — | Experimental path; no comparable final composite score recorded in the README. |
| Regression: pips / R:R / probability | — | Experimental path. |
| Ensemble + regime detection | — | Experimental path. |
| Direct P&L neural net | — | Experimental path. |

Composite score:

```text
(buy_precision + sell_precision) / 2
- 0.25 * abs(buy_precision - sell_precision)
```

These numbers are experiment results, not evidence of expected future returns.

## How it works

```text
1-minute EUR/USD OHLC
        ↓
TP/SL-hit-first signal generation
        ↓
feature engineering + temporal split
        ↓
model training
        ↓
classification / composite-score evaluation
        ↓
MT5-oriented backtest and execution analysis
```

Key implementation decisions represented in the code:

- time-ordered train/test splitting rather than random market-data shuffling;
- entry-price correction to the next candle open to avoid using information unavailable at decision time;
- multi-timeframe and volatility/session features;
- class-sensitive loss/regularization to reduce majority-class collapse;
- deterministic PyTorch settings where supported;
- separate evaluation utilities under `backtest_eval/`.

## Current state

### Implemented

- Canonical data-preparation scripts under `data_pipeline/`.
- Dual-tower GTN training implementation in [`model/gtn_dual_tower.py`](model/gtn_dual_tower.py).
- Performance-analysis code under `backtest_eval/`.
- Temporal splitting, feature engineering, class-sensitive training, and output artifact generation in the GTN path.
- Architecture diagram and the representative experiment results above.

### Implemented but still under research acceptance

- The GTN and baseline comparisons are experimental rather than production trading validation.
- The best LightGBM score is explicitly not stable across repeated runs.
- The mapping from offline composite score to MT5 P&L remains unresolved.
- The repository contains representative experiments, not the full private experiment history.

### Planned

- Walk-forward validation instead of relying on one train/test period.
- Ensemble evaluation of the strongest models.
- Economic-calendar / sentiment features if they improve controlled evaluations.
- More direct optimization against trading outcomes only after the validation protocol is strong enough to detect overfitting.

### Intentionally unsupported / not claimed

- No claim of profitable live trading.
- No claim that the reported scores generalize to future market regimes.
- No complete redistributable dataset is bundled with the repository.
- No production-grade automated trading release is documented here.

## Engineering lessons already encoded

- **MT5 CSV encoding matters:** the integration path expects UTF-16LE rather than assuming UTF-8.
- **Label ordering matters:** `sklearn.LabelEncoder` alphabetic ordering must be remapped before downstream trading logic interprets class values.
- **Tree models do not need sequence construction:** building sequences for LightGBM/CatBoost adds runtime without giving those models temporal sequence semantics.
- **Majority-class collapse is easy:** the experiments required explicit class/cost treatment rather than trusting raw accuracy.

## What sets this project apart

The useful distinction is not that the GTN is automatically better than standard baselines—it currently is not.

The project is valuable because the repository records the harder failure: **a model can improve the ML metric without proving that it improves the trading objective.** The code and roadmap are organized around closing that gap instead of hiding it behind one attractive accuracy number.

## Evals and test series

Current evaluation evidence lives in the training scripts and `backtest_eval/` utilities rather than a unified CI benchmark.

What the current results support:

- comparison of representative model runs under the repository's composite metric;
- inspection of buy/sell precision balance;
- temporal train/test evaluation in the GTN path;
- downstream performance analysis from generated signals.

What they do not support:

- statistically robust future-return estimates;
- stability across many walk-forward windows;
- production execution reliability;
- a claim that GTN beats LightGBM or CatBoost in general.

The smallest useful next eval is the planned walk-forward test with fixed data/label logic and repeated baseline runs.

## Repository map

- `data_pipeline/` — market-data preparation and canonical signal generation.
- `model/gtn_dual_tower.py` — flagship GTN training path.
- `backtest_eval/` — performance metrics and analysis utilities.
- `docs/architecture.png` — model/pipeline architecture overview.

## Future development

Do not add another model just to increase the model count. The next useful work is stronger validation: walk-forward windows, repeated baselines, and an evaluation that connects offline scores to realized backtest outcomes without leakage.
