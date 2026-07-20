# Financial Market Analysis — Double-Tower Transformer for FX Signal Prediction

![architecture](docs/architecture.png)

Predictive ML system generating trading signals from 5 years of 1-minute EUR/USD data. Flagship model is a custom **Gated Transformer Network (GTN)** with a dual-tower architecture (features × time), benchmarked against 5 other approaches, with a live MetaTrader 5 execution pipeline.

## Results

| Model | Composite Score | Notes |
|---|---|---|
| LightGBM (baseline) | 0.37 | Best single run, but stochastic/not reproducible (typical: 0.20–0.25) |
| **GTN (dual-tower Transformer)** | 0.36 | Learned time embeddings > sin/cos encoding |
| CatBoost + Sharpe optimization | — | Optimizes Sharpe directly instead of accuracy |
| Regression (pips/R:R/prob) | — | Multi-stage, volatility-adaptive |
| Ensemble + regime detection | — | Regime-specific TP/SL |
| Direct P&L neural net | — | Custom P&L loss, Kelly sizing |

*Composite Score = (buy_precision + sell_precision)/2 − 0.25·|buy_precision − sell_precision|*

**Core open problem:** a high composite score doesn't reliably translate to live MT5 P&L — every experiment above attacks this from a different angle.

## Pipeline

```
1-min OHLC (5yr) → TP/SL signal generation (24h lookahead) → feature engineering
  → model training → MT5 backtest (0.1 lot, R:R 1:2/1:3)
```

- Signal logic evolved from fixed-horizon labels ("+10 pips in N hours") to **TP/SL-hit-first labels**, far more robust to market structure.
- Entry price corrected to next-candle open (removes lookahead bias).
- Multi-timeframe features (5m–4h), ATR/volatility regime, session-overlap timing, rolling S/R levels.

## Engineering gotchas (the expensive lessons)

- **MT5 requires UTF-16LE** for signal CSVs — UTF-8 silently fails to load.
- **`sklearn.LabelEncoder` sorts alphabetically** (`buy, keep, sell` → `0,1,2`) — must be explicitly remapped to `(1, 0, -1)` before MQ5 sees it.
- Sequence construction is **unnecessary for tree models** (CatBoost/LGBM) — building it anyway cost hours of runtime for zero benefit.
- Models default to over-predicting "keep" (majority class) — fixed with a custom cost matrix + weighted cross-entropy, which beat focal loss and undersampling.

## Repo contents

Core signal-generation pipeline, the flagship model, and evaluation code — not the full experiment history (30+ model variants live in a private research log; only the representative/best ones are here).

- `data_pipeline/` — canonical TP/SL signal generation + raw data prep
- `model/` — GTN dual-tower Transformer (best deep-learning result)
- `backtest_eval/` — composite score / performance metrics used to score every model above

## Next steps

- Ensemble the top 2–3 models
- Walk-forward validation (rolling windows) instead of a single train/test split
- Incorporate economic calendar / sentiment data
- Full RL trading agent (long-term)
