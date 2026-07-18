# Stock Predict

A multi-factor quantitative trading system for U.S. equities, integrated with the Alpaca paper-trading API.

The system trains an ensemble of factor models (momentum, trend, volatility, volume, market, alpha), generates daily signals, executes trades through Alpaca, and runs an intraday risk monitor with automatic stop-loss / take-profit / end-of-day flat.

> 中文使用文档见 [`docs/zh/`](docs/zh/) — 详细命令和流程说明。

---

## Table of contents

- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Daily workflow](#daily-workflow)
- [Operational scripts](#operational-scripts)
- [Risk parameters](#risk-parameters)
- [Documentation](#documentation)

---

## Architecture

```
                    ┌────────────────────────┐
                    │  research/  (offline)  │
                    │  data → features →     │
                    │  factors → train →     │
                    │  artifacts (.pkl)      │
                    └───────────┬────────────┘
                                │  signals_multi_factor_*.json
                                ▼
┌──────────────────────────────────────────────────────────┐
│  Daily pipeline (root)                                   │
│                                                          │
│   fetch_ohlcv → prepare_prediction_data →                │
│   get_daily_signals_multi_factor → alpaca_trader →       │
│   monitor_dynamic_trading (intraday loop)                │
└──────────────────────────────────────────────────────────┘
                                │
                                ▼
            trading_logs/  paper_trading_reports/  alerts/
```

Two distinct layers:

| Layer | Folder | Cadence | Purpose |
|---|---|---|---|
| **Research** | [research/](research/) | weekly retrain | Feature engineering, model training, backtests |
| **Production** | repo root | daily | Signal generation, order execution, risk monitor |

---

## Project layout

```
stock_predict/
├── README.md                           ← you are here
├── pyproject.toml                      ← project metadata, tooling config
├── requirements.txt                    ← runtime dependencies (pinned)
├── requirements-dev.txt                ← dev/test/lint dependencies
├── .gitignore
│
├── config_alpaca.py                    ← Alpaca credentials (gitignored)
├── config_finnhub.py                   ← Finnhub credentials (gitignored)
│
├── run_menu.py                         ← interactive CLI entry point
├── run_auto_trading.py                 ← end-to-end daily trading pipeline
├── run_retrain_models.py               ← weekly model retraining
│
├── alpaca_trader.py                    ← order placement
├── monitor_dynamic_trading.py          ← intraday stop-loss / take-profit / EOD
├── monitor_paper_trading.py            ← performance reporting
├── alert_system.py                     ← alerting (log/email)
├── dashboard.py                        ← Streamlit live dashboard
│
├── analyze_full_performance.py         ← post-hoc performance analysis
├── analyze_losses.py                   ← loss attribution
├── factor_attribution.py               ← per-factor contribution
├── market_regime.py                    ← regime classifier (bull/bear)
├── data_quality_monitor.py             ← data sanity checks
├── optimize_parameters.py              ← stop-loss / take-profit optimizer
├── finbert_sentiment.py                ← FinBERT news sentiment
│
├── check_account.py                    ← account snapshot
├── check_monitor_status.py             ← scheduled-task health check
├── check_and_cancel_orders.py          ← cancel pending orders
├── log_comprehensive_trading.py        ← structured trade logging
├── view_signals.py                     ← inspect today's signals
├── view_news_sentiment.py              ← inspect news sentiment
├── view_win_rates.py                   ← per-stock win-rate breakdown
│
├── test_alpaca_connection.py           ← smoke test: broker
├── test_finnhub.py                     ← smoke test: news
│
├── *.bat                               ← Windows launchers
│
├── research/                           ← offline modelling
│   ├── data/                           ← raw OHLCV / news / IV fetchers
│   ├── features/                       ← feature engineering
│   ├── factors/                        ← factor definitions
│   ├── train/                          ← model training scripts
│   ├── backtest/                       ← backtest engine + walk-forward
│   ├── artifacts/                      ← trained models (.pkl)
│   ├── config.yaml
│   └── get_daily_signals_multi_factor.py
│
├── docs/
│   ├── zh/                             ← Chinese usage guides
│   └── research/                       ← research-side guides (English)
│
├── trading_logs/                       ← runtime artefacts (gitignored)
├── paper_trading_reports/              ← runtime artefacts (gitignored)
├── alerts/                             ← runtime artefacts (gitignored)
└── news_cache/                         ← runtime cache (gitignored)
```

---

## Requirements

- **Python 3.10+** (developed against 3.12)
- Alpaca account (paper or live) with API key + secret
- Finnhub account (free tier works) for news sentiment
- Windows is the developed-on platform (scripts use `.bat`); the Python code is cross-platform

---

## Installation

```bash
# 1. Clone and enter the project
git clone <repo-url> stock_predict
cd stock_predict

# 2. Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install runtime dependencies
pip install -r requirements.txt

# 4. (optional) Install dev dependencies for tests / linting
pip install -r requirements-dev.txt
```

---

## Configuration

Create the two config files at the project root (they are gitignored):

```python
# config_alpaca.py
ALPACA_API_KEY    = "PK..."
ALPACA_SECRET_KEY = "..."
ALPACA_PAPER      = True            # False = live trading
ALPACA_BASE_URL   = (
    "https://paper-api.alpaca.markets"
    if ALPACA_PAPER else
    "https://api.alpaca.markets"
)
```

```python
# config_finnhub.py
FINNHUB_API_KEY          = "..."
NEWS_LOOKBACK_HOURS      = 24
SENTIMENT_THRESHOLD      = 0.5
MIN_NEWS_COUNT           = 3
CACHE_DURATION_MINUTES   = 30
```

> ⚠️ Keep these files out of version control. They are listed in `.gitignore` for both the project root and `research/`.

---

## Daily workflow

| When | Command | Notes |
|---|---|---|
| 21:00 ET | `python run_auto_trading.py` | Fetch data → generate signals → flatten → enter new positions |
| Intraday (every 5 min) | `python monitor_dynamic_trading.py --auto` | Stop-loss / take-profit / EOD flat. Run from Windows Task Scheduler. |
| 16:00 ET | (automatic) | EOD flatten triggered by the monitor |
| Weekly (Sunday) | `python run_retrain_models.py` | Refit all factor models on the latest data |

The interactive menu wraps all of this:

```bash
python run_menu.py
```

---

## Operational scripts

```bash
python check_account.py              # current portfolio snapshot
python check_monitor_status.py       # scheduled-task / process / log health
python check_and_cancel_orders.py    # cancel any open orders
python view_signals.py               # today's recommended trades
python view_news_sentiment.py        # latest news sentiment
python view_win_rates.py             # per-stock historical win rate
python analyze_losses.py             # diagnostic on losing trades
python factor_attribution.py         # which factors paid off
python data_quality_monitor.py       # data freshness / completeness
python optimize_parameters.py        # search for optimal stop / take levels
streamlit run dashboard.py           # live Streamlit dashboard
```

---

## Risk parameters

Defined in [monitor_dynamic_trading.py](monitor_dynamic_trading.py):

| Parameter | Default | Meaning |
|---|---|---|
| `STOP_LOSS_PCT` | 0.025 | Per-position stop-loss (-2.5%) |
| `TAKE_PROFIT_PCT` | 0.025 | Per-position take-profit (+2.5%) |
| `MAX_DAILY_LOSS_PCT` | 0.03 | Account-wide intraday loss kill switch (-3%) |
| `EOD_CLOSE_TIME` | 16:00 ET | Force-flatten time (DST-aware via `zoneinfo`) |

Position sizing in [alpaca_trader.py](alpaca_trader.py):

| Parameter | Default |
|---|---|
| `max_position_pct` | 0.15 (15% of equity per name) |

---

## Documentation

- **English**
  - [docs/research/QUICK_START.md](docs/research/QUICK_START.md) — research pipeline quick-start
  - [docs/research/EXTERNAL_DATA_GUIDE.md](docs/research/EXTERNAL_DATA_GUIDE.md) — adding IV / news data
  - [docs/research/FINAL_OPTIMIZATION_REPORT.md](docs/research/FINAL_OPTIMIZATION_REPORT.md)
  - [docs/research/STAGE1_OPTIMIZATION_GUIDE.md](docs/research/STAGE1_OPTIMIZATION_GUIDE.md)
  - [docs/research/OPTIMIZE_TO_62_PERCENT.md](docs/research/OPTIMIZE_TO_62_PERCENT.md)
  - [docs/research/GDELT_IMPLEMENTATION_GUIDE.md](docs/research/GDELT_IMPLEMENTATION_GUIDE.md)
  - [docs/research/TRAIN_GPU.md](docs/research/TRAIN_GPU.md)

- **中文**
  - [docs/zh/usage-guide.zh.txt](docs/zh/usage-guide.zh.txt) — 完整使用流程
  - [docs/zh/command-reference.zh.txt](docs/zh/command-reference.zh.txt) — 命令速查
  - [docs/zh/how-to-run.zh.txt](docs/zh/how-to-run.zh.txt) — 如何运行批处理

---

## Disclaimer

This software is provided for **educational and research purposes only**. It is not financial advice. Trading involves substantial risk of loss. Run in paper-trading mode (`ALPACA_PAPER = True`) until you fully understand the system's behaviour.
