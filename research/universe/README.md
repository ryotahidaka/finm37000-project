# Cross-commodity study code

Builds everything behind [docs/STUDY.md](../../docs/STUDY.md) from the Databento cache. Nothing here downloads data except `pull_universe.py`, which is guarded (see below).

| File | What it does |
| --- | --- |
| `core.py` | Contract specs for CME and ICE products, loading bars and settlements (CME and ICE symbol formats), open-interest or volume rolls limited to nearby contracts, a calendar-roll fallback for ICE without open interest, the PnL engine, summary stats, Hurst, deflated Sharpe |
| `spread_strategies.py` | Fixed-ratio structures and rolling-beta pairs, z-score signals, walk-forward selection, yearly formation/trading screens, PCA residual |
| `run_universe.py` | Every structure (calendars, substitutes, controls) on a 3x3 grid plus walk-forward, screens and PCA by universe, risk-parity books, deflated Sharpe |
| `lab3.py` | Every pair and calendar with four strategy families, two cost models (per leg, or one order) and yearly out-of-sample strategy choice; relationship map; single-product controls |
| `regimes.py` | Recession and stress windows, ex-ante volatility regime, recession shading for charts (idea from Harry Posner's `recession_bars.py`, MIT) |
| `study.py` | Runs all of the above and writes the tables, figures and `summary.json` in `docs/study/` |
| `build_universe_report.py`, `build_lab3.py` | Optional single-page HTML version of the results |
| `pull_universe.py` | Databento pulls for extra products, with hard guardrails |

Run (cmd, from the repo root):

```
set PYTHONPATH=src
python research\universe\study.py --cache data\databento --runs output\study --docs docs\study
```

Data pulls. `pull_universe.py` refuses to download unless `data/databento/budget.json` holds a budget, the estimate printed in the same run fits both the budget and `--max-cost` (default $0), the user types the exact dollar figure, and no other pull holds the lock file. Every request is logged to `data/databento/spend_ledger.csv` and the run stops at 80% of the budget. Start with one product, one schema and a recent year, and look at the real cost first.
