# Research lab

Exploration code that sits on top of the `gcsi` package. It asks a wider question than the main pipeline: across the CME metals complex, which kind of relative-value trade (if any) survives costs and a multiple-testing correction? The GC/SI pipeline in `src/gcsi` stays the tested core; this folder reuses its contract specs, rolls, cost model and ledger.

## Files

| File | What it does |
| --- | --- |
| `lab.py` | Extra contract specs (PL, PA, HG), data loading from the Databento cache with settlement prices when present, continuous series per root, deflated Sharpe, Hurst exponent, summary stats |
| `strategies.py` | Strategy families and a shared dollar-target simulator with per-leg costs and roll charges |
| `lab_curve.py` | Curve structures on each metal: calendars, butterflies, carry vs carry, a fixed grid plus walk-forward, per-leg or spread-book costs |
| `run_all.py` | Runs every family, writes the trial tables and the deflated Sharpe for the leaders |
| `build_report.py` | Builds a single self-contained HTML report from the output folder |

## Strategy families

- Pairs, formation and trading (Gatev, Goetzmann and Rouwenhorst style): each year, pick the 3 most cointegrated metal pairs on the previous 3 years, freeze the hedge, trade the year. Optional Hurst gate that only trades while the spread looks mean-reverting.
- PCA residual stat-arb (Avellaneda and Lee): regress each metal's returns on the first principal component of the group, model the cumulative residual as OU and trade its s-score with the paper's thresholds (open 1.25, close 0.5), not tuned here.
- Johansen basket: each year, the 3-metal basket with the strongest Johansen trace statistic on the previous 3 years, weighted by its first cointegrating vector and traded on a z-score.
- Trend and carry: spread momentum on every pair, cross-sectional carry (annualised roll yield, read from the roll gaps of each continuous series) and 12-1 month cross-sectional momentum, long the top and short the bottom.
- Curve structures: calendars, butterflies and carry vs carry, same-contract z-score, grid fixed before the run, then walk-forward on the previous 3 years.

## Running it (cmd)

```
set PYTHONPATH=src
python research\run_all.py --cache data\databento --out output\lab
python research\build_report.py --out output\lab
```

It uses whatever metals are in the cache. Settlement files (`<ROOT>_settle_<year>.parquet`, from the `statistics` schema) replace the last-trade close when present. A full run takes about 13 minutes.

## Rules every configuration follows

- Signal on day t uses data up to t, the fill is at t+1's price.
- Half a tick plus exchange fee plus commission per contract per side, rolls charged as two trades.
- Leg notional capped at $2m so very quiet spreads cannot be levered without limit.
- Every configuration tried is kept in `all_trials.csv` and counted in the deflated Sharpe ratio, including the earlier GC/SI runs.

Results are summarised in [docs/results/lab/summary.md](../docs/results/lab/summary.md).
