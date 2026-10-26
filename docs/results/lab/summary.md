# Metals strategy lab: results summary

Data: Databento GLBX.MDP3 daily bars and official settlement prices (statistics schema, stat 3) for gold (GC), silver (SI), platinum (PL), palladium (PA) and copper (HG). Scored on 2014 to 2026 net PnL on $1m capital, fills one day after the signal, half a tick plus exchange fee plus commission per leg, legs capped at $2m notional.

191 configurations in total: 182 run in the lab plus the 9 earlier GC/SI runs, all counted for the multiple-testing correction.

| Family | Configs | Median Sharpe | Best Sharpe |
| --- | --- | --- | --- |
| PCA residual (s-score, OU) | 2 | 0.43 | 0.54 |
| Johansen 3-metal basket | 1 | 0.23 | 0.23 |
| Trend and carry (spread momentum, cross-sectional carry and momentum) | 7 | 0.07 | 0.29 |
| Pairs, formation/trading with Engle-Granger selection | 4 | -0.11 | 0.05 |
| Curve grid (calendars, butterflies, carry vs carry) | 144 | -0.52 | 0.53 |
| Curve walk-forward | 24 | -0.59 | 0.31 |

Leaders

| Configuration | Sharpe | Return a year | Max drawdown |
| --- | --- | --- | --- |
| PCA residual s-score, OU window 30d | 0.54 | 5.3% | -23.3% |
| GC calendar, spread-book costs, 60d, entry 2.0 | 0.53 | 0.25% | -0.9% |
| GC calendar, spread-book costs, walk-forward | 0.31 | 0.15% | -1.6% |
| PCA residual s-score, OU window 60d | 0.32 | 3.1% | -23.1% |
| Cross-sectional momentum 12-1m, 2 per side | 0.29 | 2.5% | -24.4% |
| Johansen 3-metal basket | 0.23 | 1.1% | -9.4% |

Multiple testing. The deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) over all 191 trials puts the Sharpe you would expect from the best of 191 lucky draws at 2.02, so nothing passes. That benchmark is driven by how widely the 144 curve-grid configurations spread. Judged only within its own family, the PCA residual (2 tries) has a 97% probability of a positive true Sharpe, the GC calendar (144 tries) 0%, and cross-sectional momentum (7 tries) 65%.

Stability. PCA residual 30d by period: Sharpe -0.44 in 2014-2017, 0.35 in 2018-2021, 1.12 in 2022-2026. Most of the gain is recent.

Files: all_trials.csv (every configuration), leaders.csv (top 10 with deflated Sharpe), curve_wf.csv (walk-forward curve structures, both cost modes). Regenerate with research/run_all.py.
