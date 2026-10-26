"""Regime labels and recession shading for the cross-commodity study.

Recession shading follows the idea of Harry Posner's `recession_bars.py` gist (MIT licence,
https://gist.github.com/harryposner/ab51fd496e9e48ac39847cada10f6557): shade NBER business-cycle
recessions on a matplotlib time-series axis. This is our own short implementation limited to the
recessions inside our data window; the dates are the NBER peak and trough months.

NBER dates are announced months after the fact, so recession labels are only used to describe
what happened. They are never used as a trading signal. The volatility regime below is built
only from data available on each day, so it could be used live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# NBER peak and trough months inside 2001-2026 (trough method: the shading ends at the end of
# the trough month, like the FRED USREC series)
NBER = [("2001-03-01", "2001-11-01"), ("2007-12-01", "2009-06-01"), ("2020-02-01", "2020-04-01")]

STRESS = {  # commodity-specific stress windows we chose before looking at strategy results
    "2011 silver and gold blow-off": ("2011-08-01", "2011-12-31"),
    "2013 gold crash": ("2013-04-01", "2013-04-30"),
    "2022 energy and metals shock": ("2022-02-15", "2022-06-30"),
}


def _ts(x):
    return pd.Timestamp(x, tz="UTC")


def recession_windows():
    return [(_ts(p), _ts(t) + pd.DateOffset(months=1)) for p, t in NBER]


def add_recession_bars(ax, color="lightgrey", stress=True, stress_color="#F3D9B1"):
    """Shade NBER recessions (grey) and, optionally, our stress windows (tan) on `ax`."""
    lo, hi = ax.get_xlim()
    for a, b in recession_windows():
        ax.axvspan(a, b, color=color, zorder=0, lw=0)
    if stress:
        for a, b in STRESS.values():
            ax.axvspan(_ts(a), _ts(b), color=stress_color, alpha=0.6, zorder=0, lw=0)
    ax.set_xlim(lo, hi)
    return ax


def label_regimes(index: pd.DatetimeIndex, basket_returns: pd.DataFrame, q=0.8) -> pd.Series:
    """One label per day: NBER recession > named stress window > high volatility > calm.

    High volatility uses only past data: the trailing 63-day volatility of an equal-weight basket
    is compared with the expanding 80th percentile of its own history up to the day before.
    """
    lab = pd.Series("calm", index=index)
    bv = basket_returns.reindex(index).fillna(0).mean(axis=1).rolling(63).std().shift(1)
    thr = bv.expanding(252).quantile(q)
    lab[bv > thr] = "high volatility"
    for name, (a, b) in STRESS.items():
        lab[(index >= _ts(a)) & (index <= _ts(b))] = "stress: " + name
    for a, b in recession_windows():
        lab[(index >= a) & (index < b)] = "NBER recession"
    return lab


def by_regime(daily_pnl: pd.Series, labels: pd.Series, cap: float) -> pd.DataFrame:
    rows = []
    for g, x in daily_pnl.groupby(labels.reindex(daily_pnl.index)):
        sd = x.std()
        rows.append({"regime": g, "days": len(x), "ann_ret": x.mean() * 252 / cap,
                     "sharpe": x.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
                     "period_ret": x.sum() / cap})
    return pd.DataFrame(rows)
