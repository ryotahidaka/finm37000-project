"""Shared pieces of the metals strategy lab: extra contract specs, data loading, statistics.

Data: data/databento cache (ohlcv-1d per contract) for GC SI PL PA HG, plus optional
<root>_settle_<year>.parquet from the statistics schema. Settlement replaces the last-trade
close when present.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gcsi import contracts, rolls
from gcsi.contracts import ContractSpec
from gcsi.costs import CostModel, LegCost

# ---- extra contract specs (CME/NYMEX/COMEX contract specs; fees assumed like GC) --------
EXTRA = {
    "PL": ContractSpec("PL", "NYMEX Platinum", 50.0, 0.10, (1, 4, 7, 10), "platinum"),
    "PA": ContractSpec("PA", "NYMEX Palladium", 100.0, 0.05, (3, 6, 9, 12), "palladium"),
    "HG": ContractSpec("HG", "COMEX Copper", 25000.0, 0.0005, (3, 5, 7, 9, 12), "copper"),
}
for k, v in EXTRA.items():
    contracts.SPECS.setdefault(k, v)

ROOTS = ["GC", "SI", "PL", "PA", "HG"]
COSTS = CostModel({**CostModel().legs, **{r: LegCost(0.5, 1.60, 0.85) for r in EXTRA}})
CAP = 1_000_000.0


def available_roots(cache: str) -> list[str]:
    return [r for r in ROOTS if glob.glob(os.path.join(cache, f"{r}_ohlcv-1d_*.parquet"))]


def load_bars(cache: str, root: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(cache, f"{root}_ohlcv-1d_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    pat = re.compile(rf"{root}[FGHJKMNQUVXZ]\d$")
    df = df[df["symbol"].astype(str).map(lambda s: bool(pat.match(s)))]
    df = df[(df["close"] > 0) & (df["ts_event"].dt.dayofweek < 5)]
    df = df.drop_duplicates(["ts_event", "instrument_id"], keep="last")
    settle = load_settle(cache, root)
    if settle is not None:
        s = settle[settle["stat"] == 3][["trade_date", "instrument_id", "value"]]
        s = s.rename(columns={"trade_date": "ts_event", "value": "settle"})
        # a trading day's settlement can arrive in the next year's file too: keep one
        s = s.drop_duplicates(["ts_event", "instrument_id"], keep="last")
        df = df.merge(s, on=["ts_event", "instrument_id"], how="left")
        df["last_trade"] = df["close"]
        # settlement only: mixing settles with last trades on the days a settle is missing
        # creates artificial jumps between two different price definitions
        df = df[df["settle"].notna()].copy()
        df["close"] = df["settle"]
    return df.sort_values(["ts_event", "instrument_id"]).reset_index(drop=True)


def load_settle(cache: str, root: str):
    files = sorted(glob.glob(os.path.join(cache, f"{root}_settle_*.parquet")))
    if not files:
        return None
    s = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    s["trade_date"] = pd.to_datetime(s["trade_date"], utc=True).dt.normalize()
    return s


@dataclass
class Leg:
    root: str
    bars: pd.DataFrame
    cont: pd.DataFrame        # rolls.build_continuous output (close = ratio adjusted, raw_close)
    logs: pd.DataFrame


def build_leg(cache: str, root: str) -> Leg:
    b = load_bars(cache, root)
    sched = rolls.roll_schedule(b, root, rule="fnd", days_before_fnd=5)
    cont = rolls.build_continuous(b, root, adjustment="ratio", schedule=sched)
    return Leg(root, b, cont, rolls.roll_log(b, sched))


# ---- statistics ---------------------------------------------------------------------------
def deflated_sharpe(daily: pd.Series, trial_sharpes: list[float]) -> dict:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio, on daily returns.

    trial_sharpes: annualized Sharpe of every configuration tried (the multiple-testing set).
    Returns the annualized Sharpe, the expected max Sharpe under the null (SR0), and the
    probability that the true Sharpe is above zero after accounting for the trials.
    """
    from scipy.stats import norm, skew, kurtosis
    r = daily.dropna()
    # keep flat days: the Sharpe being tested is the calendar-time one
    T = len(r)
    sr = r.mean() / r.std() if r.std() > 0 else 0.0
    trials = np.asarray([s for s in trial_sharpes if np.isfinite(s)]) / np.sqrt(252)
    n = max(len(trials), 1)
    g = 0.5772156649
    v = trials.var(ddof=1) if n > 1 else 0.0
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e))) if n > 1 else 0.0
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / den)
    return {"sharpe_ann": sr * np.sqrt(252), "sr0_ann": sr0 * np.sqrt(252), "dsr": float(dsr),
            "n_trials": int(n), "days": int(T)}


def hurst(x: np.ndarray) -> float:
    """Hurst exponent from the variance of lagged differences (lags 2..20). < 0.5 = mean-reverting."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 40:
        return np.nan
    lags = np.arange(2, 21)
    tau = [np.std(x[l:] - x[:-l]) for l in lags]
    if min(tau) <= 0:
        return np.nan
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def summarize(net: pd.Series) -> dict:
    r = net / CAP
    sd = r.std()
    eq = CAP + net.cumsum()
    peak = np.maximum.accumulate(np.r_[CAP, eq.to_numpy()])[1:]
    return {"sharpe": float(np.sqrt(252) * r.mean() / sd) if sd > 0 else np.nan,
            "ann_ret": float(r.mean() * 252), "ann_vol": float(sd * np.sqrt(252)),
            "mdd": float((eq.to_numpy() / peak - 1).min())}
