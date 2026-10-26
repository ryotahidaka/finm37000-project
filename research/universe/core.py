"""Universe lab core: contract specs, loading (CME + ICE), open-interest rolls, PnL engine.

Works on any root pulled by pull_universe.py (or the metals cache). Nothing here needs the
`definition` schema: contract month and year are parsed from the symbol and the roll follows
open interest (volume when OI is missing), so the same code covers CME and ICE products.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CAP = 1_000_000.0
SETTLE_MIN = 0.95   # share of bar rows that need a settlement before settlements replace last trades
MONTHS = "FGHJKMNQUVXZ"


# --------------------------------------------------------------------------------------------
# Specs: usd_per_point = USD (or currency) value of a 1.0 move in the quoted price, per contract
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    name: str
    folder: str          # "" for GLBX (cache root), else "IFUS" / "IFEU" / "NDEX"
    per_point: float     # currency per 1.0 price unit per contract
    tick: float
    ccy: str = "USD"     # quote currency; non-USD legs are converted with the FX future
    fee: float = 2.45    # exchange + clearing + commission per contract per side (USD)
    group: str = ""
    label: str = ""


S = Spec
SPECS = {s.name: s for s in [
    # metals (already cached)
    S("GC", "", 100, 0.10, group="metals", label="Gold"),
    S("SI", "", 5000, 0.005, group="metals", label="Silver"),
    S("PL", "", 50, 0.10, group="metals", label="Platinum"),
    S("PA", "", 100, 0.05, group="metals", label="Palladium"),
    S("HG", "", 25000, 0.0005, group="metals", label="Copper"),
    # CME energy
    S("CL", "", 1000, 0.01, group="energy", label="WTI crude (NYMEX)"),
    S("RB", "", 42000, 0.0001, group="energy", label="RBOB gasoline"),
    S("HO", "", 42000, 0.0001, group="energy", label="ULSD heating oil"),
    S("NG", "", 10000, 0.001, group="energy", label="Henry Hub gas (physical)"),
    S("HH", "", 10000, 0.001, group="energy", label="Henry Hub gas (last-day financial)"),
    S("BZ", "", 1000, 0.01, group="energy", label="Brent (NYMEX)"),
    S("CU", "", 42000, 0.0001, group="energy", label="Chicago ethanol"),
    # CME ags (prices in cents/bu or cents/lb, meal in $/short ton)
    S("ZS", "", 50, 0.25, fee=2.95, group="grains", label="Soybeans"),
    S("ZM", "", 100, 0.10, fee=2.95, group="grains", label="Soybean meal"),
    S("ZL", "", 600, 0.01, fee=2.95, group="grains", label="Soybean oil"),
    S("ZC", "", 50, 0.25, fee=2.95, group="grains", label="Corn"),
    S("ZW", "", 50, 0.25, fee=2.95, group="grains", label="Chicago wheat"),
    S("KE", "", 50, 0.25, fee=2.95, group="grains", label="KC wheat"),
    # CME FX (USD per unit of foreign currency)
    S("6B", "", 62500, 0.0001, group="fx", label="GBP"),
    S("6E", "", 125000, 0.00005, group="fx", label="EUR"),
    S("6L", "", 100000, 0.00005, group="fx", label="BRL"),
    S("6C", "", 100000, 0.00005, group="fx", label="CAD"),
    # ICE Futures US
    S("CC", "IFUS", 10, 1.0, group="softs", label="NY cocoa ($/t)"),
    S("KC", "IFUS", 375, 0.05, group="softs", label="Arabica coffee (c/lb)"),
    S("SB", "IFUS", 1120, 0.01, group="softs", label="Raw sugar No.11 (c/lb)"),
    S("CT", "IFUS", 500, 0.01, group="softs", label="Cotton (c/lb)"),
    S("RS", "IFUS", 20, 0.10, ccy="CAD", group="grains", label="Canola (CAD/t)"),
    # ICE Futures Europe
    S("BRN", "IFEU", 1000, 0.01, group="energy", label="ICE Brent"),
    S("GO", "IFEU", 100, 0.25, group="energy", label="ICE gasoil ($/t)"),
    S("WTI", "IFEU", 1000, 0.01, group="energy", label="ICE WTI"),
    S("LCC", "IFEU", 10, 1.0, ccy="GBP", group="softs", label="London cocoa (GBP/t)"),
    S("RC", "IFEU", 10, 1.0, group="softs", label="Robusta coffee ($/t)"),
    S("WS", "IFEU", 50, 0.10, group="softs", label="White sugar ($/t)"),
    # ICE Endex
    S("TTF", "NDEX", 730, 0.005, ccy="EUR", group="energy", label="Dutch TTF gas (EUR/MWh)"),
]}
SCHED_DAYS = {"BRN": 50, "WTI": 40, "GO": 30, "CC": 35, "LCC": 35, "KC": 35, "RC": 35,
              "SB": 20, "WS": 45, "CT": 35, "RS": 35, "TTF": 10}
FX_FOR = {"GBP": "6B", "EUR": "6E", "CAD": "6C", "BRL": "6L"}


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
_CME = re.compile(r"^([A-Z0-9]{1,3})([FGHJKMNQUVXZ])(\d{1,2})$")
_ICE = re.compile(r"([FGHJKMNQUVXZ])(\d{2}|\d{4})!?$")


def parse_contract(sym: str, root_codes: tuple[str, ...], bar_year: int) -> int | None:
    """Expiry key year*12+month for an outright futures symbol, None for spreads/options."""
    s = str(sym)
    if any(ch in s for ch in "-:_ ") and "  " not in s and not s.endswith("!"):
        return None
    m = _CME.match(s)
    if m and m.group(1) in root_codes:
        mon, d = MONTHS.index(m.group(2)) + 1, int(m.group(3))
        if len(m.group(3)) == 1:
            y = bar_year - bar_year % 10 + d
            if y < bar_year - 1:
                y += 10
        else:
            y = 2000 + d
        return y * 12 + mon - 1
    if "-" in s or ":" in s:
        return None
    m2 = _ICE.search(s)
    if m2 and s.endswith("!") and s.split()[0] in root_codes:
        mon, y = MONTHS.index(m2.group(1)) + 1, int(m2.group(2)) % 100 + 2000
        return y * 12 + mon - 1
    return None


def _files(cache: str, spec: Spec, kind: str) -> list[str]:
    base = os.path.join(cache, spec.folder) if spec.folder else cache
    return sorted(glob.glob(os.path.join(base, f"{spec.name}_{kind}_*.parquet")))


def available(cache: str) -> list[str]:
    return [n for n, s in SPECS.items() if _files(cache, s, "ohlcv-1d")]


def load(cache: str, name: str, use_settle: bool = True) -> pd.DataFrame:
    """Long table: date, iid, symbol, ym, px, volume, oi (outright contracts only)."""
    spec = SPECS[name]
    bars = pd.concat([pd.read_parquet(f) for f in _files(cache, spec, "ohlcv-1d")], ignore_index=True)
    bars["date"] = pd.to_datetime(bars["ts_event"], utc=True).dt.normalize()
    bars = bars[bars["date"].dt.dayofweek < 5]
    codes = tuple(sorted({str(s).split()[0] if " " in str(s) else _CME.match(str(s)).group(1)
                          for s in bars["symbol"].unique()
                          if (" " in str(s)) or _CME.match(str(s))}))
    # the parent's own outright code is the most common one
    if codes:
        heads = bars["symbol"].astype(str).map(
            lambda s: s.split()[0] if " " in s else (_CME.match(s).group(1) if _CME.match(s) else None))
        root_code = heads.value_counts().index[0]
        codes = (root_code,)
    # symbols are reused every decade (GCZ6 = Dec 2016 and Dec 2026): key on instrument_id
    first = bars.sort_values("date").drop_duplicates("instrument_id")
    keys = {iid: parse_contract(sym, codes, d.year)
            for iid, sym, d in first[["instrument_id", "symbol", "date"]].itertuples(index=False)}
    bars["ym"] = bars["instrument_id"].map(keys)
    bars = bars[bars["ym"].notna() & (bars["close"] > 0)].rename(columns={"instrument_id": "iid"})
    bars = bars.drop_duplicates(["date", "iid"], keep="last")
    df = bars[["date", "iid", "symbol", "ym", "close", "volume"]].copy()
    df["px"] = df["close"]
    sf = _files(cache, spec, "settle")
    df["oi"] = np.nan
    if sf:
        st = pd.concat([pd.read_parquet(f) for f in sf], ignore_index=True)
        st["date"] = pd.to_datetime(st["trade_date"], utc=True).dt.normalize()
        st = st.drop_duplicates(["date", "instrument_id", "stat"], keep="last")
        w = st.pivot_table(index=["date", "instrument_id"], columns="stat", values="value", aggfunc="last")
        w = w.rename(columns={3: "settle", 9: "oi_s"}).reset_index().rename(columns={"instrument_id": "iid"})
        df = df.merge(w[[c for c in ["date", "iid", "settle", "oi_s"] if c in w.columns]],
                      on=["date", "iid"], how="left")
        if "oi_s" in df:
            df["oi"] = df["oi_s"]
        # settlement only when it covers (almost) the whole history; a partial settle history
        # mixed with last trades would create jumps between two price definitions
        if use_settle and "settle" in df and df["settle"].notna().mean() > SETTLE_MIN:
            df = df[df["settle"].notna() & (df["settle"] > 0)].copy()
            df["px"] = df["settle"]
            df.attrs["price_source"] = "settlement"
    df["ym"] = df["ym"].astype(int)
    src = df.attrs.get("price_source", "last trade")
    out = df[["date", "iid", "symbol", "ym", "px", "volume", "oi"]].sort_values(["date", "ym"]).reset_index(drop=True)
    out.attrs["price_source"] = src
    return out


# --------------------------------------------------------------------------------------------
# Continuous contract by open interest
# --------------------------------------------------------------------------------------------
@dataclass
class Leg:
    name: str
    spec: Spec
    px: pd.DataFrame                    # date x ym  price wide table
    active: pd.Series                   # ym held from close t to close t+1
    nxt: pd.Series                      # next listed ym after active (for calendars / carry)
    dpx: pd.Series                      # price change of yesterday's active contract (t-1 -> t)
    level: pd.Series                    # price of today's active contract
    logret: pd.Series                   # log return of yesterday's active contract
    fx: pd.Series = field(default=None)  # USD per unit of quote currency (1.0 for USD)

    @property
    def usd_point(self) -> pd.Series:
        return self.spec.per_point * self.fx

    @property
    def notional(self) -> pd.Series:
        return self.level * self.usd_point

    @property
    def cost_unit(self) -> pd.Series:
        """USD per contract per side: half a tick plus fees."""
        return 0.5 * self.spec.tick * self.usd_point + self.spec.fee

    @property
    def carry(self) -> pd.Series:
        """Annualized log spread between the active and the deferred contract (the curve slope)."""
        cols = {c: j for j, c in enumerate(self.px.columns)}
        P = self.px.to_numpy()
        out = np.full(len(P), np.nan)
        for t, (a, b) in enumerate(zip(self.active.to_numpy(), self.nxt.to_numpy())):
            if a in cols and b in cols and b > a:
                pa, pb = P[t, cols[a]], P[t, cols[b]]
                if pa > 0 and pb > 0:
                    out[t] = np.log(pb / pa) * 12 / (b - a)
        return pd.Series(out, index=self.px.index)

    @property
    def logpx(self) -> pd.Series:
        return self.logret.fillna(0).cumsum()


def build_leg(df: pd.DataFrame, name: str, min_days_left: int = 0) -> Leg:
    df = df.astype({"px": float, "volume": float, "oi": float})
    px = df.pivot_table(index="date", columns="ym", values="px", aggfunc="last").sort_index()
    oi_cov = df["oi"].notna().mean()
    col = "oi" if oi_cov > 0.9 else "volume"   # never mix OI and volume scales in one series
    liq = df.pivot_table(index="date", columns="ym", values=col, aggfunc="last").reindex_like(px)
    if col == "volume" and name in SCHED_DAYS:
        return _finish(px, _schedule_roll(px, SCHED_DAYS[name]), liq.fillna(0).to_numpy(), name)
    cols = np.array(px.columns)
    P, L = px.to_numpy(), liq.fillna(0).to_numpy()
    n, m = P.shape
    # 3-day smoothed liquidity; only the next few listed contracts may take over, so a one-day
    # volume spike in a far month can never hijack the roll
    LR = pd.DataFrame(L).rolling(3 if col == "oi" else 7, min_periods=1).mean().to_numpy()
    act = np.full(n, -1)
    cur = -1
    for t in range(n):
        valid = np.where(~np.isnan(P[t]))[0]
        if cur < 0 or np.isnan(P[t, cur]) and (t + 1 >= n or np.isnan(P[t + 1, cur])):
            cand = valid[valid > cur][:4] if cur >= 0 else valid[:4]
            if len(cand) == 0:
                act[t] = cur
                continue
            cur = cand[np.argmax(LR[t, cand])]
        else:
            nearby = valid[valid > cur][:3]
            if len(nearby):
                j = nearby[np.argmax(LR[t, nearby])]
                if LR[t, j] > LR[t, cur]:
                    cur = j
        act[t] = cur
    return _finish(px, act, L, name)


def _schedule_roll(px: pd.DataFrame, days: int) -> np.ndarray:
    """Calendar roll for products without reliable OI or volume (ICE without settle files):
    hold the first listed contract whose delivery month starts more than `days` days ahead."""
    cols = np.array(px.columns)
    P = px.to_numpy()
    act = np.full(len(px), -1)
    for t, d in enumerate(px.index):
        valid = np.where(~np.isnan(P[t]))[0]
        for j in valid:
            y, mth = divmod(int(cols[j]), 12)
            start = pd.Timestamp(year=y, month=mth + 1, day=1, tz="UTC")
            if (start - d).days > days:
                act[t] = j
                break
        if act[t] < 0 and t > 0:
            act[t] = act[t - 1]
    return act


def _finish(px, act, L, name) -> "Leg":
    cols = np.array(px.columns)
    P = px.to_numpy()
    n, m = P.shape
    act_s = pd.Series(np.where(act >= 0, cols[np.clip(act, 0, None)], np.nan), index=px.index)
    # next listed contract after the active one, with a price today
    # deferred leg: the next contract in the product's traded cycle (the months that are ever
    # held as front), so GC Dec pairs with Feb, not with an illiquid serial month
    held = pd.Series(cols[act[act >= 0]] % 12)
    share = held.value_counts(normalize=True)
    cycle = set(share[share > 0.02].index)
    in_cycle = np.array([c % 12 in cycle for c in cols])
    nxt = np.full(n, np.nan)
    for t in range(n):
        if act[t] < 0:
            continue
        later = np.where(~np.isnan(P[t]) & (np.arange(m) > act[t]) & in_cycle)[0]
        if len(later):
            nxt[t] = cols[later[0]]
    level = pd.Series([P[t, act[t]] if act[t] >= 0 else np.nan for t in range(n)], index=px.index)
    dp = np.full(n, np.nan)
    lr = np.full(n, np.nan)
    for t in range(1, n):
        a = act[t - 1]
        if a < 0:
            continue
        p0, p1 = P[t - 1, a], P[t, a]
        if np.isnan(p1):  # held contract did not print today: carry the last price, no PnL today
            p1 = p0
        dp[t] = p1 - p0
        lr[t] = np.log(p1 / p0) if p0 > 0 and p1 > 0 else np.nan
    return Leg(name, SPECS[name], px, act_s, pd.Series(nxt, index=px.index),
               pd.Series(dp, index=px.index), level, pd.Series(lr, index=px.index))


def load_legs(cache: str, names: list[str]) -> dict[str, Leg]:
    legs = {}
    for nm in names:
        try:
            legs[nm] = build_leg(load(cache, nm), nm)
        except Exception as exc:  # a root with odd symbols should not stop the run
            print(f"  skip {nm}: {exc}", flush=True)
    for nm, lg in legs.items():
        if lg.spec.ccy == "USD":
            lg.fx = pd.Series(1.0, index=lg.px.index)
        else:
            f = legs.get(FX_FOR[lg.spec.ccy])
            lg.fx = f.level.reindex(lg.px.index).ffill() if f is not None else pd.Series(np.nan, index=lg.px.index)
    return legs


# --------------------------------------------------------------------------------------------
# Engine: lots per leg (decided at close t) -> net USD PnL with fill lag, costs and rolls
# --------------------------------------------------------------------------------------------
def simulate(lots: pd.DataFrame, legs: dict[str, Leg], fill_lag: int = 1) -> pd.DataFrame:
    idx = lots.index
    held = lots.shift(fill_lag).fillna(0)
    gross = pd.Series(0.0, index=idx)
    cost = pd.Series(0.0, index=idx)
    expo = pd.Series(0.0, index=idx)
    for r in lots.columns:
        lg = legs[r]
        n = held[r]
        dp = lg.dpx.reindex(idx).fillna(0)
        upt = lg.usd_point.reindex(idx).ffill()
        gross += n.shift(1).fillna(0) * dp * upt
        cu = lg.cost_unit.reindex(idx).ffill()
        act = lg.active.reindex(idx).ffill()
        rolls = (act != act.shift(1)) & act.shift(1).notna()
        traded = (n - n.shift(1).fillna(0)).abs() + rolls.astype(float) * 2 * n.shift(1).fillna(0).abs()
        cost += traded * cu
        expo += (n.abs() * lg.notional.reindex(idx).ffill()).fillna(0)
    out = pd.DataFrame({"gross": gross, "cost": cost, "exposure": expo})
    out["net"] = out["gross"] - out["cost"]
    return out.fillna(0)


def to_lots(dollars: pd.DataFrame, legs: dict[str, Leg]) -> pd.DataFrame:
    out = {}
    for r in dollars.columns:
        nt = legs[r].notional.reindex(dollars.index).ffill()
        out[r] = (dollars[r] / nt).replace([np.inf, -np.inf], np.nan).fillna(0).round()
    return pd.DataFrame(out, index=dollars.index)


# --------------------------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------------------------
def summarize(net: pd.Series) -> dict:
    r = net / CAP
    sd = r.std()
    eq = CAP + net.cumsum()
    peak = np.maximum.accumulate(np.r_[CAP, eq.to_numpy()])[1:]
    return {"sharpe": float(np.sqrt(252) * r.mean() / sd) if sd > 0 else np.nan,
            "ann_ret": float(r.mean() * 252), "ann_vol": float(sd * np.sqrt(252)),
            "mdd": float((eq.to_numpy() / peak - 1).min()) if len(eq) else np.nan}


def half_life(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 30:
        return np.nan
    dx, lag = x.diff().iloc[1:], x.shift(1).iloc[1:]
    b = np.polyfit(lag.to_numpy(), dx.to_numpy(), 1)[0]
    return float(-np.log(2) / b) if b < 0 else np.inf


def hurst(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 60:
        return np.nan
    lags = np.arange(2, 21)
    tau = [np.std(x[l:] - x[:-l]) for l in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0]) if min(tau) > 0 else np.nan


def deflated_sharpe(daily_ret: pd.Series, trial_sharpes) -> dict:
    from scipy.stats import kurtosis, norm, skew
    r = daily_ret.dropna()
    T = len(r)
    sr = r.mean() / r.std() if r.std() > 0 else 0.0
    tr = np.asarray([s for s in trial_sharpes if np.isfinite(s)]) / np.sqrt(252)
    n = len(tr)
    g = 0.5772156649
    sr0 = np.sqrt(tr.var(ddof=1)) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e))) if n > 1 else 0.0
    den = np.sqrt(max(1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr ** 2, 1e-12))
    return {"dsr": float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / den)), "sr0_ann": float(sr0 * np.sqrt(252)),
            "psr": float(norm.cdf(sr * np.sqrt(T - 1) / den)), "n": int(n)}
