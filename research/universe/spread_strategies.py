"""Strategy families for the universe lab.

1. Economic structures (cracks, crush, white premium, venue spreads, calendars, ...):
   fixed contract ratios or a rolling-beta hedge, z-score entry/exit, vol targeted.
   A 3x3 parameter grid per structure plus a walk-forward pick (trailing 3 years).
2. Formation/trading screen (Gatev-Goetzmann-Rouwenhorst style): each year pick the most
   cointegrated pairs / Johansen triples inside a universe on the previous 3 years, trade them
   the next year with frozen weights.
3. PCA residual s-score (Avellaneda-Lee) inside a universe.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

import core

warnings.filterwarnings("ignore")
VOL_TGT = 0.10            # annual vol target on CAP per strategy
DAILY_TGT = VOL_TGT / np.sqrt(252) * core.CAP
GROSS_CAP = 4_000_000.0   # max gross notional for a spread strategy
GRID = [(L, e) for L in (20, 60, 120) for e in (1.5, 2.0, 2.5)]


def threshold(z: np.ndarray, entry: float, exit_: float = 0.0, stop: float = 4.0) -> np.ndarray:
    out, pos, stopped = np.zeros(len(z)), 0, False
    for i, zi in enumerate(z):
        if np.isnan(zi):
            pos = 0
        elif abs(zi) >= stop:
            pos, stopped = 0, True
        elif stopped:
            stopped = abs(zi) >= entry
        elif pos == 1 and zi >= exit_:
            pos = 0
        elif pos == -1 and zi <= exit_:
            pos = 0
        elif pos == 0 and zi >= entry:
            pos = -1
        elif pos == 0 and zi <= -entry:
            pos = 1
        out[i] = pos
    return out


def common_index(legs, names, start=None):
    idx = None
    for n in names:
        i = legs[n].level.dropna().index
        idx = i if idx is None else idx.union(i)
    lo = max(legs[n].level.dropna().index[0] for n in names)
    hi = min(legs[n].level.dropna().index[-1] for n in names)
    idx = idx[(idx >= lo) & (idx <= hi)]
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    return idx


# --------------------------------------------------------------------------------------------
# 1a. fixed-ratio structures, simulated per spread unit
# --------------------------------------------------------------------------------------------
def _held_dp(leg: core.Leg, held: pd.Series, idx) -> pd.Series:
    """Price change t-1 -> t of the contract held at t-1 (held = ym series)."""
    px = leg.px.reindex(idx).ffill(limit=3)
    h = held.reindex(idx).shift(1)
    out = np.full(len(idx), np.nan)
    P = px.to_numpy()
    cols = {c: j for j, c in enumerate(px.columns)}
    hv = h.to_numpy()
    for t in range(1, len(idx)):
        j = cols.get(hv[t])
        if j is None:
            continue
        p0, p1 = P[t - 1, j], P[t, j]
        out[t] = (p1 - p0) if not (np.isnan(p0) or np.isnan(p1)) else 0.0
    return pd.Series(out, index=idx).fillna(0)


class Fixed:
    """Spread unit = sum_i w_i contracts of leg i (calendar legs use 'X@next')."""

    def __init__(self, name, cls, weights: dict, legs, signal: pd.Series | None = None, book=False):
        self.name, self.cls, self.w = name, cls, weights
        base = [k.split("@")[0] for k in weights]
        self.idx = common_index(legs, sorted(set(base)))
        pnl = pd.Series(0.0, index=self.idx)
        side_cost = pd.Series(0.0, index=self.idx)
        roll_cost = pd.Series(0.0, index=self.idx)
        notion = pd.Series(0.0, index=self.idx)
        for k, w in weights.items():
            r, which = (k.split("@") + ["active"])[:2]
            lg = legs[r]
            held = lg.active if which == "active" else lg.nxt
            upt = lg.usd_point.reindex(self.idx).ffill()
            pnl += w * _held_dp(lg, held, self.idx) * upt
            cu = lg.cost_unit.reindex(self.idx).ffill()
            if book:  # listed spread: fees on every leg, half a tick once for the whole spread
                cu = pd.Series(lg.spec.fee, index=self.idx) + (0.5 * lg.spec.tick * upt / sum(abs(v) for v in weights.values()))
            side_cost += abs(w) * cu
            h = held.reindex(self.idx).ffill()
            roll_cost += ((h != h.shift(1)) & h.shift(1).notna()).astype(float) * 2 * abs(w) * cu
            notion += abs(w) * lg.level.reindex(self.idx).ffill() * upt
        self.unit_pnl, self.side_cost, self.roll_cost, self.unit_notional = pnl, side_cost, roll_cost, notion
        self.level = pnl.cumsum()          # back-adjusted spread value in USD per unit
        # signal: the back-adjusted level, or a supplied series (calendars use the curve slope,
        # which is the price people quote; long the unit = long front / short deferred, which
        # gains when the slope falls, so the slope enters with a minus sign)
        self.sig_level = self.level if signal is None else -signal.reindex(self.idx).ffill()

    def run(self, L, entry, exit_=0.0, stop=4.0):
        lv = self.sig_level
        z = ((lv - lv.rolling(L, min_periods=L).mean()) / lv.rolling(L, min_periods=L).std()).to_numpy()
        sig = threshold(z, entry, exit_, stop)
        uvol = self.unit_pnl.rolling(63, min_periods=40).std().shift(1)
        k = (DAILY_TGT / uvol).clip(upper=GROSS_CAP / self.unit_notional).fillna(0).round()
        units = pd.Series(sig, index=self.idx) * k
        held = units.shift(1).fillna(0)                 # fill one day after the signal
        gross = held.shift(1).fillna(0) * self.unit_pnl
        cost = (held - held.shift(1).fillna(0)).abs() * self.side_cost + held.shift(1).fillna(0).abs() * self.roll_cost
        return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost})


# --------------------------------------------------------------------------------------------
# 1b. rolling-beta pair (log prices), dollar sized
# --------------------------------------------------------------------------------------------
class Hedged:
    def __init__(self, name, cls, a, b, legs, beta_win=252):
        self.name, self.cls, self.a, self.b, self.legs = name, cls, a, b, legs
        self.idx = common_index(legs, [a, b])
        la = legs[a].logpx.reindex(self.idx).ffill()
        lb = legs[b].logpx.reindex(self.idx).ffill()
        ra, rb = la.diff(), lb.diff()
        cov = ra.rolling(beta_win, min_periods=120).cov(rb)
        var = rb.rolling(beta_win, min_periods=120).var()
        self.beta = (cov / var).clip(0.2, 5).shift(1)
        # level: log spread with the slow beta (re-anchored daily: cumulated hedged returns)
        self.hret = (ra - self.beta * rb).fillna(0)
        self.level = self.hret.cumsum()
        self.unit_pnl = self.hret * core.CAP     # PnL of $1m long A / beta short B, for vol only

    def run(self, L, entry, exit_=0.0, stop=4.0):
        lv = self.level
        z = ((lv - lv.rolling(L, min_periods=L).mean()) / lv.rolling(L, min_periods=L).std()).to_numpy()
        sig = pd.Series(threshold(z, entry, exit_, stop), index=self.idx)
        hv = self.hret.rolling(63, min_periods=40).std().shift(1)
        D = (DAILY_TGT / hv).clip(upper=GROSS_CAP / (1 + self.beta.fillna(1))).fillna(0)
        dollars = pd.DataFrame({self.a: sig * D, self.b: -sig * D * self.beta.fillna(0)}, index=self.idx)
        lots = core.to_lots(dollars, self.legs)
        return core.simulate(lots, self.legs)[["gross", "cost", "net"]]


def walk_forward(nets: dict, train_years=3, others: dict | None = None):
    """nets: param -> daily net. Each year use the param with the best trailing net Sharpe.
    others: optional param -> series (e.g. gross) that follows the same yearly choices."""
    df = pd.DataFrame(nets).fillna(0)
    od = pd.DataFrame(others).reindex(df.index).fillna(0) if others else None
    years = sorted(set(df.index.year))
    out = pd.Series(0.0, index=df.index)
    out2 = pd.Series(0.0, index=df.index)
    for y in years[train_years:]:
        tr = df[(df.index.year >= y - train_years) & (df.index.year < y)]
        sh = tr.mean() / tr.std().replace(0, np.nan)
        if sh.notna().any():
            best = sh.idxmax()
            m = df.index.year == y
            out[m] = df.loc[m, best]
            if od is not None:
                out2[m] = od.loc[m, best]
    first = years[train_years] if len(years) > train_years else years[-1]
    keep = out.index.year >= first
    return (out[keep], out2[keep]) if others else out[keep]


# --------------------------------------------------------------------------------------------
# 2. formation / trading screen
# --------------------------------------------------------------------------------------------
def formation_trading(legs, names, kind="pair", top_k=4, form_years=3, z_mode="formation",
                      entry=2.0, exit_=0.0, stop=4.0, hl_range=(2, 60)):
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    idx = common_index(legs, names)
    LP = pd.DataFrame({n: legs[n].logpx.reindex(idx).ffill() for n in names})
    years = sorted(set(idx.year))
    dollars = pd.DataFrame(0.0, index=idx, columns=names)
    log = {}
    for y in years[form_years:]:
        f = LP[(LP.index.year >= y - form_years) & (LP.index.year < y)].dropna()
        tmask = LP.index.year == y
        if len(f) < 250 or tmask.sum() == 0:
            continue
        cands = []
        if kind == "pair":
            for a, b in itertools.combinations(names, 2):
                try:
                    p = coint(f[a], f[b])[1]
                except Exception:
                    continue
                beta = np.polyfit(f[b], f[a], 1)[0]
                if not (0.1 < abs(beta) < 10):
                    continue
                w = {a: 1.0, b: -beta}
                s = f[a] - beta * f[b]
                hl = core.half_life(s)
                if p < 0.05 and hl_range[0] <= hl <= hl_range[1]:
                    cands.append((p, w, s))
            cands.sort(key=lambda c: c[0])
        else:
            for tri in itertools.combinations(names, 3):
                try:
                    j = coint_johansen(f[list(tri)], 0, 1)
                except Exception:
                    continue
                stat, crit = j.lr1[0], j.cvt[0, 1]
                if stat < crit:
                    continue
                v = np.real(j.evec[:, 0])
                v = v / np.abs(v).sum()
                s = f[list(tri)] @ v
                hl = core.half_life(s)
                if hl_range[0] <= hl <= hl_range[1]:
                    cands.append((-stat / crit, dict(zip(tri, v)), s))
            cands.sort(key=lambda c: c[0])
        chosen = cands[:top_k]
        log[y] = [" / ".join(f"{k}{v:+.2f}" for k, v in w.items()) for _, w, _ in chosen]
        if not chosen:
            continue
        per = 1.0 / len(chosen)
        for _, w, s_form in chosen:
            ks = list(w)
            sfull = LP[ks] @ pd.Series(w)
            if z_mode == "formation":
                z = (sfull - s_form.mean()) / s_form.std()
            else:
                z = (sfull - sfull.rolling(60).mean()) / sfull.rolling(60).std()
            zz = z[tmask].to_numpy()
            sig = pd.Series(threshold(zz, entry, exit_, stop), index=LP.index[tmask])
            r = (LP[ks].diff() @ pd.Series(w)).rolling(63, min_periods=40).std().shift(1)
            gross_w = sum(abs(v) for v in w.values())
            D = (per * DAILY_TGT / r[tmask]).clip(upper=per * GROSS_CAP / gross_w).fillna(0)
            for k_, v in w.items():
                dollars.loc[tmask, k_] += sig * D * v
    lots = core.to_lots(dollars, legs)
    res = core.simulate(lots, legs)[["gross", "cost", "net"]]
    first = years[form_years] if len(years) > form_years else years[-1]
    return res[res.index.year >= first], log


# --------------------------------------------------------------------------------------------
# 3. PCA residual s-score
# --------------------------------------------------------------------------------------------
def pca_residual(legs, names, win=252, win_ou=60, n_fac=1, s_open=1.25, s_close=0.5, every=5):
    idx = common_index(legs, names)
    R = pd.DataFrame({n: legs[n].logret.reindex(idx) for n in names}).fillna(0)
    Rv = R.to_numpy()
    T, N = Rv.shape
    pos = np.zeros((T, N))
    cur = np.zeros(N)
    last_s = np.full(N, np.nan)
    for t in range(win, T):
        if (t - win) % every == 0:
            X = Rv[t - win:t]
            sd = X.std(0)
            sd[sd == 0] = 1
            Z = (X - X.mean(0)) / sd
            ev, V = np.linalg.eigh(np.corrcoef(Z.T))
            F = Z @ V[:, -n_fac:] / np.sqrt(np.maximum(ev[-n_fac:], 1e-12))
            Xo = X[-win_ou:]
            Fo = F[-win_ou:]
            A = np.c_[np.ones(win_ou), Fo]
            coef, *_ = np.linalg.lstsq(A, Xo, rcond=None)
            eps = Xo - A @ coef
            Xc = np.cumsum(eps, 0)
            for i in range(N):
                x0, x1 = Xc[:-1, i], Xc[1:, i]
                b, a = np.polyfit(x0, x1, 1)
                if not (0 < b < 1):
                    last_s[i] = np.nan
                    continue
                kappa = -np.log(b) * 252
                m = a / (1 - b)
                res = x1 - (a + b * x0)
                seq = np.sqrt(res.var() / (1 - b * b))
                last_s[i] = (Xc[-1, i] - m) / seq if kappa > 252 / 30 and seq > 0 else np.nan
        for i in range(N):
            s = last_s[i]
            if np.isnan(s):
                cur[i] = 0
            elif cur[i] == 0 and s > s_open:
                cur[i] = -1
            elif cur[i] == 0 and s < -s_open:
                cur[i] = 1
            elif cur[i] == 1 and s > -s_close:
                cur[i] = 0
            elif cur[i] == -1 and s < s_close:
                cur[i] = 0
        pos[t] = cur
    P = pd.DataFrame(pos, index=idx, columns=names)
    vol = R.rolling(63, min_periods=40).std().shift(1)
    D = DAILY_TGT / np.sqrt(N) / vol
    dollars = (P * D).sub((P * D).mean(1), axis=0).clip(-GROSS_CAP / N, GROSS_CAP / N).fillna(0)
    lots = core.to_lots(dollars, legs)
    res = core.simulate(lots, legs)[["gross", "cost", "net"]]
    return res.iloc[win:]
