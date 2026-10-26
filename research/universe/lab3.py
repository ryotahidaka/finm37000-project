"""Lab v3: every pair and calendar on the data already pulled, several strategy families,
two cost models, out-of-sample strategy choice, and a relationship map.

Cost models
  leg     every leg pays half a tick plus fees on every contract traded (crossing both sides)
  spread  the spread is worked as one order: every leg still pays fees, but only one leg
          crosses (half a tick on the leg with the biggest tick value traded); the others fill
          at mid. This is the "leg the hard side, hit the hedge" execution idea.

Strategy families (per spread)
  mr       z-score mean reversion, z window L, entry e, exit 0, stop 4      (9 configs)
  mom      trend on the spread: sign(level - its L-day mean), weekly rebalance  (3)
  regime   rolling Hurst of the spread: MR when H < 0.45, trend when H > 0.55, else flat (1)
  kalman   Kalman-filter hedge ratio, trade the standardized forecast error (3)
  meta     each year pick, out of sample, the config with the best trailing 3-year net Sharpe
           among all 16 for that spread (this is "choose the strategy" honestly)

python lab3.py <cache> <out> [--exclude HH]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import core
import spread_strategies as st

VOL = st.DAILY_TGT
CAPG = st.GROSS_CAP
START = pd.Timestamp("2014-01-01", tz="UTC")


# ------------------------------------------------------------------------------------------
# signals
# ------------------------------------------------------------------------------------------
def sig_mr(level, L, e):
    z = (level - level.rolling(L, min_periods=L).mean()) / level.rolling(L, min_periods=L).std()
    return pd.Series(st.threshold(z.to_numpy(), e), index=level.index)


def sig_mom(level, L):
    s = np.sign(level - level.rolling(L, min_periods=L).mean())
    s[level.rolling(L, min_periods=L).mean().isna()] = 0
    weekly = s.copy()
    weekly[:] = np.nan
    weekly.iloc[::5] = s.iloc[::5]
    return weekly.ffill().fillna(0)


def rolling_hurst(level, win=126, every=5):
    x = level.to_numpy()
    out = np.full(len(x), np.nan)
    for t in range(win, len(x), every):
        out[t] = core.hurst(x[t - win:t])
    return pd.Series(out, index=level.index).ffill()


def sig_regime(level, H):
    mr = sig_mr(level, 60, 2.0)
    mom = sig_mom(level, 126)
    return pd.Series(np.where(H < 0.45, mr, np.where(H > 0.55, mom, 0.0)), index=level.index)


def kalman(y, x, delta=1e-4, ve=1e-3):
    """Chan-style Kalman regression y = a + b x. Returns forecast error e, its std sqrt(Q), beta."""
    n = len(y)
    th = np.zeros(2)
    P = np.zeros((2, 2))
    Vw = delta / (1 - delta) * np.eye(2)
    e = np.full(n, np.nan)
    Q = np.full(n, np.nan)
    b = np.full(n, np.nan)
    for t in range(n):
        F = np.array([1.0, x[t]])
        R = P + Vw
        yhat = F @ th
        Q[t] = F @ R @ F + ve
        e[t] = y[t] - yhat
        b[t] = th[1]
        K = R @ F / Q[t]
        th = th + K * e[t]
        P = R - np.outer(K, F) @ R
    return e, np.sqrt(Q), b


def sig_kalman(z, entry):
    return pd.Series(st.threshold(z.to_numpy(), entry, 0.0, 99.0), index=z.index)


# ------------------------------------------------------------------------------------------
# engine with two cost models (lots per leg decided at close t, filled at t+1)
# ------------------------------------------------------------------------------------------
def simulate(lots: pd.DataFrame, legs, mode="leg"):
    idx = lots.index
    held = lots.shift(1).fillna(0)
    gross = pd.Series(0.0, index=idx)
    fees = pd.Series(0.0, index=idx)
    ticks = []
    for r in lots.columns:
        lg = legs[r]
        n = held[r]
        upt = lg.usd_point.reindex(idx).ffill()
        gross += n.shift(1).fillna(0) * lg.dpx.reindex(idx).fillna(0) * upt
        act = lg.active.reindex(idx).ffill()
        rolls = ((act != act.shift(1)) & act.shift(1).notna()).astype(float)
        traded = (n - n.shift(1).fillna(0)).abs() + rolls * 2 * n.shift(1).fillna(0).abs()
        fees += traded * lg.spec.fee
        ticks.append(traded * 0.5 * lg.spec.tick * upt)
    T = pd.concat(ticks, axis=1).fillna(0)
    tick_cost = T.sum(axis=1) if mode == "leg" else T.max(axis=1)
    cost = fees + tick_cost
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost}).fillna(0)


class Pair:
    def __init__(self, a, b, legs):
        self.a, self.b, self.legs = a, b, legs
        self.idx = st.common_index(legs, [a, b])
        la = legs[a].logpx.reindex(self.idx).ffill()
        lb = legs[b].logpx.reindex(self.idx).ffill()
        ra, rb = la.diff(), lb.diff()
        beta = (ra.rolling(252, min_periods=120).cov(rb) / rb.rolling(252, min_periods=120).var())
        self.beta = beta.clip(0.1, 10).shift(1)
        self.hret = (ra - self.beta * rb).fillna(0)
        self.level = self.hret.cumsum()
        self.H = rolling_hurst(self.level)
        e, sq, kb = kalman(la.to_numpy(), lb.to_numpy())
        self.kz = pd.Series(e / sq, index=self.idx)
        self.kz.iloc[:120] = np.nan
        self.kbeta = pd.Series(kb, index=self.idx).clip(0.1, 10).shift(1)
        self.khret = (ra - self.kbeta * rb).fillna(0)

    def run(self, sig, mode, use_kalman=False):
        beta = (self.kbeta if use_kalman else self.beta).fillna(0)
        hret = self.khret if use_kalman else self.hret
        hv = hret.rolling(63, min_periods=40).std().shift(1)
        D = (VOL / hv).clip(upper=CAPG / (1 + beta.abs().replace(0, 1))).fillna(0)
        dollars = pd.DataFrame({self.a: sig * D, self.b: -sig * D * beta}, index=self.idx)
        return simulate(core.to_lots(dollars, self.legs), self.legs, mode)

    def configs(self):
        out = {}
        for L, e in st.GRID:
            out[f"mr z{L} e{e}"] = (sig_mr(self.level, L, e), False)
        for L in (63, 126, 252):
            out[f"mom {L}d"] = (sig_mom(self.level, L), False)
        out["regime hurst"] = (sig_regime(self.level, self.H), False)
        for e in (1.0, 1.5, 2.0):
            out[f"kalman e{e}"] = (sig_kalman(self.kz, e), True)
        return out


def calendar_runs(root, legs, mode):
    obj = st.Fixed(f"{root} calendar", "calendar", {root: 1, f"{root}@next": -1}, legs,
                   legs[root].carry, book=(mode == "spread"))
    lv = obj.sig_level
    H = rolling_hurst(lv.ffill().fillna(0))
    sigs = {f"mr z{L} e{e}": sig_mr(lv, L, e) for L, e in st.GRID}
    sigs.update({f"mom {L}d": sig_mom(lv, L) for L in (63, 126, 252)})
    sigs["regime hurst"] = sig_regime(lv, H)
    out = {}
    uvol = obj.unit_pnl.rolling(63, min_periods=40).std().shift(1)
    k = (VOL / uvol).clip(upper=CAPG / obj.unit_notional).fillna(0).round()
    for nm, s in sigs.items():
        units = s * k
        held = units.shift(1).fillna(0)
        gross = held.shift(1).fillna(0) * obj.unit_pnl
        cost = (held - held.shift(1).fillna(0)).abs() * obj.side_cost + held.shift(1).fillna(0).abs() * obj.roll_cost
        out[nm] = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost})
    return out, lv


# ------------------------------------------------------------------------------------------
def relationship_map(legs, names, out):
    """Correlation, Engle-Granger p-values, stability and half-life for every pair."""
    from statsmodels.tsa.stattools import coint
    rows = []
    R = pd.DataFrame({n: legs[n].logret for n in names})
    LP = pd.DataFrame({n: legs[n].logpx.where(legs[n].level.notna()) for n in names})
    for a, b in itertools.combinations(names, 2):
        d = LP[[a, b]].dropna()
        d = d[d.index >= START]
        r = R[[a, b]].dropna()
        r = r[r.index >= START]
        if len(d) < 500:
            continue
        p = coint(d[a], d[b])[1]
        # stability: share of 2-year windows (stepping 6 months) where EG p < 0.05
        wins = []
        for s0 in pd.date_range(d.index[0], d.index[-1] - pd.DateOffset(years=2), freq="6MS"):
            w = d[(d.index >= s0) & (d.index < s0 + pd.DateOffset(years=2))]
            if len(w) > 300:
                wins.append(coint(w[a], w[b])[1] < 0.05)
        beta = np.polyfit(d[b], d[a], 1)[0]
        hl = core.half_life(d[a] - beta * d[b])
        rows.append({"a": a, "b": b, "corr": r[a].corr(r[b]), "corr_2y_min": r[a].rolling(504).corr(r[b]).min(),
                     "eg_p": p, "stable_share": np.mean(wins) if wins else np.nan, "half_life": hl,
                     "beta": beta, "start": str(d.index[0].date())})
    M = pd.DataFrame(rows)
    M.to_csv(f"{out}/relationships.csv", index=False)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("--exclude", nargs="*", default=["HH"])
    ap.add_argument("--settle-min", type=float, default=0.95)
    a = ap.parse_args()
    t0 = time.time()
    core.SETTLE_MIN = a.settle_min
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in core.available(a.cache) if n not in a.exclude]
    legs = core.load_legs(a.cache, names)
    tradable = [n for n in legs if core.SPECS[n].group != "fx"]
    print("tradable:", tradable, flush=True)
    M = relationship_map(legs, tradable, a.out)
    print(f"relationship map: {len(M)} pairs ({time.time() - t0:.0f}s)", flush=True)

    trials, series, wf_rows = [], {}, []

    def rec(spread, family, cfg, mode, res, kind):
        res = res[res.index >= START]
        if len(res) < 250 or res["net"].abs().sum() == 0:
            return
        s = core.summarize(res["net"])
        name = f"{spread} | {cfg} | {mode}"
        trials.append({"spread": spread, "kind": kind, "family": family, "config": cfg, "cost_mode": mode,
                       "name": name, **s, "gross_sharpe": core.summarize(res["gross"])["sharpe"],
                       "cost_usd": res["cost"].sum(), "start": str(res.index[0].date())})
        series[name] = res["net"]
        return res

    def do_spread(spread, kind, runs_by_mode):
        for mode, runs in runs_by_mode.items():
            kept = {}
            for cfg, res in runs.items():
                r = rec(spread, cfg.split()[0], cfg, mode, res, kind)
                if r is not None:
                    kept[cfg] = r
            fams = {}
            for cfg in kept:
                fams.setdefault(cfg.split()[0], []).append(cfg)
            for fam, cfgs in list(fams.items()) + [("meta", list(kept))]:
                nets = {c: kept[c]["net"] for c in cfgs}
                gr = {c: kept[c]["gross"] for c in cfgs}
                cs = {c: kept[c]["cost"] for c in cfgs}
                if not nets:
                    continue
                wf, wfg = st.walk_forward(nets, others=gr)
                _, wfc = st.walk_forward(nets, others=cs)
                res = pd.DataFrame({"net": wf, "gross": wfg, "cost": wfc})
                if len(res) < 250:
                    continue
                s = core.summarize(res["net"])
                name = f"{spread} | WF {fam} | {mode}"
                series[name] = res["net"]
                wf_rows.append({"spread": spread, "kind": kind, "family": fam, "cost_mode": mode, "name": name, **s,
                                "gross_sharpe": core.summarize(res["gross"])["sharpe"], "cost_usd": res["cost"].sum(),
                                "start": str(res.index[0].date())})

    for x, y in itertools.combinations(tradable, 2):
        try:
            P = Pair(x, y, legs)
        except Exception as exc:
            print("skip", x, y, exc)
            continue
        cfgs = P.configs()
        runs = {m: {c: P.run(s, m, k) for c, (s, k) in cfgs.items()} for m in ("leg", "spread")}
        do_spread(f"{x}/{y}", "pair", runs)
        print(f"{x}/{y} done ({time.time() - t0:.0f}s)", flush=True)

    # control: the same Kalman rule on each product alone (regression on a constant = adaptive
    # level). If the "pairs" results match this, they are outright mean reversion, not spreads.
    for r in tradable:
        idx = legs[r].level.dropna().index
        lp = legs[r].logpx.reindex(idx).ffill()
        e, sq, _ = kalman(lp.to_numpy(), np.zeros(len(lp)))
        z = pd.Series(e / sq, index=idx)
        z.iloc[:120] = np.nan
        rv = lp.diff().rolling(63, min_periods=40).std().shift(1)
        D = (VOL / rv).clip(upper=CAPG).fillna(0)
        runs = {m: {f"kalman e{x}": simulate(core.to_lots(pd.DataFrame({r: sig_kalman(z, x) * D}), legs), legs, m)
                    for x in (1.0, 1.5, 2.0)} for m in ("leg", "spread")}
        do_spread(f"{r} outright (control)", "outright", runs)

    for r in tradable:
        if legs[r].nxt.notna().mean() < 0.8:
            continue
        runs = {m: calendar_runs(r, legs, m)[0] for m in ("leg", "spread")}
        do_spread(f"{r} calendar", "calendar", runs)
        print(f"{r} calendar done ({time.time() - t0:.0f}s)", flush=True)

    T = pd.DataFrame(trials)
    W = pd.DataFrame(wf_rows)
    allsh = T["sharpe"].dropna().tolist()
    # deflated Sharpe for the walk-forward results: against every config tried, and against the
    # number of spreads x families actually compared (the WF table itself)
    wfsh = W["sharpe"].dropna().tolist()
    for i, r in W.iterrows():
        ret = series[r["name"]] / core.CAP
        d_all = core.deflated_sharpe(ret, allsh)
        d_wf = core.deflated_sharpe(ret, wfsh)
        W.loc[i, ["psr", "dsr_vs_configs", "dsr_vs_wf"]] = [d_all["psr"], d_all["dsr"], d_wf["dsr"]]

    # portfolios: all meta-WF spreads, risk parity, per cost mode; plus pre-screened (stable pairs)
    def book(names_, label):
        X = pd.DataFrame({n: series[n] for n in names_}).fillna(0)
        if X.shape[1] < 2:
            return
        vol = X.rolling(126, min_periods=60).std().shift(1)
        w = (1 / vol).replace(np.inf, np.nan)
        w = w.div(w.sum(1), axis=0).fillna(0)
        p = (X * w).sum(1)
        net = (p * (VOL / p.rolling(126, min_periods=60).std().shift(1)).clip(upper=3.0)).fillna(0)
        net = net[net.index >= pd.Timestamp("2017-01-01", tz="UTC")]
        s = core.summarize(net)
        series[label] = net
        wf_rows.append({"spread": label, "kind": "portfolio", "family": "meta", "cost_mode": label.split("[")[-1].rstrip("]"),
                        "name": label, **s, "gross_sharpe": np.nan, "cost_usd": np.nan, "start": str(net.index[0].date())})

    for mode in ("leg", "spread"):
        meta = W[(W["family"] == "meta") & (W["cost_mode"] == mode)]
        book(meta["name"].tolist(), f"Book: every spread, strategy chosen yearly [{mode}]")
        stable = M[M["stable_share"] >= 0.5]
        keys = {f"{r.a}/{r.b}" for r in stable.itertuples()}
        book(meta[meta["spread"].isin(keys) | (meta["kind"] == "calendar")]["name"].tolist(),
             f"Book: stable pairs + calendars [{mode}]")
        book(meta[meta["kind"] == "calendar"]["name"].tolist(), f"Book: calendars only [{mode}]")
    W = pd.DataFrame(wf_rows)
    for i, r in W.iterrows():
        if pd.isna(r.get("psr")):
            ret = series[r["name"]] / core.CAP
            d = core.deflated_sharpe(ret, allsh)
            W.loc[i, ["psr", "dsr_vs_configs", "dsr_vs_wf"]] = [d["psr"], d["dsr"], core.deflated_sharpe(ret, wfsh)["dsr"]]

    T.sort_values("sharpe", ascending=False).to_csv(f"{a.out}/configs.csv", index=False)
    W.sort_values("sharpe", ascending=False).to_csv(f"{a.out}/walkforward.csv", index=False)
    pd.DataFrame(series).to_parquet(f"{a.out}/series.parquet")
    json.dump({"roots": tradable, "n_configs": len(T), "n_wf": len(wf_rows), "minutes": (time.time() - t0) / 60,
               "sr0_configs": core.deflated_sharpe(pd.Series(np.random.default_rng(0).normal(size=999)), allsh)["sr0_ann"],
               "sr0_wf": core.deflated_sharpe(pd.Series(np.random.default_rng(0).normal(size=999)), wfsh)["sr0_ann"]},
              open(f"{a.out}/meta.json", "w"), indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min: {len(T)} configs, {len(W)} walk-forward series")
    print(W.sort_values("sharpe", ascending=False)[["name", "sharpe", "gross_sharpe", "ann_ret", "mdd", "psr", "dsr_vs_wf"]]
          .head(25).round(3).to_string())


if __name__ == "__main__":
    main()
