"""Curve structures (calendars, butterflies, carry-vs-carry) on all available metals.

Grid fixed in advance (L in {20, 60}, entry in {1.5, 2, 2.5},
exit 0, stop 4), same-contract z, fill lag 1, per-leg costs, 10% vol target, MIN_VOL liquidity,
walk-forward from 2015 on the previous 3 years. Prices are settlements where available.
"""
import itertools

import numpy as np
import pandas as pd

from gcsi import contracts, metrics, rolls, signals
import lab

MIN_VOL = 200
MAX_LEG_NOTIONAL = 2_000_000.0
COST_MODE = "per_leg"
CAP, TARGET_VOL = 1_000_000.0, 0.10
CM = lab.COSTS
CURVES, IDX = {}, None

def curve(root, cache):
    o = rolls._outrights(lab.load_bars(cache, root), root)
    o["expiration"] = [contracts.last_trading_day(y, m).tz_localize("UTC") for y, m in zip(o["year"], o["month"])]
    px = o.pivot_table(index="ts_event", columns="contract", values="close")
    vol = o.pivot_table(index="ts_event", columns="contract", values="volume").fillna(0)
    meta = o.drop_duplicates("contract").set_index("contract")[["year", "month", "expiration"]]
    bday = pd.offsets.BDay(5)
    meta["last_hold"] = [(contracts.first_notice_day(y, m) - bday).tz_localize("UTC")
                         for y, m in zip(meta["year"], meta["month"])]
    meta = meta.sort_values(["year", "month"])
    slots = []
    for t in px.index:
        live = [c for c in meta.index if meta.at[c, "last_hold"] >= t and not np.isnan(px.at[t, c])]
        slots.append(live[:3] + [None] * (3 - len(live[:3])))
    slots = pd.DataFrame(slots, index=px.index, columns=["s1", "s2", "s3"])
    return px, vol, meta, slots


def years_between(meta, a, b):
    return (meta.at[b, "expiration"] - meta.at[a, "expiration"]).days / 365.25


# a structure = list of (root, slot, weight function(t, contracts) -> weight in lots per unit)
def calendar(root):
    return [(root, "s1", lambda t, c: -1.0), (root, "s2", lambda t, c: 1.0)]


def fly(root):
    return [(root, "s1", lambda t, c: 1.0), (root, "s2", lambda t, c: -2.0), (root, "s3", lambda t, c: 1.0)]


def carry_vs_carry(ra, rb):
    """Long ra calendar, short rb calendar, sized to equal $ per 1% of annual carry."""
    def wb(t, c):
        apx, _, ameta, _ = CURVES[ra]
        bpx, _, bmeta, _ = CURVES[rb]
        a1, a2, b1, b2 = c[(ra, "s1")], c[(ra, "s2")], c[(rb, "s1")], c[(rb, "s2")]
        a_dv = apx.at[t, a1] * years_between(ameta, a1, a2) * contracts.get_spec(ra).multiplier
        b_dv = bpx.at[t, b1] * years_between(bmeta, b1, b2) * contracts.get_spec(rb).multiplier
        return a_dv / b_dv
    return [(ra, "s1", lambda t, c: -1.0), (ra, "s2", lambda t, c: 1.0),
            (rb, "s1", lambda t, c: wb(t, c)), (rb, "s2", lambda t, c: -wb(t, c))]


def build(struct, L):
    """Per day: contracts held per leg, unit weights, value x, same-contract z, liquidity flag."""
    rows = []
    for i, t in enumerate(IDX):
        c = {(r, s): CURVES[r][3].at[t, s] for r, s, _ in struct}
        if any(not isinstance(v, str) for v in c.values()):
            rows.append(None); continue
        try:
            w = [f(t, c) for _, _, f in struct]
        except Exception:
            rows.append(None); continue
        # same-contract history over the previous L days (and today)
        hist = IDX[max(0, i - L):i + 1]
        x = sum(wk * np.log(CURVES[r][0].loc[hist, c[(r, s)]]) for (r, s, _), wk in
                zip(struct, [np.sign(v) * abs(v) for v in w]))
        liq = all(CURVES[r][1].at[t, c[(r, s)]] >= MIN_VOL for r, s, _ in struct)
        if x.iloc[:-1].notna().sum() < L * 0.9:
            z = np.nan
        else:
            past = x.iloc[:-1].dropna()
            z = (x.iloc[-1] - past.mean()) / past.std() if past.std() > 0 else np.nan
        rows.append({"contracts": c, "w": w, "z": z, "liq": liq})
    return rows


def unit_pnl(struct, rows):
    """USD PnL of one unit held from close t-1 to close t, using the contracts held at t-1."""
    pnl = np.zeros(len(IDX))
    for i in range(1, len(IDX)):
        r0 = rows[i - 1]
        if r0 is None:
            continue
        t0, t1 = IDX[i - 1], IDX[i]
        tot = 0.0
        for (root, s, _), wk in zip(struct, r0["w"]):
            px = CURVES[root][0]
            c = r0["contracts"][(root, s)]
            p0, p1 = px.at[t0, c], px.at[t1, c]
            if np.isnan(p1):
                p1 = p0  # no trade that day: stale mark
            tot += wk * (p1 - p0) * contracts.get_spec(root).multiplier
        pnl[i] = tot
    return pd.Series(pnl, index=IDX)


def simulate(struct, rows, entry, L):
    z = pd.Series([r["z"] if r else np.nan for r in rows], index=IDX)
    liq = pd.Series([bool(r and r["liq"]) for r in rows], index=IDX)
    sig = signals.generate_signals(z, entry, 0.0, 4.0)
    # no new positions on illiquid days
    s = sig.to_numpy().copy()
    for i in range(len(s)):
        prev = s[i - 1] if i else 0
        if s[i] != 0 and s[i] != prev and not liq.iloc[i]:
            s[i] = prev if prev == s[i] else 0
    sig = pd.Series(s, index=IDX)
    up = unit_pnl(struct, rows)
    vol = up.rolling(60).std().shift(1)
    units = (CAP * TARGET_VOL / np.sqrt(252) / vol).replace([np.inf], np.nan).fillna(0)
    # leverage cap: no leg above MAX_LEG_NOTIONAL (a quiet spread would otherwise get huge size)
    cap_units = []
    for i, t in enumerate(IDX):
        r = rows[i]
        if r is None:
            cap_units.append(0.0); continue
        per_unit = max(abs(wk) * CURVES[root][0].at[t, r["contracts"][(root, sl)]] * contracts.get_spec(root).multiplier
                       for (root, sl, _), wk in zip(struct, r["w"]))
        cap_units.append(MAX_LEG_NOTIONAL / per_unit if per_unit > 0 else 0.0)
    units = np.minimum(units, pd.Series(cap_units, index=IDX).fillna(0))
    target_units = (-sig * units)                      # z high -> short the structure
    held_units = target_units.shift(1).fillna(0)      # fill lag 1
    # integer lots per contract
    lots = []
    for i, t in enumerate(IDX):
        r = rows[i]
        d = {}
        if r is not None and held_units.iloc[i] != 0:
            for (root, sl, _), wk in zip(struct, r["w"]):
                d[(root, r["contracts"][(root, sl)])] = int(round(held_units.iloc[i] * wk))
        lots.append(d)
    # this-day PnL from yesterday's lots; costs from lot changes
    pnl, cost = np.zeros(len(IDX)), np.zeros(len(IDX))
    for i in range(1, len(IDX)):
        t0, t1 = IDX[i - 1], IDX[i]
        for (root, c), n in lots[i - 1].items():
            px = CURVES[root][0]
            p0, p1 = px.at[t0, c], px.at[t1, c]
            p1 = p0 if np.isnan(p1) else p1
            pnl[i] += n * (p1 - p0) * contracts.get_spec(root).multiplier
        keys = set(lots[i - 1]) | set(lots[i])
        for k in keys:
            d = lots[i].get(k, 0) - lots[i - 1].get(k, 0)
            if COST_MODE == "spread_book":
                # traded as one CME spread instrument: one half-tick for the whole structure
                # (charged across its legs) plus exchange fee and commission on every leg
                lc = CM.legs[k[0]]
                half = 0.5 * contracts.get_spec(k[0]).tick_value / len(struct)
                cost[i] += abs(d) * (half + lc.exchange_fee + lc.commission)
            else:
                cost[i] += CM.cost(k[0], d)
    out = pd.DataFrame({"gross": pnl, "cost": cost, "net": pnl - cost, "signal": sig}, index=IDX)
    return out


def ou_stats(rows, L=None):
    # mean reversion of the same-contract value: AR(1) of daily x changes vs level, crude
    z = pd.Series([r["z"] if r else np.nan for r in rows], index=IDX).dropna()
    return float(z.abs().gt(2).mean())


def run(cache, roots, cost_mode="per_leg"):
    global IDX, COST_MODE
    COST_MODE = cost_mode
    for r in roots:
        CURVES[r] = curve(r, cache)
    IDX = None
    for r in roots:
        IDX = CURVES[r][0].index if IDX is None else IDX.intersection(CURVES[r][0].index)
    structs = {}
    for r in roots:
        structs[f"{r} calendar"] = calendar(r)
        structs[f"{r} butterfly"] = fly(r)
    for ra, rb in (("GC", "SI"), ("PL", "PA"), ("GC", "PL")):
        if ra in roots and rb in roots:
            structs[f"{ra} carry vs {rb} carry"] = carry_vs_carry(ra, rb)
    results, wf_rows, series = [], [], {}
    GRID = list(itertools.product([20, 60], [1.5, 2.0, 2.5]))
    for name, struct in structs.items():
        built = {L: build(struct, L) for L in (20, 60)}
        runs = {(L, e): simulate(struct, built[L], e, L) for L, e in GRID}
        start = IDX[IDX.year >= 2011][0]
        for (L, e), out in runs.items():
            live = out.loc[start:]
            s = metrics.summarize(live["net"], CAP)
            g = metrics.summarize(live["gross"], CAP)
            t = metrics.trade_stats(live["signal"], live["net"], 1)
            results.append({"cost_mode": cost_mode, "structure": name, "L": L, "entry": e, "sharpe": s["sharpe"], "gross_sharpe": g["sharpe"],
                            "ann_ret": s["ann_return"], "mdd": s["max_drawdown"], "trades_yr": t["trades_per_year"],
                            "hold": t["avg_holding_days"], "win": t["win_rate"],
                            "cost_usd": float(live["cost"].sum()), "gross_usd": float(live["gross"].sum())})
            series[(name + ("" if cost_mode == "per_leg" else " [spread book]"), L, e)] = live["net"]
        pieces = []
        for y in range(2015, int(IDX.year.max()) + 1):
            tr = (IDX.year >= y - 3) & (IDX.year < y)
            best = max(runs, key=lambda k: np.nan_to_num(metrics.summarize(runs[k].loc[tr, "net"], CAP)["sharpe"], nan=-9))
            pieces.append(runs[best].loc[IDX.year == y, "net"])
        wf = pd.concat(pieces)
        series[(name + ("" if cost_mode == "per_leg" else " [spread book]"), "walk-forward")] = wf
        w = metrics.summarize(wf, CAP)
        sh = [r["sharpe"] for r in results if r["structure"] == name]
        wf_rows.append({"cost_mode": cost_mode, "structure": name, "wf_sharpe": w["sharpe"], "wf_ann": w["ann_return"], "wf_mdd": w["max_drawdown"],
                        "best_in_sample": max(sh), "median_grid": float(np.median(sh))})
        print(name, "done", flush=True)
    return pd.DataFrame(results), pd.DataFrame(wf_rows).set_index("structure"), series
