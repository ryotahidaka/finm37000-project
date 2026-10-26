"""Run every structure and screen on whatever roots are cached.

python run_universe.py <cache_dir> <out_dir> [--start 2014]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import core
import spread_strategies as st

# (name, class, kind, spec). Fixed: contract weights per unit. Hedged: (A, B).
STRUCTURES = [
    # venue: same underlying, different contract or venue
    ("NG vs HH (physical vs financial)", "venue", "fixed", {"NG": 1, "HH": -1}),
    ("NYMEX WTI vs ICE WTI", "venue", "fixed", {"CL": 1, "WTI": -1}),
    ("NYMEX Brent vs ICE Brent", "venue", "fixed", {"BZ": 1, "BRN": -1}),
    # processing margins
    ("3-2-1 crack", "processing", "fixed", {"CL": -3, "RB": 2, "HO": 1}),
    ("Gasoline crack", "processing", "fixed", {"CL": -1, "RB": 1}),
    ("Diesel crack", "processing", "fixed", {"CL": -1, "HO": 1}),
    ("Brent gasoil crack (ICE)", "processing", "fixed", {"BRN": -3, "GO": 4}),
    ("Soybean board crush", "processing", "fixed", {"ZS": -10, "ZM": 11, "ZL": 9}),
    ("Corn ethanol margin", "processing", "fixed", {"ZC": -3, "CU": 1}),
    ("Sugar white premium", "processing", "fixed", {"SB": -1, "WS": 1}),
    ("Oil share (soy oil vs meal)", "processing", "hedged", ("ZL", "ZM")),
    ("Sugar vs ethanol", "processing", "hedged", ("SB", "CU")),
    # substitutes and locations
    ("WTI vs Brent (NYMEX)", "substitute", "fixed", {"CL": 1, "BZ": -1}),
    ("WTI (NYMEX) vs Brent (ICE)", "substitute", "fixed", {"CL": 1, "BRN": -1}),
    ("Crude vs natural gas", "substitute", "hedged", ("CL", "NG")),
    ("Brent vs cocoa (control: no economic link)", "substitute", "hedged", ("BRN", "CC")),
    ("Gasoline vs diesel", "substitute", "fixed", {"RB": 1, "HO": -1}),
    ("US vs European gas", "substitute", "hedged", ("NG", "TTF")),
    ("NY vs London cocoa", "substitute", "fixed", {"CC": 1, "LCC": -1}),
    ("Arabica vs robusta", "substitute", "hedged", ("KC", "RC")),
    ("KC vs Chicago wheat", "substitute", "fixed", {"KE": 1, "ZW": -1}),
    ("Corn vs wheat", "substitute", "fixed", {"ZC": 1, "ZW": -1}),
    ("Soy oil vs canola", "substitute", "hedged", ("ZL", "RS")),
    ("Soy oil vs heating oil (biodiesel)", "substitute", "hedged", ("ZL", "HO")),
    ("Sugar vs gasoline", "substitute", "hedged", ("SB", "RB")),
    ("Gold vs silver", "substitute", "hedged", ("GC", "SI")),
    ("Platinum vs palladium", "substitute", "hedged", ("PL", "PA")),
    ("Corn vs soybeans", "substitute", "hedged", ("ZS", "ZC")),
]
CAL_ROOTS = ["CL", "BZ", "BRN", "NG", "RB", "HO", "ZS", "ZM", "ZL", "ZC", "ZW", "KE", "SB", "CC", "KC", "GC", "HG"]
UNIVERSES = {
    "energy": ["CL", "BZ", "RB", "HO", "NG", "HH", "CU", "BRN", "GO", "WTI", "TTF"],
    "soy and grains": ["ZS", "ZM", "ZL", "ZC", "ZW", "KE", "RS", "CU"],
    "softs": ["CC", "LCC", "KC", "RC", "SB", "WS", "CT"],
    "metals": ["GC", "SI", "PL", "PA", "HG"],
    "cross-commodity": None,  # everything available
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("--start", default="2014")
    ap.add_argument("--skip-screen", action="store_true")
    ap.add_argument("--settle-min", type=float, default=0.95,
                    help="use settlements (and drop days without one) when they cover this share of rows")
    ap.add_argument("--exclude", nargs="*", default=[], help="roots to leave out (e.g. too illiquid)")
    a = ap.parse_args()
    t0 = time.time()
    core.SETTLE_MIN = a.settle_min
    names = [n for n in core.available(a.cache) if n in core.SPECS and n not in a.exclude]
    print("available:", names, flush=True)
    legs = core.load_legs(a.cache, names)
    have = set(legs)
    trials, series, summary, logs = [], {}, [], {}

    def rec(family, cls, name, res, extra=None):
        res = res[res.index >= pd.Timestamp(a.start, tz="UTC")]
        if len(res) < 250 or res["net"].abs().sum() == 0:
            return None
        s = core.summarize(res["net"])
        g = core.summarize(res["gross"])
        row = {"family": family, "class": cls, "name": name, **s, "gross_sharpe": g["sharpe"],
               "cost_usd": float(res["cost"].sum()), "gross_usd": float(res["gross"].sum()),
               "start": str(res.index[0].date()), **(extra or {})}
        trials.append(row)
        series[name] = res["net"]
        return row

    # ---- 1. structures -------------------------------------------------------------------
    structs = list(STRUCTURES) + [(f"{r} calendar (M1-M2)", "calendar", "fixed", {r: 1, f"{r}@next": -1})
                                  for r in CAL_ROOTS]
    LISTED = {"Gasoline crack", "Diesel crack", "3-2-1 crack", "Soybean board crush", "NG vs HH (physical vs financial)",
              "Brent gasoil crack (ICE)", "Sugar white premium", "KC vs Chicago wheat", "Corn vs wheat", "NYMEX WTI vs ICE WTI",
              "NYMEX Brent vs ICE Brent", "WTI vs Brent (NYMEX)"}
    structs = structs + [(n + " [spread book]", c, k, s) for n, c, k, s in structs
                         if k == "fixed" and (c == "calendar" or n in LISTED)]
    for name, cls, kind, spec in structs:
        need = {k.split("@")[0] for k in spec} if kind == "fixed" else set(spec)
        if not need <= have or any(legs[n].fx.isna().all() for n in need):
            continue
        try:
            sigl = legs[next(iter(spec))].carry if cls == "calendar" else None
            obj = st.Fixed(name, cls, spec, legs, sigl, book=name.endswith("[spread book]")) if kind == "fixed" else st.Hedged(name, cls, *spec, legs)
        except Exception as exc:
            print("  fail", name, exc, flush=True)
            continue
        nets, grosses = {}, {}
        for L, e in st.GRID:
            r = obj.run(L, e)
            row = rec("structure grid", cls, f"{name} | z{L} e{e}", r, {"structure": name, "L": L, "entry": e})
            if row:
                nets[(L, e)] = r["net"]
                grosses[(L, e)] = r["gross"]
        if not nets:
            continue
        wf, wfg = st.walk_forward(nets, others=grosses)
        row = rec("structure walk-forward", cls, f"{name} | walk-forward",
                  pd.DataFrame({"net": wf, "gross": wfg.reindex(wf.index).fillna(0),
                                "cost": (wfg.reindex(wf.index).fillna(0) - wf)}), {"structure": name})
        lvl = obj.sig_level if kind == "fixed" else obj.level
        lvl = lvl[lvl.index >= pd.Timestamp(a.start, tz="UTC")]
        try:
            from statsmodels.tsa.stattools import adfuller
            adf = adfuller(lvl.dropna().to_numpy(), maxlag=10)[1]
        except Exception:
            adf = np.nan
        grid = [t for t in trials if t.get("structure") == name and t["family"] == "structure grid"]
        summary.append({"structure": name, "class": cls, "legs": " ".join(sorted(need)),
                        "start": str(lvl.index[0].date()) if len(lvl) else "",
                        "half_life_days": core.half_life(lvl), "hurst": core.hurst(lvl.to_numpy()),
                        "adf_p": adf,
                        "best_grid": max(t["sharpe"] for t in grid), "median_grid": float(np.median([t["sharpe"] for t in grid])),
                        "median_grid_gross": float(np.median([t["gross_sharpe"] for t in grid])),
                        "wf_sharpe": row["sharpe"] if row else np.nan, "wf_gross": row["gross_sharpe"] if row else np.nan,
                        "wf_ann": row["ann_ret"] if row else np.nan, "wf_mdd": row["mdd"] if row else np.nan})
        print(f"{name:40s} best {summary[-1]['best_grid']:+.2f} wf {summary[-1]['wf_sharpe']:+.2f} "
              f"hl {summary[-1]['half_life_days']:.0f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 1b. portfolios of the walk-forward structures (no selection: every structure in) -----
    wf_names = [t["name"] for t in trials if t["family"] == "structure walk-forward"]
    def book(name_filter, label, cls):
        cols = [n for n in wf_names if name_filter(n)]
        if len(cols) < 2:
            return
        W = pd.DataFrame({n: series[n] for n in cols}).fillna(0)
        vol = W.rolling(126, min_periods=60).std().shift(1)
        w = (1 / vol).replace([np.inf], np.nan)
        w = w.div(w.sum(1), axis=0).fillna(0)
        port = (W * w).sum(1)
        scale = st.DAILY_TGT / port.rolling(126, min_periods=60).std().shift(1)
        net = (port * scale.clip(upper=len(cols))).fillna(0)
        rec("portfolio", cls, f"{label} ({len(cols)} structures)", pd.DataFrame({"net": net, "gross": net, "cost": 0 * net}))
        print(label, round(trials[-1]["sharpe"], 2), flush=True)
    realistic = lambda n: ("[spread book]" in n) or not any(n.replace(" | walk-forward", "") + " [spread book]" in m for m in wf_names)
    book(realistic, "All structures, risk-parity book", "portfolio")
    for c in ["venue", "processing", "substitute", "calendar"]:
        cl = {t["name"] for t in trials if t["class"] == c and t["family"] == "structure walk-forward"}
        book(lambda n, cl=cl: n in cl and realistic(n), f"{c.capitalize()} book", "portfolio")

    # ---- 2. formation/trading screens and PCA by universe ----------------------------------
    if not a.skip_screen:
        for uni, members in UNIVERSES.items():
            mem = [n for n in (members or [n for n in names if core.SPECS[n].group != "fx"]) if n in have]
            if len(mem) < 3:
                continue
            # the cross-commodity screen only uses roots with long history (avoid starting in 2019)
            if members is None:
                mem = [n for n in mem if legs[n].level.dropna().index[0].year <= 2012]
            for kind in ("pair", "triple"):
                if kind == "triple" and len(mem) > 14:
                    continue
                for zm in ("formation", "rolling"):
                    nm = f"{uni}: top {kind}s, z={zm}"
                    try:
                        r, lg = st.formation_trading(legs, mem, kind=kind, z_mode=zm)
                    except Exception as exc:
                        print("  fail", nm, exc, flush=True)
                        continue
                    rec("screen", uni, nm, r)
                    logs[nm] = lg
                    print(nm, round(trials[-1]["sharpe"], 2) if trials and trials[-1]["name"] == nm else "-", f"({time.time() - t0:.0f}s)", flush=True)
            for w in (60, 30):
                nm = f"{uni}: PCA residual s-score, OU {w}d"
                try:
                    r = st.pca_residual(legs, mem, win_ou=w, n_fac=1 if len(mem) < 6 else 2)
                except Exception as exc:
                    print("  fail", nm, exc, flush=True)
                    continue
                rec("pca", uni, nm, r)
                print(nm, round(trials[-1]["sharpe"], 2) if trials and trials[-1]["name"] == nm else "-", f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 3. multiple testing ------------------------------------------------------------
    T = pd.DataFrame(trials)
    allsh = T["sharpe"].dropna().tolist()
    T["psr"] = np.nan
    T["dsr_all"] = np.nan
    T["dsr_family"] = np.nan
    for i, r in T.iterrows():
        ret = series[r["name"]] / core.CAP
        d = core.deflated_sharpe(ret, allsh)
        if r["family"] == "structure grid":
            fam = T.loc[T["structure"] == r["structure"], "sharpe"].tolist()
        else:
            fam = T.loc[T["family"] == r["family"], "sharpe"].tolist()
        T.loc[i, ["psr", "dsr_all", "dsr_family"]] = [d["psr"], d["dsr"], core.deflated_sharpe(ret, fam)["dsr"] if len(fam) > 1 else d["psr"]]
    import os
    os.makedirs(a.out, exist_ok=True)
    T.sort_values("sharpe", ascending=False).to_csv(f"{a.out}/trials.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{a.out}/structures.csv", index=False)
    pd.DataFrame(series).to_parquet(f"{a.out}/series.parquet")
    sr0 = core.deflated_sharpe(pd.Series(np.random.default_rng(0).normal(size=1000)), allsh)["sr0_ann"]
    json.dump({"roots": names, "n_trials": len(allsh), "sr0_all": sr0, "start": a.start, "logs": logs,
               "minutes": (time.time() - t0) / 60}, open(f"{a.out}/meta.json", "w"), indent=1, default=str)
    print(f"done: {len(T)} trials in {(time.time() - t0) / 60:.1f} min, SR0 {sr0:.2f}")
    cols = ["family", "class", "name", "sharpe", "gross_sharpe", "ann_ret", "mdd", "dsr_family"]
    print(T.sort_values("sharpe", ascending=False)[cols].head(15).round(3).to_string())


if __name__ == "__main__":
    main()
