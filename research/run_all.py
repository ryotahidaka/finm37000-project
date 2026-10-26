r"""Run every strategy family on whatever metals are cached and write the trial tables.

From the repo root (cmd):

    set PYTHONPATH=src
    python research\run_all.py --cache data\databento --out output\lab

Writes all_trials.csv, leaders.csv (with deflated Sharpe), curve_grid.csv, curve_wf.csv,
series.parquet (daily net PnL per configuration) and meta.json to --out.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import lab
import lab_curve
import strategies as S

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--cache", default=os.path.join("data", "databento"), help="Databento parquet cache")
_ap.add_argument("--out", default=os.path.join("output", "lab"), help="output folder")
_args = _ap.parse_args()
CACHE, OUT = _args.cache, _args.out
t0 = time.time()
roots = lab.available_roots(CACHE)
settled = [r for r in roots if lab.load_settle(CACHE, r) is not None]
print("roots:", roots, "with settlement:", settled, flush=True)
legs = {r: lab.build_leg(CACHE, r) for r in roots}

trials = []      # every configuration tried: (family, name, sharpe) on 2014+ net PnL
series = {}      # name -> daily net PnL (USD)
tables = {}


def record(family, name, net, extra=None):
    net = net.loc["2014":]
    s = lab.summarize(net)
    trials.append({"family": family, "name": name, **s, **(extra or {})})
    series[name] = net


# ---- 1. pairs, formation/trading (GGR + Engle-Granger selection) ----------------------
pair_logs = {}
for z_mode in ("formation", "rolling60"):
    for gate in (None, "hurst"):
        name = f"Pairs top-3, z={z_mode}, gate={gate or 'none'}"
        tg, log = S.pairs_formation_trading(legs, z_mode=z_mode, gate=gate)
        out = S.simulate(tg, legs)
        record("pairs", name, out["net"], {"cost_usd": out["cost"].loc["2014":].sum()})
        pair_logs[name] = log
        print(name, round(trials[-1]["sharpe"], 3), flush=True)

# ---- 2. PCA residual stat-arb (needs 3+ metals) -----------------------------------------
if len(roots) >= 3:
    for win_ou in (60, 30):
        name = f"PCA residual s-score, OU window {win_ou}d"
        tg = S.pca_residual(legs, win_ou=win_ou)
        out = S.simulate(tg, legs)
        record("pca", name, out["net"], {"cost_usd": out["cost"].loc["2014":].sum()})
        print(name, round(trials[-1]["sharpe"], 3), flush=True)

# ---- 2b. Johansen basket of 3 ------------------------------------------------------------
if len(roots) >= 3:
    name = "Johansen 3-metal basket, formation z"
    tg, jlog = S.johansen_basket(legs)
    out = S.simulate(tg, legs)
    record("johansen", name, out["net"], {"cost_usd": out["cost"].loc["2014":].sum()})
    pair_logs[name] = jlog
    print(name, round(trials[-1]["sharpe"], 3), flush=True)

# ---- 2c. spread momentum, cross-sectional carry and momentum (trend side of the diagnosis) ----
for lb in (63, 126, 252):
    name = f"Spread momentum, all pairs, {lb}d"
    out = S.simulate(S.spread_momentum(legs, lookback=lb), legs)
    record("trend/carry", name, out["net"], {"cost_usd": out["cost"].loc["2014":].sum()})
    print(name, round(trials[-1]["sharpe"], 3), flush=True)
if len(roots) >= 4:
    for n_side in (1, 2):
        name = f"Cross-sectional carry, long/short {n_side}"
        out = S.simulate(S.xs_carry(legs, n_side=n_side), legs)
        record("trend/carry", name, out["net"], {"cost_usd": out["cost"].loc["2014":].sum()})
        print(name, round(trials[-1]["sharpe"], 3), flush=True)
        name = f"Cross-sectional momentum 12-1m, long/short {n_side}"
        out = S.simulate(S.xs_momentum(legs, n_side=n_side), legs)
        record("trend/carry", name, out["net"], {"cost_usd": out["cost"].loc["2014":].sum()})
        print(name, round(trials[-1]["sharpe"], 3), flush=True)

# ---- 3. curve structures (calendars, butterflies, carry vs carry) -----------------------
res1, wf1, cser1 = lab_curve.run(CACHE, roots, "per_leg")
res2, wf2, cser2 = lab_curve.run(CACHE, roots, "spread_book")
res, wf, cser = pd.concat([res1, res2]), pd.concat([wf1, wf2]), {**cser1, **cser2}
tables["curve_grid"] = res
tables["curve_wf"] = wf
for k, v in cser.items():
    name = " ".join(str(x) for x in k)
    if k[1] == "walk-forward":
        record("curve walk-forward", name, v)
    else:
        record("curve grid", name, v)

# ---- 4. original GC/SI results (already run) count as trials ------------------------------
prior = {"GC/SI baseline (fixed)": -0.21, "GC/SI walk-forward": -0.85, "regime filter": 0.04,
         "short horizon": -0.28, "Kalman": -0.13, "log ratio dollar-neutral": -0.09,
         "regime + short": -0.45, "Kalman (report)": -0.10, "dollar-neutral (report)": 0.15}
n_prior = len(prior)

# ---- deflated Sharpe for the leaders ------------------------------------------------------
T = pd.DataFrame(trials)
all_sharpes = list(T["sharpe"].dropna()) + list(prior.values())
T = T.sort_values("sharpe", ascending=False)
leaders = T.head(8).copy()
dsr = []
for _, row in leaders.iterrows():
    d = lab.deflated_sharpe(series[row["name"]] / lab.CAP, all_sharpes)
    dsr.append(d["dsr"])
leaders["deflated_sharpe_prob"] = dsr
leaders["sr0_expected_max_under_null"] = lab.deflated_sharpe(series[leaders.iloc[0]["name"]] / lab.CAP,
                                                              all_sharpes)["sr0_ann"]

os.makedirs(OUT, exist_ok=True)
T.to_csv(f"{OUT}/all_trials.csv", index=False)
leaders.to_csv(f"{OUT}/leaders.csv", index=False)
res.to_csv(f"{OUT}/curve_grid.csv", index=False)
wf.to_csv(f"{OUT}/curve_wf.csv")
pd.DataFrame({k: v for k, v in series.items()}).to_parquet(f"{OUT}/series.parquet")
json.dump({"roots": roots, "settled": settled, "n_trials": len(all_sharpes),
           "pair_logs": pair_logs, "minutes": (time.time() - t0) / 60},
          open(f"{OUT}/meta.json", "w"), indent=1, default=str)
print(f"done in {(time.time() - t0) / 60:.1f} min; {len(all_sharpes)} trials")
print(leaders[["family", "name", "sharpe", "ann_ret", "mdd", "deflated_sharpe_prob"]].round(3).to_string())
