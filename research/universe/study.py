"""The cross-commodity study: every test, table and figure behind docs/STUDY.md, from the data
already in the Databento cache. Downloads nothing.

    python research/universe/study.py --cache data/databento --runs output/study --docs docs/study

Steps
  1. data inventory and quality per product (coverage, price source, roll behaviour)
  2. statistical tests per product: ADF and KPSS on log prices, ADF on returns, Hurst, vol, tails
  3. structures run (run_universe.py) twice: last-trade prices and settlement prices
  4. every pair and calendar with four strategy families and two cost models (lab3.py)
  5. regime study: NBER recession, named stress windows, ex-ante high volatility, calm
  6. figures and summary.json with every headline number quoted in the report
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402
import regimes  # noqa: E402

plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 8})
NAVY, GOLD, SLATE, BLUE, RUST = "#14213D", "#B8862B", "#6B7B8C", "#2E6FBA", "#C8642B"
START = pd.Timestamp("2014-01-01", tz="UTC")

# headline strategies followed through the regime study (names as written by the runs)
FOLLOW = {
    "Gold vs silver (walk-forward)": ("last", "Gold vs silver | walk-forward"),
    "Metals PCA residual, 30d": ("last", "metals: PCA residual s-score, OU 30d"),
    "WTI vs ICE Brent (walk-forward)": ("last", "WTI (NYMEX) vs Brent (ICE) | walk-forward"),
    "Palladium mean reversion (Kalman)": ("lab3", "PA outright (control) | WF kalman | spread"),
    "Platinum mean reversion (Kalman)": ("lab3", "PL outright (control) | WF kalman | spread"),
    "Silver calendar, worked as one order": ("lab3", "SI calendar | WF meta | spread"),
    "Brent vs cocoa (control)": ("last", "Brent vs cocoa (control: no economic link) | walk-forward"),
}


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def inventory(cache, names):
    rows, legs = [], {}
    for n in names:
        df = core.load(cache, n)
        lg = core.build_leg(df, n)
        legs[n] = lg
        a = lg.active.dropna()
        lv = lg.level.dropna()
        spec = core.SPECS[n]
        rows.append({"product": n, "name": spec.label, "exchange": {"": "CME", "IFUS": "ICE US", "IFEU": "ICE Europe",
                                                                     "NDEX": "ICE Endex"}[spec.folder],
                     "first": str(lv.index[0].date()), "last": str(lv.index[-1].date()), "days": len(lv),
                     "contracts": df["ym"].nunique(),
                     "price": df.attrs.get("price_source", "last trade"),
                     "oi_coverage": float(df["oi"].notna().mean()),
                     "rolls_per_year": float((a != a.shift()).sum() / (len(a) / 252)),
                     "months_held": " ".join(core.MONTHS[int(m)] for m in sorted(set(a % 12)))})
    return pd.DataFrame(rows), legs


def stat_tests(legs):
    from scipy.stats import kurtosis, skew
    from statsmodels.tsa.stattools import adfuller, kpss
    rows = []
    for n, lg in legs.items():
        lp = lg.logpx.where(lg.level.notna()).dropna()
        lp = lp[lp.index >= START] if lp.index[-1] > START else lp
        r = lg.logret.reindex(lp.index).dropna()
        try:
            kp = kpss(lp.to_numpy(), regression="c", nlags="auto")[1]
        except Exception:
            kp = np.nan
        rows.append({"product": n, "adf_p_logprice": adfuller(lp.to_numpy(), maxlag=10)[1],
                     "kpss_p_logprice": kp, "adf_p_returns": adfuller(r.to_numpy(), maxlag=10)[1],
                     "hurst_logprice": core.hurst(lp.to_numpy()), "ann_vol": r.std() * np.sqrt(252),
                     "skew": skew(r), "excess_kurtosis": kurtosis(r), "autocorr_1d": r.autocorr(1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/databento")
    ap.add_argument("--runs", default="output/study")
    ap.add_argument("--docs", default="docs/study")
    ap.add_argument("--exclude", nargs="*", default=["HH"],
                    help="products too illiquid to use (HH has no settlement file and rare trades)")
    ap.add_argument("--skip-runs", action="store_true", help="reuse existing runs in --runs")
    a = ap.parse_args()
    os.makedirs(a.runs, exist_ok=True)
    os.makedirs(a.docs, exist_ok=True)
    py = sys.executable
    last, settle, lab = (os.path.join(a.runs, x) for x in ("structures_last", "structures_settle", "pairs_lab3"))
    if not a.skip_runs:
        run([py, os.path.join(HERE, "run_universe.py"), a.cache, last, "--exclude", *a.exclude])
        run([py, os.path.join(HERE, "run_universe.py"), a.cache, settle, "--exclude", *a.exclude, "--settle-min", "0.5"])
        run([py, os.path.join(HERE, "lab3.py"), a.cache, lab, "--exclude", *a.exclude])

    names = [n for n in core.available(a.cache) if n not in a.exclude]
    inv, legs = inventory(a.cache, names)
    inv.to_csv(f"{a.docs}/data_inventory.csv", index=False)
    tradable = [n for n in names if core.SPECS[n].group != "fx"]
    tests = stat_tests({n: legs[n] for n in tradable})
    tests.to_csv(f"{a.docs}/stat_tests.csv", index=False)

    # copy the result tables the report quotes
    S_last = pd.read_parquet(f"{last}/series.parquet")
    S_lab = pd.read_parquet(f"{lab}/series.parquet")
    st_last = pd.read_csv(f"{last}/structures.csv")
    st_set = pd.read_csv(f"{settle}/structures.csv")
    sens = st_last[["structure", "class", "half_life_days", "hurst", "adf_p", "wf_sharpe"]].merge(
        st_set[["structure", "wf_sharpe"]], on="structure", how="left", suffixes=("_last_trade", "_settlement"))
    sens.to_csv(f"{a.docs}/structures.csv", index=False)
    tr = pd.read_csv(f"{last}/trials.csv")
    tr[tr["family"] != "structure grid"].to_csv(f"{a.docs}/structures_walkforward.csv", index=False)
    rel = pd.read_csv(f"{lab}/relationships.csv")
    rel.to_csv(f"{a.docs}/relationships.csv", index=False)
    wf = pd.read_csv(f"{lab}/walkforward.csv")
    wf.to_csv(f"{a.docs}/pairs_walkforward.csv", index=False)
    fam = wf[wf["kind"].isin(["pair", "calendar", "outright"])].groupby(["kind", "family", "cost_mode"])["sharpe"] \
        .agg(spreads="count", median="median", share_positive=lambda s: (s > 0).mean()).reset_index()
    fam.to_csv(f"{a.docs}/families.csv", index=False)
    pv = wf[(wf["kind"] == "calendar") & (wf["family"] == "meta")].pivot(index="spread", columns="cost_mode", values="sharpe")
    cal = pd.DataFrame({"calendar": pv.index, "per_leg": pv["leg"].values, "one_order": pv["spread"].values})
    cal["change"] = cal["one_order"] - cal["per_leg"]
    cal.to_csv(f"{a.docs}/cost_models_calendars.csv", index=False)

    # regime study
    idx = S_lab.index[S_lab.index >= START]
    basket = pd.DataFrame({n: legs[n].logret for n in ["GC", "SI", "PL", "PA", "HG", "CL", "NG"] if n in legs})
    labels = regimes.label_regimes(idx, basket)
    reg_rows, follow = [], {}
    for k, (src, nm) in FOLLOW.items():
        S = S_last if src == "last" else S_lab
        if nm not in S:
            continue
        s = S[nm].reindex(idx)
        s = s[s.index >= s.fillna(0).ne(0).idxmax()].fillna(0)
        follow[k] = s
        for _, r in regimes.by_regime(s, labels, core.CAP).iterrows():
            reg_rows.append({"strategy": k, **r.to_dict()})
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(f"{a.docs}/regimes.csv", index=False)
    R = pd.DataFrame({n: legs[n].logret for n in ["GC", "SI", "CL", "BRN", "PL", "PA"] if n in legs}).reindex(idx)
    corr_rows = []
    for x, y in [("GC", "SI"), ("CL", "BRN"), ("PL", "PA")]:
        if x in R and y in R:
            for g in labels.unique():
                m = labels == g
                corr_rows.append({"pair": f"{x}/{y}", "regime": g, "corr": R[x][m].corr(R[y][m]), "days": int(m.sum())})
    pd.DataFrame(corr_rows).to_csv(f"{a.docs}/regime_correlations.csv", index=False)

    # ---------------- figures ----------------
    # 1. relationships
    order = [n for n in ["GC", "SI", "PL", "PA", "HG", "CL", "BRN", "NG", "ZS", "CC"] if n in tradable]
    C = pd.DataFrame(np.nan, index=order, columns=order)
    P = C.copy()
    for r in rel.itertuples():
        if r.a in order and r.b in order:
            C.loc[r.a, r.b] = C.loc[r.b, r.a] = r.corr
            P.loc[r.a, r.b] = P.loc[r.b, r.a] = r.eg_p
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for k, (D, t, cm, lo, hi) in enumerate([(C, "Daily return correlation", "RdBu_r", -1, 1),
                                            (P, "Engle-Granger p-value (low = cointegrated)", "viridis_r", 0, 1)]):
        im = ax[k].imshow(D.to_numpy(float), cmap=cm, vmin=lo, vmax=hi)
        ax[k].set_xticks(range(len(order)), order)
        ax[k].set_yticks(range(len(order)), order)
        ax[k].grid(False)
        for i in range(len(order)):
            for j in range(len(order)):
                v = D.iloc[i, j]
                if pd.notna(v):
                    ax[k].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                               color="white" if (abs(v) > 0.6 if k == 0 else v < 0.4) else "black")
        ax[k].set_title(t)
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.savefig(f"{a.docs}/fig_relationships.png", bbox_inches="tight")
    plt.close(fig)

    # 2. half-life vs walk-forward result by class
    base = st_last[~st_last["structure"].str.contains("spread book")]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for c, col in [("calendar", RUST), ("substitute", NAVY)]:
        d = base[base["class"] == c]
        ax.scatter(d["half_life_days"].clip(1, 3000), d["wf_sharpe"], color=col, s=24, label=c)
        for r in d.itertuples():
            ax.annotate(r.structure.replace(" (M1-M2)", "").replace(" (control: no economic link)", " (control)"),
                        (min(max(r.half_life_days, 1), 3000), r.wf_sharpe), fontsize=5.5, xytext=(3, 2),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(0, color="#A9B4C2", lw=1)
    ax.set_xlabel("half-life of the spread, days (log scale)")
    ax.set_ylabel("walk-forward net Sharpe")
    ax.legend(fontsize=7)
    ax.set_title("Tighter spreads revert faster, but neither class pays after costs")
    fig.savefig(f"{a.docs}/fig_halflife_vs_result.png", bbox_inches="tight")
    plt.close(fig)

    # 3. strategy families
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    groups = [("pair", f) for f in ["mr", "mom", "regime", "kalman", "meta"]] + [("calendar", f) for f in ["mr", "mom", "regime", "meta"]]
    data = [wf[(wf["kind"] == k) & (wf["family"] == f) & (wf["cost_mode"] == "spread")]["sharpe"].dropna() for k, f in groups]
    ax.boxplot(data, tick_labels=[f"{k}\n{f}" for k, f in groups], showfliers=True, flierprops={"markersize": 2})
    ax.axhline(0, color="#A9B4C2", lw=1)
    ax.set_ylabel("walk-forward net Sharpe")
    ax.set_title("Strategy families across all 45 pairs and 10 calendars (one-order costs)")
    fig.savefig(f"{a.docs}/fig_families.png", bbox_inches="tight")
    plt.close(fig)

    # 4. cost models on calendars
    c2 = cal.sort_values("one_order")
    fig, ax = plt.subplots(figsize=(6.5, 3))
    y = np.arange(len(c2))
    ax.barh(y - 0.2, c2["per_leg"], height=0.4, color=SLATE, label="every leg crosses")
    ax.barh(y + 0.2, c2["one_order"], height=0.4, color=GOLD, label="worked as one spread order")
    ax.set_yticks(y, c2["calendar"])
    ax.axvline(0, color="#A9B4C2", lw=1)
    ax.set_xlabel("walk-forward net Sharpe")
    ax.legend(fontsize=7)
    ax.set_title("Calendar spreads under the two cost models")
    fig.savefig(f"{a.docs}/fig_cost_models.png", bbox_inches="tight")
    plt.close(fig)

    # 5. regimes: cumulative PnL with recession and stress shading
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cols = [NAVY, GOLD, SLATE, BLUE, RUST, "#7A9E7E", "#A9B4C2"]
    for (k, s), c in zip(follow.items(), cols):
        ax.plot(s.index, s.cumsum() / 1e3, color=c, lw=1.1, label=k)
    regimes.add_recession_bars(ax)
    ax.set_ylabel("cumulative net PnL ($k on $1m)")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax.set_title("Grey: NBER recession. Tan: commodity stress windows")
    fig.savefig(f"{a.docs}/fig_regimes.png", bbox_inches="tight")
    plt.close(fig)

    # 6. regime table as heatmap
    pvr = reg.pivot(index="strategy", columns="regime", values="sharpe")
    pvr = pvr[[c for c in ["calm", "high volatility"] + sorted(c for c in pvr.columns if c.startswith("stress")) + ["NBER recession"] if c in pvr.columns]]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(pvr.to_numpy(float), cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(pvr.shape[1]), [c.replace("stress: ", "") for c in pvr.columns], rotation=15, fontsize=6.5)
    ax.set_yticks(range(pvr.shape[0]), pvr.index, fontsize=7)
    ax.grid(False)
    for i in range(pvr.shape[0]):
        for j in range(pvr.shape[1]):
            v = pvr.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Sharpe by regime (short windows: read the sign, not the size)")
    fig.savefig(f"{a.docs}/fig_regime_table.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- summary for the report ----------------
    meta_last = json.load(open(f"{last}/meta.json"))
    meta_lab = json.load(open(f"{lab}/meta.json"))

    def wfrow(df, name):
        r = df[df["name"] == name]
        return r.iloc[0].to_dict() if len(r) else {}

    summary = {
        "products": names, "tradable": tradable,
        "n_trials_structures": meta_last["n_trials"], "n_configs_pairs": meta_lab["n_configs"],
        "n_wf_series_pairs": meta_lab["n_wf"], "sr0_structures": meta_last["sr0_all"],
        "pairs_eg_p_below_5pct": int((rel["eg_p"] < 0.05).sum()), "pairs_total": len(rel),
        "max_stable_share": float(rel["stable_share"].max()),
        "corr_CL_BRN": float(rel.query("a=='CL' and b=='BRN'")["corr"].iloc[0]) if len(rel.query("a=='CL' and b=='BRN'")) else None,
        "corr_GC_SI": float(rel.query("a=='GC' and b=='SI'")["corr"].iloc[0]),
        "median_half_life": base.groupby("class")["half_life_days"].median().to_dict(),
        "follow": {k: {**core.summarize(s), "start": str(s.index[0].date())} for k, s in follow.items()},
        "wti_brent": wfrow(tr, "WTI (NYMEX) vs Brent (ICE) | walk-forward"),
        "families": fam.to_dict("records"),
        "calendars_cost": cal.to_dict("records"),
        "regimes": reg.to_dict("records"),
    }
    json.dump(summary, open(f"{a.docs}/summary.json", "w"), indent=1, default=float)
    print("study written to", a.docs)


if __name__ == "__main__":
    main()
