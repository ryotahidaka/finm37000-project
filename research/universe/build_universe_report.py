"""Universe lab output folder -> one self-contained HTML page (figures embedded)."""
import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT, HTML = sys.argv[1], sys.argv[2]
ALT = sys.argv[3] if len(sys.argv) > 3 else None   # same run on the other price definition
T = pd.read_csv(f"{OUT}/trials.csv")
SUM = pd.read_csv(f"{OUT}/structures.csv")
S = pd.read_parquet(f"{OUT}/series.parquet")
meta = json.load(open(f"{OUT}/meta.json"))
extra = json.load(open(f"{OUT}/extra.json")) if os.path.exists(f"{OUT}/extra.json") else {}

CLS_COL = {"venue": "#2E6FBA", "processing": "#B8862B", "substitute": "#14213D", "calendar": "#C8642B"}
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


figs = {}
# 1. taxonomy: gross vs net Sharpe by class (median grid), one dot per structure
if len(SUM):
    base = SUM[~SUM["structure"].str.endswith("[spread book]")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    order = [c for c in ["venue", "processing", "substitute", "calendar"] if c in set(base["class"])]
    for i, c in enumerate(order):
        d = base[base["class"] == c]
        jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(d))
        ax[0].scatter(np.full(len(d), i) + jit, d["median_grid_gross"], color=CLS_COL[c], s=22)
        ax[0].scatter(np.full(len(d), i) + jit, d["wf_sharpe"], facecolors="none", edgecolors=CLS_COL[c], s=22)
    ax[0].axhline(0, color="#A9B4C2", lw=1)
    ax[0].set_xticks(range(len(order)), order)
    ax[0].set_ylabel("Sharpe")
    ax[0].set_title("Filled: median gross over the grid. Hollow: walk-forward net")
    hl = base["half_life_days"].replace(np.inf, np.nan)
    for c in order:
        d = base[base["class"] == c]
        ax[1].scatter(np.clip(hl[d.index], 1, 2000), d["wf_sharpe"], color=CLS_COL[c], s=22, label=c)
    ax[1].set_xscale("log")
    ax[1].axhline(0, color="#A9B4C2", lw=1)
    ax[1].set_xlabel("half-life of the spread (days, log)")
    ax[1].set_ylabel("walk-forward net Sharpe")
    ax[1].legend(fontsize=7)
    ax[1].set_title("Faster mean reversion vs what survives costs")
    figs["tax"] = b64(fig)

# 2. cumulative PnL of the five best distinct ideas
best = []
for _, r in T.sort_values("sharpe", ascending=False).iterrows():
    key = r["structure"] if isinstance(r.get("structure"), str) else r["name"]
    key = key.replace(" [spread book]", "")
    if key not in [k for k, _ in best] and r["family"] != "structure grid":
        best.append((key, r["name"]))
    if len(best) == 5:
        break
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, (_, nm) in enumerate(best):
    s = S[nm].dropna().cumsum() / 1e3
    ax.plot(s.index, s, lw=1.2, label=nm[:60], color=["#14213D", "#B8862B", "#6B7B8C", "#2E6FBA", "#C8642B"][i])
ax.set_ylabel("cumulative net PnL ($k on $1m)")
ax.legend(fontsize=7, loc="upper left")
ax.set_title("Best five ideas, honest versions (walk-forward or fixed rules)")
figs["pnl"] = b64(fig)


def table(df, cols, fmt, heads=None):
    h = "<tr>" + "".join(f"<th>{(heads or {}).get(c, c)}</th>" for c in cols) + "</tr>"
    rows = ["<tr>" + "".join(f"<td>{fmt.get(c, lambda x: x)(r[c])}</td>" for c in cols) + "</tr>"
            for _, r in df.iterrows()]
    return f"<table>{h}{''.join(rows)}</table>"


num = lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "n/a"
pct = lambda x: f"{100 * x:.1f}%" if pd.notna(x) else ""
prob = lambda x: f"{100 * x:.0f}%" if pd.notna(x) else ""
days = lambda x: ("&gt;1000" if x > 1000 else f"{x:.0f}") if pd.notna(x) else ""

sum_tbl = SUM.sort_values(["class", "wf_sharpe"], ascending=[True, False])
honest = T[T["family"] != "structure grid"].sort_values("sharpe", ascending=False).head(15)

alt_html = ""
if ALT:
    A = pd.read_csv(f"{ALT}/structures.csv").set_index("structure")
    B = SUM.set_index("structure")
    C = pd.DataFrame({"class": B["class"], "main": B["wf_sharpe"], "alt": A["wf_sharpe"].reindex(B.index)}).reset_index()
    C["gap"] = (C["main"] - C["alt"]).abs()
    alt_html = ("<h2>Does the price definition change the answer?</h2><div class='card'>"
                + table(C.sort_values(["class", "structure"]), ["class", "structure", "main", "alt", "gap"],
                        {"main": num, "alt": num, "gap": num},
                        {"main": "walk-fwd net, " + extra.get("main_label", "this run"), "alt": "walk-fwd net, " + extra.get("alt_label", "other run"), "gap": "difference"})
                + f"<p>{extra.get('alt_note', '')}</p></div>")

html = f"""<title>Commodity Spread Lab</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
:root{{--bg:#F7F5EF;--ink:#14213D;--muted:#4A5568;--card:#FFFDF8;--line:#E2DCCD;--accent:#8A6416}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#10192E;--ink:#EEF1F5;--muted:#B5C0CE;--card:#18233D;--line:#2A3858;--accent:#D9B25A;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#10192E;--ink:#EEF1F5;--muted:#B5C0CE;--card:#18233D;--line:#2A3858;--accent:#D9B25A;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:16px/1.55 "IBM Plex Sans",system-ui,sans-serif;margin:0}}
main{{max-width:1040px;margin:0 auto;padding-block:32px 64px;padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px))}}
h1{{font-family:"EB Garamond",Georgia,serif;font-weight:500;font-size:40px;margin:0 0 4px}}
h2{{font-family:"EB Garamond",Georgia,serif;font-weight:500;font-size:28px;margin:40px 0 8px}}
.eyebrow{{color:var(--accent);letter-spacing:.12em;text-transform:uppercase;font-size:12px}}
.lead{{font-size:18px;color:var(--muted)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}}
img{{width:100%;height:auto;border-radius:8px;background:#fff}}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@500&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
<p class="eyebrow">FINM 37000 · Group 13 · research extension · CME + ICE daily data</p>
<h1>Commodity Spread Lab</h1>
<p class="lead">{extra.get('verdict', '')}</p>

<h2>The question</h2>
<div class="card"><p>Is spread mean reversion only as strong as the economic force that pins the spread together? We sort spreads into four classes, from tightest anchor to loosest: <b>venue</b> (same underlying, different contract or exchange), <b>processing</b> (refining, crushing and milling margins), <b>substitute</b> (related goods or locations) and <b>calendar</b> (storage and carry). {extra.get('coverage_note', '')} Roots used: {', '.join(meta['roots'])}. Results from {meta['start']} on, {meta['n_trials']} configurations in total.</p>
<p>Every number is net of half a tick plus fees on every leg (or once per listed spread, marked [spread book]), filled one day after the signal, vol targeted to 10% a year on $1m and capped at $4m gross. {extra.get('price_note', 'Prices are exchange settlements.')} Non-USD legs are converted with the CME FX future.</p></div>
{f'<div class="card"><img alt="Sharpe by spread class" src="data:image/png;base64,{figs["tax"]}"></div>' if 'tax' in figs else ''}

<h2>Every structure</h2>
<div class="card">{table(sum_tbl, ['class', 'structure', 'start', 'half_life_days', 'hurst', 'adf_p', 'median_grid_gross', 'best_grid', 'wf_sharpe', 'wf_ann', 'wf_mdd'],
    {'half_life_days': days, 'hurst': num, 'adf_p': num, 'median_grid_gross': num, 'best_grid': num, 'wf_sharpe': num, 'wf_ann': pct, 'wf_mdd': pct},
    {'half_life_days': 'half-life (d)', 'adf_p': 'ADF p', 'median_grid_gross': 'median gross', 'best_grid': 'best of 9', 'wf_sharpe': 'walk-fwd net', 'wf_ann': 'ann', 'wf_mdd': 'max DD'})}
<p>Best of 9 is the luckiest parameter set and is shown only to make the point that it is luck; walk-forward net picks the parameters each year from the previous three years and is the number to quote.</p></div>

{alt_html}

<h2>Honest leaderboard</h2>
<div class="card">{table(honest, ['family', 'class', 'name', 'sharpe', 'gross_sharpe', 'ann_ret', 'mdd', 'psr', 'dsr_family', 'dsr_all'],
    {'sharpe': num, 'gross_sharpe': num, 'ann_ret': pct, 'mdd': pct, 'psr': prob, 'dsr_family': prob, 'dsr_all': prob},
    {'sharpe': 'net Sharpe', 'gross_sharpe': 'gross', 'psr': 'P(SR&gt;0)', 'dsr_family': 'deflated, family', 'dsr_all': 'deflated, all'})}
<p>Walk-forward structures, yearly re-selected screens and PCA only (no hand-picked grid points). Deflated Sharpe (Bailey and López de Prado) asks whether the result beats the best Sharpe expected by luck: within its own family, and across all {meta['n_trials']} tries (luck benchmark Sharpe {meta['sr0_all']:.2f}).</p></div>
<div class="card"><img alt="Cumulative PnL of the best ideas" src="data:image/png;base64,{figs['pnl']}"></div>

<h2>What the screen picked</h2>
<div class="card">{''.join(f"<p><b>{k}</b><br>" + '<br>'.join(f"{y}: {'; '.join(v) or 'nothing passed'}" for y, v in lg.items()) + "</p>" for k, lg in meta['logs'].items() if k.endswith('pairs, z=formation'))}</div>

<h2>How to read this</h2>
<div class="card"><p>{extra.get('reading', '')}</p></div>
</main>"""
open(HTML, "w").write(html)
print("wrote", HTML, len(html) // 1024, "KB")
