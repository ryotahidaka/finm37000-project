r"""Turn the lab output folder into one self-contained HTML report (figures embedded).

    python research\build_report.py --out output\lab --html output\lab\metals-lab.html

An optional extra.json in --out supplies the verdict, reading notes and a robustness table.
"""
import argparse
import base64
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default="output/lab", help="folder written by run_all.py")
_ap.add_argument("--html", default=None, help="report path (default <out>/metals-lab.html)")
_args = _ap.parse_args()
OUT = _args.out
HTML = _args.html or f"{OUT}/metals-lab.html"
T = pd.read_csv(f"{OUT}/all_trials.csv")
L = pd.read_csv(f"{OUT}/leaders.csv")
wf = pd.read_csv(f"{OUT}/curve_wf.csv")
S = pd.read_parquet(f"{OUT}/series.parquet")
meta = json.load(open(f"{OUT}/meta.json"))
extra = json.load(open(f"{OUT}/extra.json")) if __import__("os").path.exists(f"{OUT}/extra.json") else {}

NAVY, GOLD, SLATE, GREY = "#14213D", "#B8862B", "#6B7B8C", "#A9B4C2"
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# figure 1: Sharpe of every trial by family
fams = T["family"].unique().tolist()
fig, ax = plt.subplots(figsize=(8, 3.2))
for i, f in enumerate(fams):
    v = T.loc[T["family"] == f, "sharpe"].dropna()
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=14,
               color=GOLD if f.startswith("curve") else NAVY, alpha=0.8)
ax.axhline(0, color=GREY, lw=1)
ax.axhline(meta.get("sr0", 0), color="#C8642B", lw=1, ls="--")
ax.set_xticks(range(len(fams)), fams, rotation=15)
ax.set_ylabel("Sharpe after costs, 2014-2026")
ax.set_title("Every configuration we tried (dashed: best Sharpe expected by luck alone)")
f1 = fig_b64(fig)

# figure 2: cumulative PnL of the top 5 + reference
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, name in enumerate(L["name"].head(5)):
    s = S[name].fillna(0).cumsum() / 1e3
    ax.plot(s.index, s, lw=1.2, label=name[:60], color=[NAVY, GOLD, SLATE, "#2E6FBA", "#C8642B"][i])
ax.set_ylabel("cumulative net PnL ($k, $1m capital)")
ax.legend(fontsize=7, loc="upper left")
ax.set_title("Top five configurations by Sharpe")
f2 = fig_b64(fig)


def table(df, cols, fmt):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    rows = []
    for _, r in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{fmt.get(c, lambda x: x)(r[c])}</td>" for c in cols) + "</tr>")
    return f"<table>{h}{''.join(rows)}</table>"


pct = lambda x: f"{100 * x:.1f}%" if pd.notna(x) else ""
num = lambda x: f"{x:.2f}" if pd.notna(x) else ""
prob = lambda x: f"{100 * x:.0f}%" if pd.notna(x) else ""

fam_tbl = T.groupby("family")["sharpe"].agg(["count", "median", "max"]).reset_index()
fam_tbl.columns = ["family", "configs", "median Sharpe", "best Sharpe"]

html = f"""<title>Metals Strategy Lab</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
:root{{--bg:#F7F5EF;--ink:#14213D;--muted:#4A5568;--card:#FFFDF8;--line:#E2DCCD;--accent:#8A6416}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#10192E;--ink:#EEF1F5;--muted:#B5C0CE;--card:#18233D;--line:#2A3858;--accent:#D9B25A;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#10192E;--ink:#EEF1F5;--muted:#B5C0CE;--card:#18233D;--line:#2A3858;--accent:#D9B25A;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:16px/1.55 "IBM Plex Sans",system-ui,sans-serif;margin:0}}
main{{max-width:980px;margin:0 auto;padding-block:32px 64px;padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px))}}
h1{{font-family:"EB Garamond",Georgia,serif;font-weight:500;font-size:40px;margin:0 0 4px}}
h2{{font-family:"EB Garamond",Georgia,serif;font-weight:500;font-size:28px;margin:40px 0 8px}}
.eyebrow{{color:var(--accent);letter-spacing:.12em;text-transform:uppercase;font-size:12px}}
.lead{{font-size:18px;color:var(--muted)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0;overflow-x:auto}}
th,td{{font-variant-numeric:tabular-nums}} table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}
img{{width:100%;height:auto;border-radius:8px;background:#fff}}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@500&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
<p class="eyebrow">FINM 37000 · Group 13 · strategy lab · real CME data 2010–2026</p>
<h1>Metals Strategy Lab</h1>
<p class="lead">{extra.get('verdict', '')}</p>

<h2>What we tested</h2>
<div class="card">
<p>Metals: {', '.join(meta['roots'])}. Settlement prices used for: {', '.join(meta['settled']) or 'none (last trade)'}.
Every configuration was fixed before it was run, filled one day after the signal, charged half a tick plus fees and commission on every leg, and scored on 2014–2026 net PnL on $1m capital.
In total we count <b>{meta['n_trials']}</b> configurations, including the 9 earlier gold/silver runs, and correct for that number with the deflated Sharpe ratio.</p>
{table(fam_tbl, ['family', 'configs', 'median Sharpe', 'best Sharpe'], {'median Sharpe': num, 'best Sharpe': num})}
</div>
<div class="card"><img alt="Sharpe ratio of every configuration by family" src="data:image/png;base64,{f1}"></div>

<h2>Leaders, after the multiple-testing correction</h2>
<div class="card">
{table(L, ['family', 'name', 'sharpe', 'ann_ret', 'mdd', 'deflated_sharpe_prob'],
       {'sharpe': num, 'ann_ret': pct, 'mdd': pct, 'deflated_sharpe_prob': prob})}
<p>The last column is the probability that the true Sharpe is above zero once you allow for having picked the best of {meta['n_trials']} tries (Bailey and López de Prado, 2014), using all {meta['n_trials']} trials. The bar here is harsh: the luck benchmark is a Sharpe of {meta.get('sr0', 0):.2f}, set by how widely the 144 curve-grid configurations spread. Names marked [spread book] charge spread-exchange costs instead of two separate legs.</p>
</div>
<div class="card"><img alt="Cumulative net PnL of the five best configurations" src="data:image/png;base64,{f2}"></div>

<h2>Robustness of the leaders</h2>
<div class="card">{extra.get('robust_html', '')}</div>

<h2>Curve trades, out of sample</h2>
<div class="card">{table(wf.sort_values('wf_sharpe', ascending=False), ['structure', 'wf_sharpe', 'wf_ann', 'wf_mdd', 'best_in_sample', 'median_grid'],
                          {'wf_sharpe': num, 'wf_ann': pct, 'wf_mdd': pct, 'best_in_sample': num, 'median_grid': num})}</div>

<h2>Which pairs the formation step picked</h2>
<div class="card"><table><tr><th>Year</th><th>Pairs (rolling 60d z, no gate)</th></tr>
{''.join(f"<tr><td>{r['year']}</td><td>{'; '.join(r['pairs']) or 'none passed p &lt; 0.05'}</td></tr>" for r in meta['pair_logs'].get('Pairs top-3, z=rolling60, gate=none', []))}
</table></div>

<h2>How to read this</h2>
<div class="card"><p>{extra.get('reading', '')}</p></div>
</main>"""
open(HTML, "w").write(html)
print("wrote", HTML, len(html) // 1024, "KB")
