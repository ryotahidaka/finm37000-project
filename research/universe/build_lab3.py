"""Append the lab v3 section (relationship map, strategy families, cost models, controls) to the
Commodity Spread Lab page. python build_lab3.py <lab3_out> <page.html>"""
import base64
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT, PAGE = sys.argv[1], sys.argv[2]
M = pd.read_csv(f"{OUT}/relationships.csv")
W = pd.read_csv(f"{OUT}/walkforward.csv")
plt.rcParams.update({"figure.dpi": 140, "font.size": 8})


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


names = ["GC", "SI", "PL", "PA", "HG", "CL", "BRN", "NG", "ZS", "CC"]
C = pd.DataFrame(np.nan, index=names, columns=names)
P = C.copy()
for r in M.itertuples():
    C.loc[r.a, r.b] = C.loc[r.b, r.a] = r.corr
    P.loc[r.a, r.b] = P.loc[r.b, r.a] = r.eg_p
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
for k, (D, title, cmap, vmin, vmax) in enumerate([(C, "Daily return correlation", "RdBu_r", -1, 1),
                                                    (P, "Engle-Granger p-value (low = cointegrated)", "viridis_r", 0, 1)]):
    im = ax[k].imshow(D.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax[k].set_xticks(range(len(names)), names)
    ax[k].set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = D.iloc[i, j]
            if pd.notna(v):
                ax[k].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                           color="white" if (abs(v) > 0.6 if k == 0 else v < 0.4) else "black")
    ax[k].set_title(title)
    fig.colorbar(im, ax=ax[k], fraction=0.046)
heat = b64(fig)

pct = lambda x: f"{100 * x:.1f}%" if pd.notna(x) else ""
num = lambda x: f"{x:.2f}" if pd.notna(x) else ""


def table(df, cols, fmt, heads):
    h = "<tr>" + "".join(f"<th>{heads.get(c, c)}</th>" for c in cols) + "</tr>"
    rows = ["<tr>" + "".join(f"<td>{fmt.get(c, lambda x: x)(r[c])}</td>" for c in cols) + "</tr>" for _, r in df.iterrows()]
    return f"<table>{h}{''.join(rows)}</table>"


fam = W[W["kind"].isin(["pair", "calendar"])].groupby(["kind", "family", "cost_mode"])["sharpe"].agg(
    ["count", "median", lambda s: (s > 0).mean()]).reset_index()
fam.columns = ["kind", "family", "cost_mode", "spreads", "median", "share_pos"]
pv = W[(W["kind"] == "calendar") & (W["family"] == "meta")].pivot(index="spread", columns="cost_mode", values="sharpe")
cal = pd.DataFrame({"calendar": pv.index, "leg": pv["leg"].values, "one_order": pv["spread"].values})
cal["gain"] = cal["one_order"] - cal["leg"]
cal = cal.sort_values("one_order", ascending=False)
top = W[(W["cost_mode"] == "spread") & (W["kind"] != "portfolio")].sort_values("sharpe", ascending=False).head(12)
ctrl = W[(W["kind"] == "outright") & (W["cost_mode"] == "spread") & (W["family"] == "kalman")].sort_values("sharpe", ascending=False)
books = W[W["kind"] == "portfolio"].sort_values("sharpe", ascending=False)
nsig = int((M["eg_p"] < 0.05).sum())

section = f"""
<h2>Lab v3: every pair, four strategy families, two cost models</h2>
<div class="card"><p>All 45 pairs of the 10 tradable products plus a calendar on each, run with four strategy families (z-score mean reversion, trend on the spread, a Hurst regime switch, and a Kalman-filter hedge), and a yearly out-of-sample choice among all 16 configurations per spread. Each is costed two ways: <b>leg</b> (every leg crosses half a tick) and <b>spread</b> (the spread is worked as one order: fees on every leg, but only one leg crosses). {len(W)} walk-forward series from 1,750 configurations.</p></div>

<h3>How the products relate</h3>
<div class="card"><img alt="Correlation and cointegration heatmaps" src="data:image/png;base64,{heat}">
<p>Correlation is not cointegration. Crude and Brent move together (0.89 daily correlation) and gold and silver too (0.78), yet {nsig} of 45 pairs pass an Engle-Granger test at 5% over 2014 to 2026, and no pair stays cointegrated in more than {M['stable_share'].max() * 100:.0f}% of rolling two-year windows. That is the core reason static pairs trading struggles in this universe.</p></div>

<h3>Which strategy family works</h3>
<div class="card">{table(fam.sort_values(['kind', 'family', 'cost_mode']), ['kind', 'family', 'cost_mode', 'spreads', 'median', 'share_pos'],
    {'median': num, 'share_pos': pct}, {'median': 'median walk-fwd Sharpe', 'share_pos': 'share positive', 'cost_mode': 'cost model'})}
<p>Only the Kalman family is positive on more than half the pairs. The control below shows why.</p></div>

<h3>The control that changes the story</h3>
<div class="card">{table(ctrl, ['spread', 'sharpe', 'ann_ret', 'mdd'], {'sharpe': num, 'ann_ret': pct, 'mdd': pct},
    {'spread': 'same Kalman rule on one product alone', 'sharpe': 'walk-fwd Sharpe', 'ann_ret': 'ann', 'mdd': 'max DD'})}
<p>The best "pairs" (palladium against soybeans, Brent or cocoa, platinum against Brent) have tiny hedge ratios (about 0.1 to 0.3), so they are really outright palladium or platinum trades. Run alone, the same rule on palladium scores the same Sharpe as the "pair". The edge is medium-term mean reversion in palladium and platinum around a slowly adapting level, not a spread relationship. It is worth a follow-up, but it should be presented as what it is.</p></div>

<h3>Does charging cost per spread help?</h3>
<div class="card">{table(cal, ['calendar', 'leg', 'one_order', 'gain'], {'leg': num, 'one_order': num, 'gain': num},
    {'leg': 'walk-fwd Sharpe, per-leg cost', 'one_order': 'worked as one spread', 'gain': 'change'})}
<p>For calendars, working the spread as one order adds roughly 0.1 to 0.35 of Sharpe (gold, copper, silver gain most) because two of the four half-ticks per round trip disappear. It does not turn losers into winners: silver is the only calendar clearly positive (0.83 with spread costs, positive in most years), and it earns well under 1% a year at this size. For cross-commodity pairs the cost model barely matters, because those trades turn over slowly.</p></div>

<h3>Top results and books</h3>
<div class="card">{table(top, ['spread', 'family', 'sharpe', 'gross_sharpe', 'ann_ret', 'mdd', 'psr', 'dsr_vs_wf'],
    {'sharpe': num, 'gross_sharpe': num, 'ann_ret': pct, 'mdd': pct, 'psr': pct, 'dsr_vs_wf': pct},
    {'spread': 'spread', 'sharpe': 'net Sharpe', 'gross_sharpe': 'gross', 'psr': 'P(SR&gt;0)', 'dsr_vs_wf': 'deflated'})}
{table(books, ['name', 'sharpe', 'ann_ret', 'mdd'], {'sharpe': num, 'ann_ret': pct, 'mdd': pct}, {'name': 'book (every spread, no hand-picking)', 'sharpe': 'net Sharpe'})}
<p>Spread cost model shown. Deflated is the probability the Sharpe is real after allowing for the {len(W)} walk-forward series compared; nothing clears 50%. Books that hold every spread with its yearly-chosen strategy are flat to slightly negative.</p></div>
"""
html = open(PAGE).read()
marker = "<h2>How to read this</h2>"
start = html.find("<h2>Lab v3:")
if start >= 0:
    html = html[:start] + html[html.find(marker):]
html = html.replace(marker, section + "\n" + marker)
open(PAGE, "w").write(html)
print("ok", len(html) // 1024, "KB")
