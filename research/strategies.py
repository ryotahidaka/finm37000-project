"""Strategies for the lab. All causal: anything used on day t is computed from data up to t-1
(or t's close for the signal, then filled at t+1's close)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from gcsi import contracts
from gcsi.ledger import roll_entry_prices

import lab

CAP = lab.CAP


# ----------------------------------------------------------------------------- simulator
def simulate(dollar_targets: pd.DataFrame, legs: dict, fill_lag: int = 1) -> pd.DataFrame:
    """Signed USD exposure per root decided at each close -> whole contracts -> PnL with rolls/costs.

    dollar_targets: index = trading days, columns = roots.
    Returns daily gross, cost, net (USD) and gross exposure.
    """
    idx = dollar_targets.index
    roots = list(dollar_targets.columns)
    raw = {r: legs[r].cont["raw_close"].reindex(idx).ffill() for r in roots}
    held_c = {r: legs[r].cont["contract"].reindex(idx).ffill() for r in roots}
    roll_px = {r: roll_entry_prices(legs[r].cont, legs[r].logs).reindex(idx) for r in roots}
    mult = {r: contracts.get_spec(r).multiplier for r in roots}
    lots = pd.DataFrame({r: (dollar_targets[r] / (raw[r] * mult[r])).round().fillna(0).astype(int)
                         for r in roots}, index=idx).shift(fill_lag).fillna(0).astype(int)
    gross = np.zeros(len(idx)); cost = np.zeros(len(idx)); expo = np.zeros(len(idx))
    for r in roots:
        p, c, rp = raw[r].to_numpy(), held_c[r].to_numpy(), roll_px[r].to_numpy()
        n = lots[r].to_numpy()
        unit = lab.COSTS.per_contract(r) if r in lab.COSTS.legs else 10.0
        for t in range(1, len(idx)):
            if n[t - 1]:
                entry = rp[t] if c[t] != c[t - 1] and not np.isnan(rp[t]) else p[t - 1]
                gross[t] += n[t - 1] * (p[t] - entry) * mult[r]
            rolls_next = t + 1 < len(idx) and c[t + 1] != c[t]
            traded = abs(n[t - 1]) + abs(n[t]) if rolls_next else abs(n[t] - n[t - 1])
            cost[t] += traded * unit
        expo += np.abs(n * p * mult[r])
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "exposure": expo}, index=idx)


def panel(legs: dict) -> pd.DataFrame:
    """Log ratio-adjusted closes on common trading days."""
    return pd.concat({r: np.log(l.cont["close"]) for r, l in legs.items()}, axis=1, join="inner").dropna()


def half_life(x: pd.Series) -> float:
    x = x.dropna()
    b = np.polyfit(x.shift(1).dropna(), x.iloc[1:], 1)[0]
    return float(np.log(2) / -np.log(b)) if 0 < b < 1 else np.inf


# ----------------------------------------------------------------------------- 1. pairs, formation/trading
def pairs_formation_trading(legs, form_years=3, top_k=3, entry=2.0, exit_=0.0, stop=4.0,
                            hl_range=(1, 63), p_max=0.05, z_mode="formation", gate=None,
                            vol_target=0.10, start_year=2014):
    """GGR-style: each calendar year, select the top_k most cointegrated pairs (EG p-value) on
    the previous form_years, freeze alpha/beta, trade that year only.

    z_mode: "formation" (mean/std of the formation residual, classic) or "rolling60".
    gate: None, or "hurst" (trade a pair only while Hurst of its trailing 120d spread < 0.5).
    """
    P = panel(legs)
    roots = list(P.columns)
    targets = pd.DataFrame(0.0, index=P.index, columns=roots)
    log = []
    for y in range(start_year, P.index.year.max() + 1):
        form = P[(P.index.year >= y - form_years) & (P.index.year < y)]
        trade_idx = P.index[P.index.year == y]
        if len(form) < 200 or len(trade_idx) == 0:
            continue
        cands = []
        for a, b in itertools.combinations(roots, 2):
            for dep, ind in ((a, b), (b, a)):
                beta, alpha = np.polyfit(form[ind], form[dep], 1)
                res = form[dep] - alpha - beta * form[ind]
                p = coint(form[dep], form[ind], trend="c")[1]
                hl = half_life(res)
                if p < p_max and hl_range[0] <= hl <= hl_range[1]:
                    cands.append((p, dep, ind, alpha, beta, res.mean(), res.std(), hl))
        # one direction per pair, best p first
        seen, chosen = set(), []
        for c in sorted(cands):
            key = frozenset(c[1:3])
            if key not in seen:
                seen.add(key); chosen.append(c)
            if len(chosen) == top_k:
                break
        log.append({"year": y, "pairs": [f"{c[1]}~{c[2]} (p={c[0]:.3f}, hl={c[7]:.0f}d, b={c[4]:.2f})" for c in chosen]})
        for p, dep, ind, alpha, beta, mu, sd, hl in chosen:
            span = P.loc[P.index[(P.index.year >= y - 1)]]   # need history for rolling z / gate
            s_all = span[dep] - alpha - beta * span[ind]
            if z_mode == "formation":
                z = (s_all - mu) / sd
            else:
                z = (s_all - s_all.rolling(60).mean().shift(1)) / s_all.rolling(60).std().shift(1)
            z = z.loc[trade_idx]
            pos = _threshold_positions(z, entry, exit_, stop)
            if gate == "hurst":
                h = s_all.rolling(120).apply(lab.hurst, raw=True).shift(1).loc[trade_idx]
                pos = _gate(pos, h < 0.5)
            pos.iloc[-1] = 0                                   # flat at year end
            # dollar sizing: target vol per pair from the formation spread's daily vol
            spread_vol = np.diff(form[dep] - beta * form[ind]).std() * np.sqrt(252)
            usd = CAP * vol_target / np.sqrt(max(len(chosen), 1)) / max(spread_vol, 1e-6)
            # long spread = long dep $usd, short ind $usd*beta (log-price hedge)
            targets.loc[trade_idx, dep] += pos * usd
            targets.loc[trade_idx, ind] -= pos * usd * beta
    return targets, log


def _threshold_positions(z, entry, exit_, stop):
    out, pos, stopped = np.zeros(len(z)), 0, False
    for i, zi in enumerate(z.to_numpy()):
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
    return pd.Series(out, index=z.index)


def _gate(pos, ok):
    out, cur = pos.to_numpy().copy(), 0
    okv = ok.fillna(False).to_numpy()
    for i in range(len(out)):
        if out[i] != 0 and out[i] != cur and not okv[i]:
            out[i] = 0
        cur = out[i]
    return pd.Series(out, index=pos.index)


# ----------------------------------------------------------------------------- 2. PCA residual stat-arb
def pca_residual(legs, win_pca=252, win_ou=60, s_open=1.25, s_close=0.5, kappa_min=252 / 30,
                 gross_per_signal=CAP * 0.25):
    """Avellaneda & Lee (2010) s-score on residuals vs the first principal component.

    Literature thresholds (open 1.25, close 0.5, kappa > 252/30), not tuned here.
    For metal i with s < -open: long $g of i, short beta_i * $g of the PC1 eigenportfolio.
    """
    P = panel(legs)
    R = P.diff().dropna()
    roots = list(R.columns)
    idx = R.index
    targets = pd.DataFrame(0.0, index=idx, columns=roots)
    state = {r: 0 for r in roots}
    for t in range(win_pca + 1, len(idx)):
        hist = R.iloc[t - win_pca:t]                               # up to t-1
        cov = np.cov(hist.T.to_numpy())
        w, v = np.linalg.eigh(cov)
        pc = v[:, -1]
        pc = pc / np.sum(np.abs(pc))
        sd = hist.std().to_numpy()
        eig_w = pc / sd                                               # eigenportfolio weights
        eig_w = eig_w / np.sum(np.abs(eig_w))
        F = (R.iloc[t - win_ou:t + 1].to_numpy() @ eig_w)            # includes today for the signal
        expo = np.zeros(len(roots))
        for i, r in enumerate(roots):
            y = R.iloc[t - win_ou:t + 1, i].to_numpy()
            b, a = np.polyfit(F[:-1], y[:-1], 1)                     # beta fitted to t-1
            eps = y - a - b * F
            X = np.cumsum(eps)
            bb, aa = np.polyfit(X[:-2], X[1:-1], 1)                   # AR(1) on residual level to t-1
            if not 0 < bb < 1:
                state[r] = 0; continue
            kappa = -np.log(bb) * 252
            m = aa / (1 - bb)
            sig_eq = np.sqrt(np.var(X[1:-1] - aa - bb * X[:-2]) / (1 - bb ** 2))
            s = (X[-1] - m) / sig_eq if sig_eq > 0 else 0
            if kappa < kappa_min:
                state[r] = 0
            elif state[r] == 0:
                state[r] = 1 if s < -s_open else -1 if s > s_open else 0
            elif state[r] == 1 and s > -s_close:
                state[r] = 0
            elif state[r] == -1 and s < s_close:
                state[r] = 0
            if state[r]:
                e = np.zeros(len(roots)); e[i] = 1.0
                expo += state[r] * gross_per_signal * (e - b * eig_w)
        targets.iloc[t] = expo
    return targets


# ----------------------------------------------------------------------------- 3. Johansen basket
def johansen_basket(legs, form_years=3, size=3, entry=2.0, exit_=0.0, stop=4.0, vol_target=0.10,
                    start_year=2014, hl_range=(1, 63)):
    """Each year pick the basket of `size` metals with the strongest Johansen trace statistic
    (rank >= 1 at 5%) on the previous form_years; weights = first cointegrating vector; trade the
    basket's z-score (formation mean/std) that year only."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    P = panel(legs)
    roots = list(P.columns)
    targets = pd.DataFrame(0.0, index=P.index, columns=roots)
    log = []
    if len(roots) < size:
        return targets, log
    for y in range(start_year, P.index.year.max() + 1):
        form = P[(P.index.year >= y - form_years) & (P.index.year < y)]
        trade_idx = P.index[P.index.year == y]
        if len(form) < 200 or len(trade_idx) == 0:
            continue
        best = None
        for combo in itertools.combinations(roots, size):
            res = coint_johansen(form[list(combo)].to_numpy(), 0, 1)
            excess = float(np.real(res.lr1[0]) - res.cvt[0, 1])
            if excess <= 0:
                continue
            w = np.real(res.evec[:, 0])
            sp = form[list(combo)].to_numpy() @ w
            hl = half_life(pd.Series(sp))
            if not hl_range[0] <= hl <= hl_range[1]:
                continue
            if best is None or excess > best[0]:
                best = (excess, combo, w, sp.mean(), sp.std(), hl)
        if best is None:
            log.append({"year": y, "basket": None}); continue
        excess, combo, w, mu, sd, hl = best
        log.append({"year": y, "basket": f"{'+'.join(combo)} w={np.round(w / np.abs(w).max(), 2).tolist()} hl={hl:.0f}d"})
        z = (P.loc[trade_idx, list(combo)].to_numpy() @ w - mu) / sd
        pos = _threshold_positions(pd.Series(z, index=trade_idx), entry, exit_, stop)
        pos.iloc[-1] = 0
        d = np.diff(form[list(combo)].to_numpy() @ w).std() * np.sqrt(252)
        # w applies to log prices -> dollar weights; normalise so the basket hits vol_target
        scale = CAP * vol_target / max(d, 1e-9)
        for r, wi in zip(combo, w):
            targets.loc[trade_idx, r] += pos * scale * wi
    return targets, log


# ----------------------------------------------------------------------------- 4. spread momentum / carry
def spread_momentum(legs, lookback=126, vol_win=63, vol_target=0.10, rebalance=21):
    """The diagnosis said spreads trend for years, so trade that: for every pair (a, b), go long
    the log ratio a/b if its trailing `lookback`-day change is positive, short if negative.
    Each pair is scaled to vol_target/sqrt(n_pairs) from its trailing realised vol; rebalanced
    every `rebalance` days. Dollar-neutral per pair (long $x of a, short $x of b)."""
    P = panel(legs)
    roots = list(P.columns)
    pairs = list(itertools.combinations(roots, 2))
    targets = pd.DataFrame(0.0, index=P.index, columns=roots)
    for a, b in pairs:
        s = P[a] - P[b]
        sig = np.sign(s - s.shift(lookback))
        vol = s.diff().rolling(vol_win).std() * np.sqrt(252)
        usd = CAP * vol_target / np.sqrt(len(pairs)) / vol
        pos = (sig * usd).where(np.arange(len(s)) % rebalance == 0).ffill()
        targets[a] += pos.fillna(0)
        targets[b] -= pos.fillna(0)
    return targets


def xs_carry(legs, vol_win=63, vol_target=0.10, rebalance=21, n_side=1):
    """Cross-sectional carry (Koijen et al. 2018 style): roll yield of each metal = log(front/next)
    annualised, using the ratio-adjusted continuous series' roll gaps as the curve slope proxy.
    Long the n_side highest-carry metals, short the n_side lowest, risk-weighted, monthly."""
    P = panel(legs)
    roots = list(P.columns)
    carry = {}
    for r in roots:
        lg = legs[r].logs
        if lg is None or len(lg) == 0:
            carry[r] = pd.Series(0.0, index=P.index); continue
        # annualised log gap of each roll, known at the roll's execution date, carried forward
        yrs = 0.25 if r in ("PL", "PA") else 1 / 6
        g = pd.Series(-np.log(lg["ratio"].to_numpy()) / yrs, index=pd.DatetimeIndex(lg["exec_date"]))
        carry[r] = g.reindex(P.index).ffill()
    C = pd.DataFrame(carry)
    vol = P.diff().rolling(vol_win).std() * np.sqrt(252)
    targets = pd.DataFrame(0.0, index=P.index, columns=roots)
    for t in range(len(P)):
        if t % rebalance or C.iloc[t].isna().any() or vol.iloc[t].isna().any():
            if t:
                targets.iloc[t] = targets.iloc[t - 1]
            continue
        ranks = C.iloc[t].sort_values()
        w = pd.Series(0.0, index=roots)
        w[ranks.index[-n_side:]] = 1.0
        w[ranks.index[:n_side]] = -1.0
        targets.iloc[t] = w * CAP * vol_target / np.sqrt(2 * n_side) / vol.iloc[t]
    return targets


def xs_momentum(legs, lookback=252, skip=21, vol_win=63, vol_target=0.10, rebalance=21, n_side=1):
    """Cross-sectional momentum across the metals: long the best trailing 12-1 month performer,
    short the worst, risk-weighted, monthly."""
    P = panel(legs)
    roots = list(P.columns)
    mom = P.shift(skip) - P.shift(lookback)
    vol = P.diff().rolling(vol_win).std() * np.sqrt(252)
    targets = pd.DataFrame(0.0, index=P.index, columns=roots)
    for t in range(len(P)):
        if t % rebalance or mom.iloc[t].isna().any() or vol.iloc[t].isna().any():
            if t:
                targets.iloc[t] = targets.iloc[t - 1]
            continue
        ranks = mom.iloc[t].sort_values()
        w = pd.Series(0.0, index=roots)
        w[ranks.index[-n_side:]] = 1.0
        w[ranks.index[:n_side]] = -1.0
        targets.iloc[t] = w * CAP * vol_target / np.sqrt(2 * n_side) / vol.iloc[t]
    return targets
