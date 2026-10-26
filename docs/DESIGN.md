# Project design

How the pieces fit together and why. The README says what the project delivers and how to run it; this document covers the pipeline, the components, the design decisions and where the work goes next.

## 1. Goal in one line

A reproducible research framework that pulls CME futures from Databento, builds roll-aware continuous series, and tests relative-value strategies with realistic contract sizing and costs, then reports honestly whether an edge survives. Gold vs silver is the first case study; the research lab extends it to five metals.

## 2. Scope

After the professor's review on 18 Oct the project has one focus: daily strategies with long-run continuous adjustment and an honest, cost-aware backtest. Trade-level execution (legging, routing, replaying intraday quotes) is Part B material and sits in [future work](#8-future-work).

- Signals once a day, on the ratio-adjusted continuous series.
- Positions in real contracts (GC/SI with MGC/SIL for the remainder), marked on actual prices, every roll charged.
- Data: `GLBX.MDP3` from 2010-06-06, `definition`, `ohlcv-1d` and `statistics` per contract via parent symbology.

## 3. Pipeline (data to result)

```mermaid
flowchart TD
    A[Databento GLBX.MDP3<br/>per-contract GC, SI, PL, PA, HG] --> B[Ingestion and cache<br/>definition, ohlcv-1d, statistics]
    B --> C[Continuous build<br/>roll 5 business days before FND]
    C --> D1[Ratio-adjusted series<br/>for SIGNALS]
    C --> D2[Actual prices + roll log<br/>for PnL]
    D1 --> E[Cointegration, hedge ratio, OU<br/>full sample and year by year]
    E --> H[Signals<br/>z-score bands, beta-neutral sizing]
    H --> I[Backtest<br/>fill lag 1, walk-forward, costs, ledger]
    D2 --> I
    I --> J[Evaluation and report<br/>Sharpe, drawdown, attribution, stress windows]
    D1 --> L[Research lab<br/>pairs, PCA, Johansen, trend, carry, curves]
    D2 --> L
    L --> M[Deflated Sharpe over every trial]
    J --> K[output/ and docs/results/]
    M --> K
```

## 4. Components

```mermaid
flowchart LR
    subgraph course[Course package finm37000 - reused]
        c1[db_env_util.py<br/>Databento key handling]
        c2[continuous.py<br/>continuous/roll helpers]
        c3[market_data.py / futures.py]
    end
    subgraph ours[Our package src/]
        s0[contracts.py<br/>specs + calendar]
        s1[data.py + rolls.py<br/>ingestion + continuous series]
        s2[cointegration.py + ou.py<br/>hedge ratio + half-life]
        s3[signals.py<br/>z-score + sizing]
        s4[backtest.py + metrics.py<br/>walk-forward + evaluation]
    end
    course --> s1
    s0 --> s1
    s1 --> s2 --> s3 --> s4
    s4 --> out[output/ results]
```

We reuse the professor's `finm37000` package for the data and roll plumbing rather than rebuilding it; our package holds the pairs-trading logic on top.

The `research/` folder sits beside `src/gcsi`: it imports the contract specs, rolls, cost model and ledger, adds platinum, palladium and copper specs, and runs the wider strategy search. It is exploration code with light tests; `src/gcsi` is the tested core.

## 5. Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Case study | Gold (GC) vs silver (SI) | Economically linked, liquid, with micros for fine sizing; course-permitted |
| Continuous series | Two series: ratio-adjusted for signals, actual prices for PnL | Mixing them corrupts either the signal (roll gaps look like moves) or the PnL |
| Roll | 5 business days before first notice | Avoids delivery and the thin last days; logged and charged as two trades |
| Hedge ratio | Trailing 1-year OLS, Kalman as a sensitivity | The relationship drifts; a static full-sample beta would look ahead |
| Signal | z-score with the lookback re-priced at today's beta | A consistent z-score; otherwise beta changes create fake signals |
| Fills | One day after the signal at the close | No same-close fills on a signal computed from that close |
| Costs | Half a tick + exchange fee + commission per contract per side | A spread crosses both legs on entry, exit and every roll |
| Attribution | Ledger splits ideal spread, sizing rounding, trade and roll costs | Shows whether losses come from the idea or from frictions |
| Parameter choice | Walk-forward on the previous 3 years | Out-of-sample by construction |
| Prices in the lab | Official settlements (`statistics`, stat 3) | Both legs priced at the same moment; last trades print at different times |
| Multiple testing | Count every configuration, report the deflated Sharpe | With 191 tries the best Sharpe is not evidence on its own |
| Leverage in the lab | Leg notional capped at $2m; spread-book cost mode for curve trades | Quiet spreads would otherwise be levered without limit; a listed spread is one fill |
| Libraries | statsmodels, scipy, pykalman | Free, standard, no paid tooling |

## 6. How the Issues map to components

| Issue | Component | Produces |
|---|---|---|
| #3 Structure + env | repo root | installable environment |
| #18 Contract specs | `contracts.py` | multipliers, ticks, FND / last-trade calendar |
| #4 Databento ingestion | `data.py` | cached GC/SI frames |
| #5 Continuous + roll | `rolls.py` | ratio-adjusted + real-price series |
| #6 EDA | `report.py` (`eda`) | cointegration by year, half-life by window, tradability assessment |
| #7 Cointegration + hedge ratio | `cointegration.py` | spread definition |
| #8 OU + half-life | `ou.py` | reversion parameters |
| #9 Signal generation | `signals.py` | dated positions |
| #10 Backtest | `backtest.py` | equity curve with costs |
| #24 Cost model | `costs.py` | cost per fill and per roll |
| #25 PnL ledger | `ledger.py` | fills, daily PnL split into ideal spread / whole-contract sizing / trade costs / roll costs |
| #23 CI | `.github/workflows` | tests on every PR |
| #11 Evaluation | `metrics.py` + `report.py` -> `output/` | metrics, stress windows, report.md + figures |

| (lab) Research lab | `research/` | 182 more configurations on five metals, trial tables, deflated Sharpe |

## 7. Central risk, and how it played out

We stated up front that the gold/silver relationship might not be stable and that a clear negative result, honestly shown, would be a valid outcome. That is what happened: the pair is not cointegrated over 2010-2026 (Engle-Granger p 0.87), the ratio trends for years, and the walk-forward Sharpe is -0.85 after costs, with costs only about $31k against a $236k gross loss. The synthetic check (Sharpe 1.15 when mean reversion is built in) shows the code is not the reason. The research lab then showed that strategies that do not assume mean reversion between two metals (PCA residuals, momentum, carry) are the ones with a positive sign, though none clears the multiple-testing bar.

## 8. Future work

| Idea | Why it's interesting | What it needs | Was |
|---|---|---|---|
| Cross-commodity spread taxonomy | Test whether mean reversion strength ranks with how hard the link between legs is: venue arbitrage (CL vs ICE WTI, BZ vs ICE Brent, NG vs HH), processing margins (cracks, soybean crush, white sugar premium, ethanol), substitutes (WTI vs Brent, NG vs TTF, NY vs London cocoa, arabica vs robusta, wheat spreads), calendars | ICE Futures US, ICE Futures Europe and ICE Endex data next to CME, FX legs, unit conversions, cross-exchange calendars. Energy mean reversion is outside this course's permitted strategies, so this is a research extension | (new) |
| Execution layer | Leg the spread on the harder-to-fill side, route between standard and micro books | Intraday data, its own validation | #26 |
| Execution simulator | Replay top-of-book to measure what the execution layer saves | MBP-1 or BBO history and a fill model | #27 |
| Copula signal | A second bivariate model to compare with cointegration and OU | Gains usually marginal after costs | #12 |
| Regime-aware PCA residual | The lab's best lead earns most of its Sharpe after 2022; test whether a regime filter or volatility scaling makes it stable | More history or more names (the ICE metals and other CME commodities) | (new) |
