"""Tests for the cross-commodity study code in research/universe (synthetic data only)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "universe"))
core = pytest.importorskip("core")
regimes = pytest.importorskip("regimes")
lab3 = pytest.importorskip("lab3")


def test_parse_contract_cme_decade_and_spreads():
    # single-digit year resolves to the decade of the first trading date
    assert core.parse_contract("GCZ6", ("GC",), 2016) == 2016 * 12 + 11
    assert core.parse_contract("GCZ6", ("GC",), 2025) == 2026 * 12 + 11
    assert core.parse_contract("CLH5-CLM5", ("CL",), 2015) is None
    assert core.parse_contract("CL:C1 RB-CL G5", ("CL",), 2015) is None


def test_parse_contract_ice_outrights_only():
    assert core.parse_contract("CC  FMN0024!", ("CC",), 2024) == 2024 * 12 + 6
    assert core.parse_contract("BRN FMU0019!", ("BRN",), 2019) == 2019 * 12 + 8
    assert core.parse_contract("CC  FMH0024_Z", ("CC",), 2024) is None          # trade at settlement
    assert core.parse_contract("CC  FMU0024-CC  FMZ0024", ("CC",), 2024) is None  # calendar spread


def _two_contract_frame():
    days = pd.bdate_range("2020-01-01", periods=60, tz="UTC")
    rows = []
    for i, d in enumerate(days):
        # front contract loses open interest to the back contract half way through
        rows.append({"date": d, "iid": 1, "symbol": "GCG0", "ym": 2020 * 12 + 1, "px": 100.0 + i,
                     "volume": 10.0, "oi": 1000.0 - 20 * i})
        rows.append({"date": d, "iid": 2, "symbol": "GCJ0", "ym": 2020 * 12 + 3, "px": 101.0 + i,
                     "volume": 10.0, "oi": 20.0 * i})
    return pd.DataFrame(rows)


def test_roll_follows_open_interest_and_pnl_has_no_roll_gap():
    leg = core.build_leg(_two_contract_frame(), "GC")
    a = leg.active.dropna()
    assert a.iloc[0] == 2020 * 12 + 1 and a.iloc[-1] == 2020 * 12 + 3
    assert (a.diff().fillna(0) >= 0).all()          # rolls only forward
    # both contracts rise by exactly 1.0 a day, so the held-contract change is always 1.0:
    # the 1.0 gap between the contracts never shows up as PnL
    assert np.allclose(leg.dpx.dropna(), 1.0)


def test_spread_cost_model_is_never_dearer_than_per_leg():
    leg = core.build_leg(_two_contract_frame(), "GC")
    leg.fx = pd.Series(1.0, index=leg.px.index)
    legs = {"GC": leg, "SI": leg}
    lots = pd.DataFrame({"GC": np.r_[np.zeros(10), np.ones(50)], "SI": np.r_[np.zeros(10), -np.ones(50)]},
                        index=leg.px.index)
    per_leg = lab3.simulate(lots, legs, "leg")["cost"].sum()
    one_order = lab3.simulate(lots, legs, "spread")["cost"].sum()
    assert 0 < one_order < per_leg


def test_volatility_regime_uses_only_the_past():
    idx = pd.bdate_range("2015-01-01", periods=800, tz="UTC")
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"x": rng.normal(0, 0.01, 800)}, index=idx)
    base = regimes.label_regimes(idx, r)
    shocked = r.copy()
    shocked.iloc[700:] *= 10                         # a shock late in the sample
    after = regimes.label_regimes(idx, shocked)
    assert (base.iloc[:700] == after.iloc[:700]).all()   # earlier labels cannot change
    assert (after.iloc[702:760] == "high volatility").any()


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.01, 2500))
    few = core.deflated_sharpe(r, [0.1, 0.2, 0.3])["dsr"]
    many = core.deflated_sharpe(r, list(rng.normal(0, 0.6, 500)))["dsr"]
    assert many < few
