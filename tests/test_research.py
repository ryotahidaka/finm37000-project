"""Light checks on the pure helpers of the research lab (research/)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import lab  # noqa: E402
import strategies  # noqa: E402


def _ar1(phi, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_half_life_matches_ar1():
    phi = 0.9
    hl = strategies.half_life(pd.Series(_ar1(phi)))
    assert hl == pytest.approx(np.log(2) / -np.log(phi), rel=0.2)


def test_half_life_random_walk_is_long():
    rw = pd.Series(np.cumsum(np.random.default_rng(1).standard_normal(3000)))
    assert strategies.half_life(rw) > 100


def test_hurst_separates_mean_reversion_from_random_walk():
    rw = np.cumsum(np.random.default_rng(2).standard_normal(3000))
    assert lab.hurst(_ar1(0.5)) < 0.3
    assert lab.hurst(rw) == pytest.approx(0.5, abs=0.1)
    assert np.isnan(lab.hurst(np.arange(10.0)))


def test_summarize_on_constant_gain():
    net = pd.Series(np.r_[np.full(252, 100.0), np.full(252, -50.0)])
    s = lab.summarize(net)
    assert s["ann_ret"] == pytest.approx(25 * 252 / lab.CAP)
    assert s["mdd"] == pytest.approx(-50 * 252 / (lab.CAP + 100 * 252))


def test_deflated_sharpe_penalises_many_trials():
    r = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 2520))
    one = lab.deflated_sharpe(r, [0.8])
    many = lab.deflated_sharpe(r, list(np.random.default_rng(4).normal(0, 0.7, 200)))
    assert one["sr0_ann"] == 0.0
    assert many["sr0_ann"] > 1.0
    assert many["dsr"] < one["dsr"]
    assert 0.0 <= many["dsr"] <= 1.0
