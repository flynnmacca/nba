import numpy as np

from nba_sim.markets.kelly import interval_contains, kelly_stake
from nba_sim.markets.edge import prob_over


def test_quarter_kelly_is_capped():
    # Fair even-money price, model at 60 percent. Full Kelly is 0.20.
    stake = kelly_stake(0.60, 100, fraction=0.25, cap=0.02)
    assert stake == 0.02


def test_negative_edge_stakes_zero():
    assert kelly_stake(0.40, -110) == 0.0


def test_wide_interval_swallows_a_tiny_edge():
    assert interval_contains(0.55, 0.548, n_sims=400, z=1.64)
    assert not interval_contains(0.62, 0.50, n_sims=4000, z=1.64)


def test_half_point_over_probability():
    samples = np.array([0, 1, 2, 3], dtype=float)
    assert prob_over(samples, 1.5) == 0.5
    # Push on an integer line is half a win.
    assert prob_over(samples, 1.0) == 0.625
