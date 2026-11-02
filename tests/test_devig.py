import math

import numpy as np

from nba_sim.markets.devig import american_to_implied, devig_american, power_devig


def test_american_implied_even_and_juice():
    assert american_to_implied(-110) == 110 / 210
    assert american_to_implied(100) == 0.5
    assert american_to_implied(150) == 0.4


def test_power_devig_symmetric_juice_is_a_coin_flip():
    fair = devig_american([-110, -110])
    assert fair.sum() == np.float64(fair.sum())
    np.testing.assert_allclose(fair.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(fair, [0.5, 0.5], atol=1e-6)


def test_power_devig_keeps_the_favorite():
    fair = devig_american([150, -180])
    np.testing.assert_allclose(fair.sum(), 1.0, atol=1e-9)
    assert fair[1] > fair[0]
    raw = np.array([american_to_implied(150), american_to_implied(-180)])
    # The method removes vig instead of just dividing by the sum.
    proportional = raw / raw.sum()
    assert not np.allclose(fair, proportional, atol=1e-4)


def test_power_devig_on_already_fair_prices_is_identity():
    prices = np.array([0.62, 0.38])
    np.testing.assert_allclose(power_devig(prices), prices, atol=1e-9)
