import numpy as np
import pytest

from src.bounds import call_bounds


def test_known_values_in_the_money():
    # S0 - K e^{-rT} = 100 - 90 e^{-0.05} = 14.3894...
    lower, upper = call_bounds(100.0, 90.0, 0.05, 1.0)
    assert lower == pytest.approx(14.3894, abs=1e-4)
    assert upper == 100.0


def test_lower_bound_is_zero_when_deep_out_of_the_money():
    # S0 - K e^{-rT} < 0 : la borne basse est tronquée à 0
    lower, upper = call_bounds(50.0, 100.0, 0.05, 1.0)
    assert lower == 0.0
    assert upper == 50.0


def test_lower_never_exceeds_upper():
    for S0 in [10.0, 80.0, 100.0, 150.0]:
        for K in [50.0, 100.0, 200.0]:
            for r in [0.0, 0.03, 0.10]:
                lower, upper = call_bounds(S0, K, r, 1.0)
                assert 0.0 <= lower <= upper


def test_zero_rate_zero_maturity_gives_intrinsic_value():
    # r = 0 ou T = 0 : borne basse = max(S0 - K, 0)
    lower, _ = call_bounds(120.0, 100.0, 0.0, 1.0)
    assert lower == pytest.approx(20.0)
    lower, _ = call_bounds(120.0, 100.0, 0.05, 0.0)
    assert lower == pytest.approx(20.0)


def test_lower_bound_increases_with_rate():
    # plus r est grand, plus K e^{-rT} est petit, donc plus la borne basse est haute
    low_small_r, _ = call_bounds(100.0, 100.0, 0.01, 1.0)
    low_big_r, _ = call_bounds(100.0, 100.0, 0.08, 1.0)
    assert low_big_r > low_small_r


def test_upper_bound_independent_of_strike_and_rate():
    assert call_bounds(100.0, 50.0, 0.01, 1.0)[1] == 100.0
    assert call_bounds(100.0, 150.0, 0.09, 2.0)[1] == 100.0


def test_vectorized_over_strikes():
    Ks = np.array([80.0, 100.0, 120.0])
    lower, upper = call_bounds(100.0, Ks, 0.05, 1.0)
    assert lower.shape == Ks.shape
    assert np.all(lower >= 0.0)
    assert np.all(lower <= upper)


def test_arbitrage_example_from_exercise():
    # C = 10 < borne basse 14.39 : le prix viole la borne
    lower, upper = call_bounds(100.0, 90.0, 0.05, 1.0)
    assert 10.0 < lower
    assert not (lower <= 10.0 <= upper)
