import numpy as np
import pytest

from src.parity import parity_gap, put_from_call, call_from_put

S0, K, r, T = 100.0, 100.0, 0.05, 1.0


def test_put_from_call_known_value():
    # P = C + K e^{-rT} - S0 = 10 + 95.1229... - 100
    assert put_from_call(10.0, S0, K, r, T) == pytest.approx(5.1229, abs=1e-4)


def test_call_from_put_known_value():
    # C = P - K e^{-rT} + S0
    assert call_from_put(5.1229, S0, K, r, T) == pytest.approx(10.0, abs=1e-4)


def test_round_trip_call_put_call():
    C = 7.3
    P = put_from_call(C, S0, K, r, T)
    assert call_from_put(P, S0, K, r, T) == pytest.approx(C)


def test_gap_is_zero_when_prices_are_consistent():
    C = 10.0
    P = put_from_call(C, S0, K, r, T)
    assert parity_gap(C, P, S0, K, r, T) == pytest.approx(0.0, abs=1e-12)


def test_gap_is_nonzero_for_inconsistent_prices():
    # les primes 5 (call) et 4 (put) de la partie A violent la parité
    assert abs(parity_gap(5.0, 4.0, S0, K, r, T)) > 1.0


def test_gap_changes_one_for_one_with_call_price():
    base = parity_gap(10.0, 5.0, S0, K, r, T)
    bumped = parity_gap(11.0, 5.0, S0, K, r, T)
    assert bumped - base == pytest.approx(1.0)


def test_zero_rate_zero_maturity_reduces_to_intrinsic_relation():
    # r = 0 : C - P = S0 - K
    assert put_from_call(10.0, 100.0, 95.0, 0.0, 1.0) == pytest.approx(10.0 - 5.0)


def test_vectorized_over_strikes():
    Ks = np.array([90.0, 100.0, 110.0])
    C = np.array([16.0, 10.0, 5.0])
    P = put_from_call(C, S0, Ks, r, T)
    assert P.shape == Ks.shape
    assert np.allclose(parity_gap(C, P, S0, Ks, r, T), 0.0)
