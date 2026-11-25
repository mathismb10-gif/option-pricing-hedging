import numpy as np
import pytest

from src.discount_factor import discount_factor


def test_zero_maturity_gives_one():
    # no discounting when T = 0
    assert discount_factor(0.05, 0) == pytest.approx(1.0)


def test_zero_rate_gives_one():
    assert discount_factor(0.0, 2.0) == pytest.approx(1.0)


def test_known_value():
    # e^{-0.05 * 2} = e^{-0.1}
    assert discount_factor(0.05, 2.0) == pytest.approx(np.exp(-0.1))
    assert discount_factor(0.05, 2.0) == pytest.approx(0.904837, abs=1e-6)


def test_between_zero_and_one_for_positive_rate():
    assert 0 < discount_factor(0.05, 1.0) < 1


def test_decreasing_in_maturity():
    # longer maturity -> stronger discounting
    assert discount_factor(0.05, 2.0) < discount_factor(0.05, 1.0)


def test_composition_over_periods():
    # D(T1 + T2) = D(T1) * D(T2) with continuous compounding
    assert discount_factor(0.05, 1.5) == pytest.approx(
        discount_factor(0.05, 1.0) * discount_factor(0.05, 0.5)
    )


def test_vectorized():
    T = np.array([0.25, 0.5, 1.0])
    assert discount_factor(0.05, T).shape == T.shape
