import numpy as np
import pytest

from src.payoffs import call_payoff, put_payoff

K = 100.0
S_T = np.linspace(0, 200, 401)


def test_call_known_values():
    # out of the money, at the money, in the money
    assert call_payoff(80, K) == 0
    assert call_payoff(100, K) == 0
    assert call_payoff(120, K) == 20


def test_put_known_values():
    assert put_payoff(120, K) == 0
    assert put_payoff(100, K) == 0
    assert put_payoff(80, K) == 20
    assert put_payoff(0, K) == K  # max put payoff = K


def test_payoffs_non_negative():
    assert np.all(call_payoff(S_T, K) >= 0)
    assert np.all(put_payoff(S_T, K) >= 0)


def test_call_put_difference_equals_forward_payoff():
    # call - put = S_T - K for every S_T
    assert np.allclose(call_payoff(S_T, K) - put_payoff(S_T, K), S_T - K)


def test_at_most_one_leg_in_the_money():
    # call and put can't both be in the money
    assert np.all(call_payoff(S_T, K) * put_payoff(S_T, K) == 0)


def test_vectorized_shape():
    assert call_payoff(S_T, K).shape == S_T.shape
    assert put_payoff(S_T, K).shape == S_T.shape


def test_breakeven_points():
    premium_call, premium_put = 5.0, 4.0
    assert call_payoff(K + premium_call, K) - premium_call == pytest.approx(0)
    assert put_payoff(K - premium_put, K) - premium_put == pytest.approx(0)
