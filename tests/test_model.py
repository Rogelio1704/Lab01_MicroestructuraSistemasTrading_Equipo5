"""Pruebas de src/model.py, ejecutables via pytest."""

import numpy as np

from src.model import (
    LIQUIDITY_INTERCEPT,
    LIQUIDITY_SLOPE,
    execution_prob,
    expected_loss_ask,
    optimize_quotes,
)


def test_execution_prob_is_never_negative():
    x = np.linspace(-5, 50, 200)
    probs = execution_prob(x)
    assert np.all(probs >= 0.0)


def test_expected_loss_ask_is_decreasing_in_A():
    S0 = 19.90
    A_low = S0 + 0.10
    A_high = S0 + 1.00
    assert expected_loss_ask(A_low) > expected_loss_ask(A_high)


def test_optimal_spread_without_informed_traders():
    """Con pi_I=0 (monopolista puro), el spread optimo por lado es
    0.50/(2*0.08) = 3.125, y el spread total A-B es 0.50/0.08 = 6.25."""
    S0 = 19.90
    result = optimize_quotes(S0=S0, pi_I=0.0, pi_L=0.60)

    half_spread_analytic = LIQUIDITY_INTERCEPT / (2 * LIQUIDITY_SLOPE)
    total_spread_analytic = LIQUIDITY_INTERCEPT / LIQUIDITY_SLOPE

    assert abs((result["ask"] - S0) - half_spread_analytic) < 0.05
    assert abs((S0 - result["bid"]) - half_spread_analytic) < 0.05
    assert abs(result["spread"] - total_spread_analytic) < 0.05
