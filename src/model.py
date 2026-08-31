"""Modelo de cotizaciones optimas de un formador de mercado (dealer).

Precio verdadero P ~ Erlang(K=60, lambda=3), media = K/lambda = 20.

Utilidad esperada por trader que llega:

    Pi(A,B) = pi_L * [ pi_LB(A-S0)*(A-S0) + pi_LS(S0-B)*(S0-B) ]
              - pi_I * [ integral_A^inf (P-A) f(P) dP
                         + integral_0^B (B-P) f(P) dP ]

donde pi_LB y pi_LS son la probabilidad de ejecucion de un trader de
liquidez (no informado), lineal y decreciente en el desvio respecto a S0:

    pi_LB(x) = pi_LS(x) = max(0, 0.50 - 0.08*x)
"""

import numpy as np
from scipy import integrate, optimize, stats

ERLANG_K = 60
ERLANG_LAMBDA = 3.0

LIQUIDITY_INTERCEPT = 0.50
LIQUIDITY_SLOPE = 0.08

BASE_S0 = 19.90
BASE_PI_I = 0.40
BASE_PI_L = 0.60


def price_pdf(P):
    """Densidad f(P) del precio verdadero, Erlang(K=60, lambda=3)."""
    return stats.erlang.pdf(P, a=ERLANG_K, scale=1.0 / ERLANG_LAMBDA)


def execution_prob(x):
    """Probabilidad de ejecucion de un trader de liquidez dado el desvio
    x respecto a S0. Lineal decreciente, acotada en [0, LIQUIDITY_INTERCEPT]."""
    return np.maximum(0.0, LIQUIDITY_INTERCEPT - LIQUIDITY_SLOPE * x)


def expected_loss_ask(A):
    """Perdida esperada frente a traders informados en el lado ask:
    integral_A^inf (P-A) f(P) dP, via scipy.integrate.quad."""
    value, _ = integrate.quad(lambda P: (P - A) * price_pdf(P), A, np.inf)
    return value


def expected_loss_bid(B):
    """Perdida esperada frente a traders informados en el lado bid:
    integral_0^B (B-P) f(P) dP, via scipy.integrate.quad."""
    value, _ = integrate.quad(lambda P: (B - P) * price_pdf(P), 0.0, B)
    return value


def expected_utility(B, A, S0, pi_I, pi_L):
    """Utilidad esperada Pi(A,B) del formador de mercado por trader que llega."""
    liquidity_term = pi_L * (
        execution_prob(A - S0) * (A - S0) + execution_prob(S0 - B) * (S0 - B)
    )
    informed_term = pi_I * (expected_loss_ask(A) + expected_loss_bid(B))
    return liquidity_term - informed_term


def _negative_utility(params, S0, pi_I, pi_L):
    B, A = params
    return -expected_utility(B, A, S0, pi_I, pi_L)


def optimize_quotes(S0=BASE_S0, pi_I=BASE_PI_I, pi_L=BASE_PI_L, x0=None):
    """Optimiza Bid y Ask maximizando la utilidad esperada del dealer.

    Restricciones: B in (0, S0], A in [S0, inf).
    Reporta bid, ask, spread y utilidad esperada redondeados a 2 decimales.
    """
    if x0 is None:
        x0 = [S0 - 0.10, S0 + 0.10]

    bounds = [(1e-6, S0), (S0, None)]
    result = optimize.minimize(
        _negative_utility,
        x0=x0,
        args=(S0, pi_I, pi_L),
        method="L-BFGS-B",
        bounds=bounds,
    )

    B_star, A_star = result.x
    utility_star = -result.fun

    return {
        "bid": round(float(B_star), 2),
        "ask": round(float(A_star), 2),
        "spread": round(float(A_star - B_star), 2),
        "expected_utility": round(float(utility_star), 2),
        "success": bool(result.success),
        "raw_result": result,
    }
