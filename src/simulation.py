"""Simulador de trades para el formador de mercado.

Cada trade que llega es informado con probabilidad pi_I o de liquidez
con probabilidad pi_L:

- Informado: conoce el precio verdadero P ~ Erlang(K, lambda) y solo
  opera cuando le conviene: compra al ask si P>A (el dealer pierde P-A)
  o vende al bid si P<B (el dealer pierde B-P). Si B<=P<=A no opera.
- Liquidez (no informado): no conoce P. Ejecuta una compra al ask con
  probabilidad execution_prob(A-S0) (PnL del dealer = A-S0), o una venta
  al bid con probabilidad execution_prob(S0-B) (PnL del dealer = S0-B);
  en otro caso no ejecuta. Como execution_prob <= 0.5 siempre, ambas
  probabilidades nunca suman mas de 1.

Esta construccion hace que el PnL esperado por trade simulado converja
exactamente a Pi(A,B) definida en src/model.py.
"""

import numpy as np
from scipy import stats

from src.model import ERLANG_K, ERLANG_LAMBDA, execution_prob


def simulate_trades(n_trades, bid, ask, S0, pi_I, pi_L):
    """Simula n_trades independientes y devuelve el PnL del dealer en cada uno."""
    is_informed = np.random.rand(n_trades) < pi_I
    pnl = np.zeros(n_trades)

    n_informed = int(is_informed.sum())
    if n_informed > 0:
        P = stats.erlang.rvs(a=ERLANG_K, scale=1.0 / ERLANG_LAMBDA, size=n_informed)
        pnl_informed = np.where(P > ask, ask - P, np.where(P < bid, P - bid, 0.0))
        pnl[is_informed] = pnl_informed

    is_liquidity = ~is_informed
    n_liquidity = int(is_liquidity.sum())
    if n_liquidity > 0:
        p_buy = execution_prob(ask - S0)
        p_sell = execution_prob(S0 - bid)
        r = np.random.rand(n_liquidity)
        pnl_liquidity = np.where(
            r < p_buy,
            ask - S0,
            np.where(r < p_buy + p_sell, S0 - bid, 0.0),
        )
        pnl[is_liquidity] = pnl_liquidity

    return pnl


def run_regime(n_trades, bid, ask, S0, pi_I, pi_L):
    """Simula un regimen de cotizacion y devuelve un resumen del PnL."""
    pnl = simulate_trades(n_trades, bid, ask, S0, pi_I, pi_L)
    return {
        "bid": bid,
        "ask": ask,
        "n_trades": n_trades,
        "pnl": pnl,
        "total_pnl": float(pnl.sum()),
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
    }


def monte_carlo(n_runs, n_trades_per_run, bid, ask, S0, pi_I, pi_L):
    """Corre n_runs simulaciones independientes de n_trades_per_run trades
    cada una y devuelve el arreglo de PnL total por corrida."""
    totals = np.empty(n_runs)
    for i in range(n_runs):
        pnl = simulate_trades(n_trades_per_run, bid, ask, S0, pi_I, pi_L)
        totals[i] = pnl.sum()
    return totals
