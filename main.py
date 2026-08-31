"""Orquesta el flujo completo del Lab01: optimizacion de cotizaciones,
simulacion de trades, analisis de Monte Carlo y generacion de figuras.

Toda la logica vive en src/; este archivo solo llama funciones y reporta
resultados.
"""

import numpy as np

from src.model import (
    BASE_PI_I,
    BASE_PI_L,
    BASE_S0,
    expected_loss_ask,
    expected_loss_bid,
    optimize_quotes,
    price_pdf,
)
from src.plots import (
    plot_loss_functions,
    plot_monte_carlo_totals,
    plot_pnl_distributions,
    plot_price_distribution,
)
from src.simulation import monte_carlo, simulate_trades

SEED = 42
N_TRADES = 10_000
MC_RUNS = 1_000
MC_TRADES = 1_000
FIGURES_DIR = "figures"


def main():
    np.random.seed(SEED)

    # 1-3. Distribucion del precio, perdidas frente a informados y
    #      optimizacion de Bid/Ask -----------------------------------------
    result = optimize_quotes(S0=BASE_S0, pi_I=BASE_PI_I, pi_L=BASE_PI_L)

    print("=== Cotizaciones optimas (caso base) ===")
    print(f"S0:                {BASE_S0:.2f}")
    print(f"pi_I / pi_L:       {BASE_PI_I:.2f} / {BASE_PI_L:.2f}")
    print(f"Bid optimo:        {result['bid']:.2f}")
    print(f"Ask optimo:        {result['ask']:.2f}")
    print(f"Spread optimo:     {result['spread']:.2f}")
    print(f"Utilidad esperada: {result['expected_utility']:.2f}")

    # 4. Simulacion de 10,000 trades bajo tres regimenes --------------------
    regimes = {
        "Optimo": (result["bid"], result["ask"]),
        "Estrecho": (19.75, 20.05),
        "Amplio": (18.40, 21.40),
    }

    print("\n=== Simulacion de 10,000 trades por regimen ===")
    pnl_by_regime = {}
    for name, (bid, ask) in regimes.items():
        pnl = simulate_trades(N_TRADES, bid, ask, BASE_S0, BASE_PI_I, BASE_PI_L)
        pnl_by_regime[name] = pnl
        print(
            f"{name:10s} Bid={bid:6.2f} Ask={ask:6.2f}  "
            f"PnL total={pnl.sum():10.2f}  PnL medio={pnl.mean():7.4f}"
        )

    # 5. Monte Carlo: 1,000 corridas x 1,000 trades --------------------------
    print("\n=== Monte Carlo: 1,000 corridas x 1,000 trades ===")
    totals_by_regime = {}
    for name, (bid, ask) in regimes.items():
        totals = monte_carlo(MC_RUNS, MC_TRADES, bid, ask, BASE_S0, BASE_PI_I, BASE_PI_L)
        totals_by_regime[name] = totals
        print(
            f"{name:10s} media={totals.mean():10.2f}  "
            f"std={totals.std():8.2f}  "
            f"p5={np.percentile(totals, 5):9.2f}  "
            f"p95={np.percentile(totals, 95):9.2f}"
        )

    # Figuras -----------------------------------------------------------------
    A_range = np.linspace(BASE_S0, BASE_S0 + 10, 100)
    B_range = np.linspace(0.5, BASE_S0, 100)
    loss_ask = [expected_loss_ask(A) for A in A_range]
    loss_bid = [expected_loss_bid(B) for B in B_range]

    plot_price_distribution(
        price_pdf, BASE_S0, save_path=f"{FIGURES_DIR}/precio_distribucion.png"
    )
    plot_loss_functions(
        A_range, loss_ask, B_range, loss_bid,
        save_path=f"{FIGURES_DIR}/perdida_esperada.png",
    )
    plot_pnl_distributions(
        pnl_by_regime, save_path=f"{FIGURES_DIR}/pnl_regimenes.png"
    )
    plot_monte_carlo_totals(
        totals_by_regime, save_path=f"{FIGURES_DIR}/monte_carlo.png"
    )

    print(f"\nFiguras guardadas en '{FIGURES_DIR}/'.")


if __name__ == "__main__":
    main()
