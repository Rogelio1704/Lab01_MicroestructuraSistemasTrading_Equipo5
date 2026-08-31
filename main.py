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
    LIQUIDITY_INTERCEPT,
    LIQUIDITY_SLOPE,
    execution_prob,
    expected_loss_ask,
    expected_loss_bid,
    optimize_quotes,
    price_pdf,
)
from src.plots import (
    plot_execution_probability,
    plot_inventory_paths,
    plot_loss_functions,
    plot_monte_carlo_totals,
    plot_pnl_distributions,
    plot_price_distribution,
    plot_sensitivity_pi_I,
)
from src.simulation import monte_carlo, run_regime

SEED = 42
N_TRADES = 10_000
MC_RUNS = 1_000
MC_TRADES = 1_000
SENSITIVITY_PI_I = [0.1, 0.4, 0.7]
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
    inventory_by_regime = {}
    for name, (bid, ask) in regimes.items():
        stats_regime = run_regime(N_TRADES, bid, ask, BASE_S0, BASE_PI_I, BASE_PI_L)
        pnl_by_regime[name] = stats_regime["pnl"]
        inventory_by_regime[name] = stats_regime["inventory_path"]
        print(
            f"{name:10s} Bid={bid:6.2f} Ask={ask:6.2f}  "
            f"PnL total={stats_regime['total_pnl']:10.2f}  "
            f"PnL medio={stats_regime['mean_pnl']:7.4f}  "
            f"Inv final={stats_regime['final_inventory']:8.0f}  "
            f"|Inv| max={stats_regime['max_abs_inventory']:8.0f}"
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

    # 6. Analisis de sensibilidad: spread optimo vs pi_I ---------------------
    print("\n=== Sensibilidad: spread optimo vs pi_I ===")
    sensitivity_spreads = []
    for pi_I in SENSITIVITY_PI_I:
        pi_L = 1.0 - pi_I
        result_pi_I = optimize_quotes(S0=BASE_S0, pi_I=pi_I, pi_L=pi_L)
        sensitivity_spreads.append(result_pi_I["spread"])
        print(
            f"pi_I={pi_I:4.2f}  Bid={result_pi_I['bid']:6.2f}  "
            f"Ask={result_pi_I['ask']:6.2f}  Spread={result_pi_I['spread']:6.2f}"
        )

    # Figuras -----------------------------------------------------------------
    A_range = np.linspace(BASE_S0, BASE_S0 + 10, 100)
    B_range = np.linspace(0.5, BASE_S0, 100)
    loss_ask = [expected_loss_ask(A) for A in A_range]
    loss_bid = [expected_loss_bid(B) for B in B_range]

    zero_spread = LIQUIDITY_INTERCEPT / LIQUIDITY_SLOPE
    spread_range = np.linspace(0, zero_spread + 1, 200)
    exec_prob = execution_prob(spread_range)

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
    plot_execution_probability(
        spread_range, exec_prob, zero_spread,
        save_path=f"{FIGURES_DIR}/probabilidad_ejecucion.png",
    )
    plot_inventory_paths(
        inventory_by_regime, save_path=f"{FIGURES_DIR}/inventario_acumulado.png"
    )
    plot_sensitivity_pi_I(
        SENSITIVITY_PI_I, sensitivity_spreads,
        save_path=f"{FIGURES_DIR}/sensibilidad_pi_I.png",
    )

    print(f"\nFiguras guardadas en '{FIGURES_DIR}/'.")


if __name__ == "__main__":
    main()
