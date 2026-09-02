"""Generacion de figuras para el laboratorio de cotizaciones optimas.

Solo contiene logica de graficacion: recibe datos ya calculados por
main.py o por el notebook y produce/guarda las figuras. matplotlib puro,
estilo limpio (seaborn-v0_8-whitegrid con fallback a 'default').
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("default")

plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "legend.frameon": False,
        "font.size": 10,
    }
)

REGIME_COLORS = {
    "Optimo": "#2a6f97",
    "Estrecho": "#e07a5f",
    "Amplio": "#6a994e",
}


def _save(fig, save_path):
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_price_distribution(price_pdf_fn, S0, p_min=5, p_max=40, save_path=None):
    """Grafica la densidad f(P) del precio verdadero."""
    P = np.linspace(p_min, p_max, 500)
    fig, ax = plt.subplots()
    ax.plot(P, price_pdf_fn(P), color="#2a6f97", lw=2)
    ax.axvline(S0, color="#333333", ls="--", lw=1, label=f"$S_0$ = {S0:.2f}")
    ax.set_title("Distribucion del precio verdadero  f(P) ~ Erlang(K=60, $\\lambda$=3)")
    ax.set_xlabel("Precio P")
    ax.set_ylabel("Densidad")
    ax.legend()
    return _save(fig, save_path)


def plot_loss_functions(A_range, loss_ask, B_range, loss_bid, save_path=None):
    """Grafica las perdidas esperadas frente a informados por lado."""
    fig, ax = plt.subplots()
    ax.plot(A_range, loss_ask, label="Perdida esperada Ask  $L_A(A)$", color="#e07a5f", lw=2)
    ax.plot(B_range, loss_bid, label="Perdida esperada Bid  $L_B(B)$", color="#2a6f97", lw=2)
    ax.set_title("Perdida esperada frente a traders informados")
    ax.set_xlabel("Cotizacion")
    ax.set_ylabel("Perdida esperada")
    ax.legend()
    return _save(fig, save_path)


def plot_pnl_distributions(pnl_by_regime, save_path=None):
    """Histograma comparativo del PnL por trade simulado en cada regimen."""
    fig, ax = plt.subplots()
    for name, pnl in pnl_by_regime.items():
        ax.hist(
            pnl,
            bins=60,
            alpha=0.5,
            density=True,
            label=name,
            color=REGIME_COLORS.get(name, "#999999"),
        )
    ax.set_title("Distribucion del PnL por trade (10,000 trades simulados)")
    ax.set_xlabel("PnL por trade")
    ax.set_ylabel("Densidad")
    ax.legend()
    return _save(fig, save_path)


def plot_monte_carlo_totals(totals_by_regime, save_path=None):
    """Histograma de la distribucion de PnL final del analisis de Monte
    Carlo, con los tres regimenes superpuestos."""
    fig, ax = plt.subplots()
    for name, totals in totals_by_regime.items():
        ax.hist(
            totals,
            bins=40,
            alpha=0.5,
            density=True,
            label=name,
            color=REGIME_COLORS.get(name, "#999999"),
        )
    ax.set_title("Monte Carlo: distribucion del PnL final por corrida\n(1,000 corridas x 1,000 trades)")
    ax.set_xlabel("PnL total por corrida")
    ax.set_ylabel("Densidad")
    ax.legend()
    return _save(fig, save_path)


def plot_execution_probability(spread_range, prob, zero_spread, save_path=None):
    """Grafica la probabilidad de ejecucion de un trader de liquidez
    (pi_LB / pi_LS) en funcion del spread respecto a S0, marcando el
    punto donde la probabilidad llega a cero."""
    fig, ax = plt.subplots()
    ax.plot(spread_range, prob, color="#2a6f97", lw=2, label=r"$\pi_{LB}(s)=\pi_{LS}(s)$")
    ax.axvline(zero_spread, color="#e07a5f", ls="--", lw=1.5)
    ax.scatter([zero_spread], [0.0], color="#e07a5f", zorder=5)
    ax.annotate(
        f"s = {zero_spread:.2f}\n(prob = 0)",
        xy=(zero_spread, 0.0),
        xytext=(zero_spread, max(prob) * 0.15),
        ha="center",
        color="#e07a5f",
        fontsize=9,
    )
    ax.set_title("Probabilidad de ejecucion vs. spread respecto a $S_0$")
    ax.set_xlabel("Spread s = |cotizacion - $S_0$|")
    ax.set_ylabel("Probabilidad de ejecucion")
    ax.legend()
    return _save(fig, save_path)


def plot_inventory_paths(inventory_by_regime, save_path=None):
    """Grafica el inventario acumulado del dealer a lo largo de los trades,
    empalmando las curvas de los tres regimenes en los mismos ejes."""
    fig, ax = plt.subplots()
    for name, inventory_path in inventory_by_regime.items():
        ax.plot(
            np.arange(1, len(inventory_path) + 1),
            inventory_path,
            label=name,
            color=REGIME_COLORS.get(name, "#999999"),
            lw=1.2,
        )
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_title("Inventario acumulado del dealer por regimen")
    ax.set_xlabel("Numero de trade")
    ax.set_ylabel("Inventario acumulado")
    ax.legend()
    return _save(fig, save_path)

def plot_cumulative_pnl(pnl_by_regime, save_path=None):
    """PnL acumulado del dealer a lo largo de los trades, con las tres
    curvas de regimen en los mismos ejes."""
    fig, ax = plt.subplots()
    for name, pnl in pnl_by_regime.items():
        cumulative = np.cumsum(pnl)
        ax.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            label=name,
            color=REGIME_COLORS.get(name, "#999999"),
            lw=1.2,
        )
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_title("PnL acumulado del dealer por regimen (10,000 trades)")
    ax.set_xlabel("Numero de trade")
    ax.set_ylabel("PnL acumulado")
    ax.legend()
    return _save(fig, save_path)

def plot_sensitivity_pi_I(pi_I_values, spreads, save_path=None):
    """Grafica el spread optimo resultante de la optimizacion contra el
    valor de pi_I (probabilidad de trader informado)."""
    fig, ax = plt.subplots()
    ax.plot(pi_I_values, spreads, marker="o", color="#6a994e", lw=2)
    ax.set_title(r"Sensibilidad: spread optimo vs. $\pi_I$")
    ax.set_xlabel(r"$\pi_I$ (probabilidad de trader informado)")
    ax.set_ylabel("Spread optimo (A* - B*)")
    return _save(fig, save_path)
