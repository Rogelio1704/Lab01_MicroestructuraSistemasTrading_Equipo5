# Lab01 — Cotizaciones Óptimas de un Formador de Mercado

Modelo de microestructura tipo Glosten–Milgrom: un *dealer* cotiza un
Bid (B) y un Ask (A) alrededor de un precio de referencia $S_0$. Cada
trader que llega es **informado** (conoce el precio verdadero $P$ y
solo opera si le conviene) con probabilidad $\pi_I$, o de **liquidez**
(no informado) con probabilidad $\pi_L = 1-\pi_I$. El dealer elige B y
A para maximizar su utilidad esperada por trade.

## Modelo

- **Precio verdadero**: $P \sim \text{Erlang}(K=60,\ \lambda=3)$,
  media $=K/\lambda=20$, vía `scipy.stats.erlang`.
- **Demanda no informada**: probabilidad de ejecución lineal y
  simétrica, decreciente en el desvío respecto a $S_0$:
  $\pi_{LB}(x)=\pi_{LS}(x)=\max(0,\ 0.50-0.08x)$.
- **Utilidad esperada por trader que llega**:

$$
\Pi(A,B) = \pi_L\Big[\pi_{LB}(A-S_0)(A-S_0) + \pi_{LS}(S_0-B)(S_0-B)\Big]
           - \pi_I\Big[\int_A^\infty (P-A)f(P)\,dP + \int_0^B (B-P)f(P)\,dP\Big]
$$

  Las integrales de pérdida frente a informados se resuelven con
  `scipy.integrate.quad` (sin aproximaciones discretas).

- **Optimización**: `scipy.optimize.minimize` sobre $-\Pi(A,B)$, con
  $B \in (0, S_0]$ y $A \in [S_0, \infty)$.

- **Caso base**: $S_0=19.90$, $\pi_I=0.40$, $\pi_L=0.60$.

Con $\pi_I=0$ (sin informados) el problema se desacopla en B y A, y el
óptimo analítico por lado es $0.50/(2\times0.08)=3.125$ (spread total
$0.50/0.08=6.25$) — es justo lo que valida
`tests/test_model.py::test_optimal_spread_without_informed_traders`.

## Estructura

```
main.py                  # orquesta todo el flujo (un solo comando)
src/model.py              # f(P), pérdidas informadas, utilidad y optimización
src/simulation.py         # simulador de trades y Monte Carlo
src/plots.py               # generación de figuras (matplotlib)
tests/test_model.py       # pruebas pytest
notebooks/analysis.ipynb  # solo importa funciones de src/ y grafica
requirements.txt
.gitignore
```

Toda la lógica de modelo y simulación vive en `src/`. `main.py` solo
orquesta llamadas; el notebook solo importa y grafica.

## Uso

```bash
pip install -r requirements.txt

python main.py            # corre todo el flujo y guarda figuras en figures/
pytest tests/ -v           # corre las 3 pruebas del modelo
jupyter notebook notebooks/analysis.ipynb
```

`main.py` fija `np.random.seed(42)` de forma global antes de cualquier
simulación, para resultados reproducibles.

## Flujo ejecutado por `main.py`

1. Optimiza Bid/Ask del caso base y reporta Bid, Ask, spread y utilidad
   esperada (redondeados a 2 decimales).
2. Simula 10,000 trades bajo tres regímenes de cotización:
   - **Óptimo**: resultado de la optimización.
   - **Estrecho**: Bid 19.75 / Ask 20.05.
   - **Amplio**: Bid 18.40 / Ask 21.40.
3. Corre un análisis de Monte Carlo con 1,000 corridas independientes
   de 1,000 trades para los tres regímenes.
4. Genera 4 figuras en `figures/`: distribución de $f(P)$, pérdidas
   esperadas por lado, distribución del PnL por trade por régimen, y
   PnL total por corrida de Monte Carlo.

### Resultado del caso base (ejemplo, seed=42)

| Bid   | Ask   | Spread | Utilidad esperada |
|-------|-------|--------|--------------------|
| 16.45 | 23.43 | 6.98   | 0.84               |

El régimen **Estrecho** cotiza tan cerca de $S_0$ que resulta rentable
para los traders informados (adverse selection), dando PnL esperado
negativo para el dealer; el **Amplio** es positivo pero deja utilidad
sobre la mesa al ejecutar muy poca demanda de liquidez; el **Óptimo**
balancea ambos efectos.
