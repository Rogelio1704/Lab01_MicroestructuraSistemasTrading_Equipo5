# Lab01 — Cotizaciones Óptimas de un Formador de Mercado

## Integrantes

- Rogelio Adrian Arroyo Valencia — Equipo 5
- Andrea Santoyo Vega — Equipo 5

## Descripción

Modelo de Copeland y Galai (1983): un *dealer* cotiza un
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

Con $\pi_I=0$ (sin informados) el spread óptimo por lado es
$0.50/(2\times0.08)=3.125$ (spread total $0.50/0.08=6.25$), que es
lo que valida el test `test_optimal_spread_without_informed_traders`.

## Estructura

```
main.py                  # orquesta todo el flujo (un solo comando)
src/model.py              # f(P), pérdidas informadas, utilidad y optimización
src/simulation.py         # simulador de trades y Monte Carlo
src/plots.py              # generación de figuras (matplotlib)
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

python main.py            # corre todo: optimizacion, simulacion, figuras y tests
jupyter notebook notebooks/analysis.ipynb   # graficas con interpretaciones
```

`main.py` fija `np.random.seed(42)` de forma global antes de cualquier
simulación, para resultados reproducibles.

## Flujo ejecutado por `main.py`

1. Optimiza Bid/Ask del caso base y reporta Bid, Ask, spread y utilidad
   esperada (redondeados a 2 decimales).
2. Simula 10,000 trades bajo tres regímenes de cotización (registrando
   PnL, cambio de inventario, tipo de trader y dirección del trade):
   - **Óptimo**: resultado de la optimización.
   - **Estrecho**: Bid 19.75 / Ask 20.05.
   - **Amplio**: Bid 18.40 / Ask 21.40.
3. Corre Monte Carlo con 1,000 corridas independientes de 1,000 trades
   para los tres regímenes.
4. Análisis de sensibilidad: reoptimiza Bid/Ask para
   $\pi_I \in \{0.10, 0.40, 0.70\}$.
5. Genera 8 figuras en `figures/` y corre las 3 pruebas de pytest.

### Resultado del caso base (seed=42)

| Bid   | Ask   | Spread | Utilidad esperada |
|-------|-------|--------|--------------------|
| 16.45 | 23.43 | 6.98   | 0.84               |

El **Estrecho** pierde dinero porque los informados lo explotan; el
**Amplio** gana pero menos porque pocos traders de liquidez operan
con precios tan alejados; el **Óptimo** balancea ambos efectos.

## Preguntas de Análisis

### 1. ¿Por qué los traders informados generan la necesidad de un spread?

El informado conoce el precio verdadero P y solo opera cuando le
conviene: compra al Ask si P > A, o vende al Bid si P < B. El dealer
siempre pierde contra un informado.

Si el dealer cotiza pegado a S₀ (régimen Estrecho, spread de 0.30),
casi cualquier desviación de P respecto a S₀ hace que el informado
opere. Nuestras cifras lo confirman: PnL medio de **-0.71 por trade**,
PnL total de **-7,064** en 10,000 trades. En Monte Carlo, el Estrecho
pierde en el **100%** de las corridas.

El spread existe para alejar B y A de S₀ lo suficiente como para que
solo los informados con P muy lejano sigan operando, reduciendo las
pérdidas del dealer.

### 2. ¿Cómo cambia el costo de selección adversa conforme se amplía el spread?

Conforme subes el Ask o bajas el Bid, menos informados encuentran
rentable operar porque necesitan un P cada vez más extremo. La pérdida
esperada del dealer cae. Los resultados lo muestran:

| Régimen  | Spread | PnL medio/trade |
|----------|--------|------------------|
| Estrecho | 0.30   | -0.71            |
| Amplio   | 3.00   | +0.33            |
| Óptimo   | 6.98   | +0.81            |

Pero ampliar el spread también reduce la probabilidad de que los
traders de liquidez operen (llega a cero en s = 6.25). El Óptimo
encuentra el punto donde la ganancia por liquidez menos la pérdida por
informados es máxima.

### 3. ¿Cuál régimen acumula el mayor desbalance de inventario y por qué?

El **Estrecho**: inventario final de **+43** y un máximo absoluto de
**74** unidades, contra +3 (máx 47) del Óptimo y +2 (máx 49) del
Amplio.

¿Por qué? Con spread chico, casi todos los trades se ejecutan (tanto
por informados como por liquidez). Más trades ejecutados significa más
movimientos de inventario y más desbalance acumulado.

Esto expone al dealer a **riesgo de inventario**: si se queda con
muchas unidades y el precio baja, pierde dinero. El modelo no captura
este riesgo porque evalúa cada trade de forma independiente, sin
considerar el valor del inventario abierto.

### 4. ¿Cómo se comporta el spread óptimo al variar πᵢ?

| $\pi_I$ | Bid*  | Ask*  | Spread* |
|---------|-------|-------|---------|
| 0.10    | 16.71 | 23.11 | 6.40    |
| 0.40    | 16.45 | 23.43 | 6.98    |
| 0.70    | 16.01 | 24.00 | 7.99    |

El spread crece conforme sube πᵢ: a más informados, el dealer necesita
protegerse más. Esto coincide con la teoría de Copeland-Galai: el
spread es un mecanismo de compensación por operar contra agentes mejor
informados.

### 5. Tres limitaciones del modelo para un formador de mercado real

1. **La simulación fuerza un trade en cada iteración**: los resultados
   miden rentabilidad por trade, no por unidad de tiempo. Un spread
   muy amplio que en la realidad casi nunca se ejecutaría sale
   favorecido. Además, el dealer no actualiza S₀ con lo que va
   observando del flujo de órdenes.
2. **No hay costo de inventario**: el modelo no penaliza quedarse con
   inventario largo o corto. En un mercado real el riesgo de inventario
   es central. Tampoco captura comisiones, tick sizes ni ejecución
   parcial.
3. **Demanda de liquidez simplificada**: la función max(0, 0.50-0.08x)
   es simétrica y fija. En mercados reales la demanda varía con la
   volatilidad, la hora del día y eventos macro, y no es igual para
   compra que para venta.

## Uso de herramientas de IA

Se utilizó Claude (Anthropic) como asistente durante el desarrollo del
proyecto para revisión de código, depuración y verificación de que la
implementación cumpliera con los requisitos del laboratorio. Todo el
código fue revisado y comprendido por ambos integrantes del equipo.
