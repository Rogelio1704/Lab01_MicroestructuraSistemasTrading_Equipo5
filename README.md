# Lab01 — Cotizaciones Óptimas de un Formador de Mercado

## Integrantes

- Rogelio Adrian Arroyo Valencia — Equipo 5
- Andrea Santoyo Vega — Equipo 5

## Descripción

Modelo de Copeland y Galai (1983) un *dealer* cotiza un
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
2. Simula 10,000 trades bajo tres regímenes de cotización (registrando
   también el cambio de inventario del dealer en cada trade: +1 si el
   trader vende al bid, -1 si compra al ask, 0 si no opera):
   - **Óptimo**: resultado de la optimización.
   - **Estrecho**: Bid 19.75 / Ask 20.05.
   - **Amplio**: Bid 18.40 / Ask 21.40.
3. Corre un análisis de Monte Carlo con 1,000 corridas independientes
   de 1,000 trades para los tres regímenes.
4. Corre un análisis de sensibilidad, reoptimizando Bid/Ask para
   $\pi_I \in \{0.10, 0.40, 0.70\}$.
5. Genera 7 figuras en `figures/`: distribución de $f(P)$, pérdidas
   esperadas por lado, probabilidad de ejecución vs. spread, distribución
   del PnL por trade por régimen, inventario acumulado por régimen,
   PnL total por corrida de Monte Carlo, y spread óptimo vs. $\pi_I$.

### Resultado del caso base (ejemplo, seed=42)

| Bid   | Ask   | Spread | Utilidad esperada |
|-------|-------|--------|--------------------|
| 16.45 | 23.43 | 6.98   | 0.84               |

El régimen **Estrecho** cotiza tan cerca de $S_0$ que resulta rentable
para los traders informados (adverse selection), dando PnL esperado
negativo para el dealer; el **Amplio** es positivo pero deja utilidad
sobre la mesa al ejecutar muy poca demanda de liquidez; el **Óptimo**
balancea ambos efectos.

## Preguntas de Análisis

### 1. ¿Por qué los traders informados generan la necesidad de un spread?

Un trader informado conoce el precio verdadero $P$ y solo opera cuando
le conviene: compra al ask si $P>A$ (el dealer vende por debajo del
valor real) o vende al bid si $P<B$ (el dealer compra por encima del
valor real). En ambos casos el dealer pierde por construcción; nunca
gana frente a un informado. Si el dealer cotiza pegado a $S_0$ (régimen
**Estrecho**, Bid 19.75 / Ask 20.05) casi cualquier desviación del
precio verdadero respecto a $S_0$ hace que un informado tenga incentivo
a operar, y el resultado simulado lo confirma: PnL medio de
**-0.7064 por trade** (PnL total de **-7,063.64** en 10,000 trades, y
media de **-675.14** por corrida de 1,000 trades en Monte Carlo). El
spread existe precisamente para separar B y A de $S_0$ lo suficiente
como para que solo los informados "muy convencidos" (con $P$ lejos del
rango $[B,A]$) sigan operando, limitando la pérdida esperada por ese
lado.

### 2. ¿Cómo cambia el costo de selección adversa conforme se amplía el spread?

El costo de selección adversa es exactamente $\pi_I\cdot[L_A(A)+L_B(B)]$,
donde $L_A$ y $L_B$ son las integrales `expected_loss_ask` /
`expected_loss_bid`. Ambas son estrictamente decrecientes en la
distancia de la cotización a $S_0$ (`test_expected_loss_ask_is_decreasing_in_A`
lo verifica), porque un ask más alto (o un bid más bajo) sólo puede
ser cruzado por precios cada vez más extremos, con densidad $f(P)$ cada
vez menor. Esto se ve directamente en los tres regímenes simulados:

| Régimen  | Bid   | Ask   | Spread | PnL medio/trade |
|----------|-------|-------|--------|------------------|
| Estrecho | 19.75 | 20.05 | 0.30   | -0.7064          |
| Óptimo   | 16.45 | 23.43 | 6.98   | 0.8069           |
| Amplio   | 18.40 | 21.40 | 3.00   | 0.3338           |

Ampliar el spread reduce el costo de selección adversa, pero también
reduce la probabilidad de ejecución de los traders de liquidez
(término $\pi_{LB}/\pi_{LS}$, que decae linealmente y llega a 0 quando
el spread por lado alcanza $0.50/0.08=6.25$). El spread **Amplio**
(3.00) todavía no es tan ancho como el **Óptimo** (6.98) y por eso deja
utilidad sobre la mesa: reduce pérdidas frente a informados pero
también sacrifica demasiada ejecución de liquidez. El **Óptimo**
resuelve exactamente ese trade-off vía `scipy.optimize.minimize`.

### 3. ¿Cuál régimen acumula el mayor desbalance de inventario y por qué? ¿A qué riesgo lo expone?

El régimen **Estrecho** acumula el mayor desbalance: inventario final
de **+43** y un máximo absoluto de **74** unidades en 10,000 trades,
frente a **+3** (máx. 47) del Óptimo y **+2** (máx. 49) del Amplio.
La razón es que, al cotizar tan cerca de $S_0$, tanto la probabilidad
de ejecución de los traders de liquidez como la fracción de informados
que encuentran ventajoso operar son altas en ambos lados, así que casi
todos los 10,000 trades intentados terminan ejecutándose: el inventario
recorre una caminata aleatoria con muchos más pasos "activos" que en
los otros regímenes, y su varianza (y por tanto su desviación máxima)
crece con el número de pasos. Ese desbalance expone al dealer a
**riesgo de inventario / riesgo de precio**: si el inventario neto
queda largo (o corto) y el precio se mueve en contra antes de poder
cerrarlo, el dealer sufre una pérdida de mark-to-market que **el
modelo actual no captura**, porque cada trade se evalúa de forma
independiente y el PnL simulado nunca se ajusta por el valor del
inventario que queda abierto al final de la sesión.

### 4. ¿Cómo se comporta el spread óptimo al variar $\pi_I$? ¿Coincide con la teoría?

| $\pi_I$ | Bid*  | Ask*  | Spread* |
|---------|-------|-------|---------|
| 0.10    | 16.71 | 23.11 | 6.40    |
| 0.40    | 16.45 | 23.43 | 6.98    |
| 0.70    | 16.01 | 24.00 | 7.99    |

El spread óptimo **crece monótonamente con $\pi_I$**: a mayor
proporción de traders informados, mayor es el peso relativo del
término de pérdida esperada $\pi_I\cdot[L_A+L_B]$ frente al término de
utilidad por liquidez $\pi_L\cdot[\ldots]$, así que el dealer se
protege alejando B y A de $S_0$. Esto coincide exactamente con la
teoría de selección adversa de Glosten–Milgrom / Copeland–Galai vista
en clase: el spread bid-ask es, en esencia, un mecanismo de
compensación por el riesgo de operar contra agentes mejor informados,
y crece con la probabilidad de enfrentarlos.

### 5. Tres limitaciones del modelo Copeland-Galai / Glosten-Milgrom aplicado aquí

1. **Un solo trade por "iteración" (independencia entre trades)**: el
   simulador evalúa cada trade de forma aislada e i.i.d., sin orden
   secuencial ni actualización Bayesiana del precio de referencia
   $S_0$ tras observar el flujo de órdenes. Esto **favorece
   artificialmente spreads amplios**, porque en la realidad el dealer
   ajustaría $S_0$ dinámicamente (aprendiendo del flujo informado) en
   vez de sostener un spread fijo y ancho contra toda la sesión.
2. **No hay costo de inventario ni límites de posición**: como se vio
   en la pregunta 3, el modelo no penaliza quedarse con inventario neto
   largo o corto, ni fuerza al dealer a cerrar posiciones o cubrirse;
   en un mercado real el riesgo de inventario (y el costo de capital
   asociado) es una restricción central en la formación de precios.
   Tampoco captura costos operativos reales como comisiones de bolsa,
   tick sizes discretos, ni el riesgo de ejecución parcial.
3. **Traders de liquidez con demanda determinista y simétrica**: la
   función $\max(0,0.50-0.08x)$ es una simplificación fuerte; en
   mercados reales la demanda de liquidez varía con la volatilidad, la
   hora del día, eventos macro y no es necesariamente simétrica entre
   compra y venta, lo que puede sesgar sistemáticamente el spread
   óptimo calculado aquí.

## Uso de herramientas de IA

Se utilizó Claude (Anthropic) como asistente durante el desarrollo del
proyecto para revisión de código, depuración y verificación de que la
implementación cumpliera con los requisitos del laboratorio. Todo el
código fue revisado y comprendido por ambos integrantes del equipo.
