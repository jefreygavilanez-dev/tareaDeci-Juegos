# Análisis de decisiones y juegos — Scripts

Requisitos: Python 3.9+ y `pip install -r requirements.txt` (opcional: Graphviz, comando `dot`, para los diagramas).

| Script | Contenido |
|---|---|
| `01_ahp_pja.py` | Proceso de Jerarquía Analítica: vector propio, CI, CR, síntesis jerárquica |
| `02_arbol_decision.py` | Árbol de decisión bietápico, EV, EVPI, EVSI, utilidad esperada |
| `03_incertidumbre.py` | Maximax, Wald, Hurwicz, Laplace, Savage. Acepta CSV propio: `python 03_incertidumbre.py pagos.csv 0.7` |
| `04_teoria_juegos.py` | Punto de silla y estrategias mixtas por PL primal-dual (`scipy.optimize.linprog`) |

Ejecutar todo: `python run_all.py`. Las figuras se guardan en `figuras/` y los resultados numéricos en `resultados/`.
