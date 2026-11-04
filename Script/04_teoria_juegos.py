# -*- coding: utf-8 -*-
"""
Subtema 4: Teoría de juegos de suma cero entre dos personas.

  Parte A: estrategias puras -> punto de silla (maximin = minimax).
  Parte B: estrategias mixtas -> Programación Lineal (primal y dual) con
           scipy.optimize.linprog (HiGHS), equivalente a lpSolve/linprog de R.

Convención: A (jugador fila) maximiza; B (jugador columna) minimiza.
Ejecución:  python 04_teoria_juegos.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG = BASE / "figuras"
RES = BASE / "resultados"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

ESTRATEGIAS_A = ["A1 Precio bajo", "A2 Publicidad", "A3 Innovación"]
ESTRATEGIAS_B = ["B1 Precio bajo", "B2 Publicidad", "B3 Innovación"]

# Pagos = puntos porcentuales de cuota de mercado ganados por la empresa A
JUEGO_SILLA = np.array([[4, 2, 5],
                        [1, -1, 3],
                        [3, 0, 6]], dtype=float)

JUEGO_MIXTO = np.array([[8, -3, 4],
                        [-2, 6, 1],
                        [5, 2, -4]], dtype=float)


# --------------------------------------------------------------------------
# Parte A: punto de silla
# --------------------------------------------------------------------------
def punto_de_silla(P):
    min_filas = P.min(axis=1)
    max_cols = P.max(axis=0)
    maximin, minimax = min_filas.max(), max_cols.min()
    sillas = [(i, j) for i in range(P.shape[0]) for j in range(P.shape[1])
              if P[i, j] == maximin == minimax and P[i, j] == min_filas[i] == max_cols[j]]
    return dict(min_filas=min_filas, max_cols=max_cols, maximin=maximin,
                minimax=minimax, sillas=sillas, existe=maximin == minimax)


# --------------------------------------------------------------------------
# Parte B: estrategias mixtas por Programación Lineal
# --------------------------------------------------------------------------
def resolver_mixto(P):
    """Resuelve el juego mediante el par primal-dual.

    Se desplaza la matriz (P' = P + k, k > 0) para garantizar V' > 0.
    Jugador A (primal):  min  sum(u_i)  s.a.  sum_i P'_ij u_i >= 1 ,  u >= 0
                         V' = 1/sum(u),  x_i = u_i V'
    Jugador B (dual):    max  sum(v_j)  s.a.  sum_j P'_ij v_j <= 1 ,  v >= 0
                         V' = 1/sum(v),  y_j = v_j V'
    Finalmente V = V' - k.
    """
    m, n = P.shape
    k = 1 - P.min() if P.min() <= 0 else 0.0
    Q = P + k

    pa = linprog(c=np.ones(m), A_ub=-Q.T, b_ub=-np.ones(n),
                 bounds=[(0, None)] * m, method="highs")
    pb = linprog(c=-np.ones(n), A_ub=Q, b_ub=np.ones(m),
                 bounds=[(0, None)] * n, method="highs")
    assert pa.success and pb.success, "El LP no tiene solución óptima"

    v_prima = 1 / pa.fun
    x = pa.x * v_prima
    y = pb.x * (1 / (-pb.fun))
    V = v_prima - k
    return dict(k=k, x=x, y=y, V=V, V_A=1 / pa.fun - k, V_B=1 / (-pb.fun) - k,
                u=pa.x, v=pb.x, sum_u=pa.fun, sum_v=-pb.fun,
                # variables duales de HiGHS (precios sombra) como contraste
                duales_A=-pa.ineqlin.marginals)


def verificar(P, x, y, V, tol=1e-8):
    """Comprueba las condiciones de optimalidad del equilibrio."""
    pago_A = x @ P          # pago esperado de cada columna frente a x
    pago_B = P @ y          # pago esperado de cada fila frente a y
    return dict(
        A_garantiza=bool(np.all(pago_A >= V - tol)),
        B_limita=bool(np.all(pago_B <= V + tol)),
        valor_esperado=float(x @ P @ y),
        pago_columnas=pago_A, pago_filas=pago_B)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def figura_matrices(P1, r1, P2):
    import matplotlib.patches as mp
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    et1 = ["A1\nPrecio bajo", "A2\nPublicidad", "A3\nInnovación"]
    et1b = ["B1\nPrecio bajo", "B2\nPublicidad", "B3\nInnovación"]
    datos = [("Juego 1: sí hay punto de silla", P1, et1, et1b), ("Juego 2: no hay punto de silla", P2, ["A1", "A2", "A3"], ["B1", "B2", "B3"])]
    for ax, (t, P, ef, ec) in zip(axs, datos):
        ax.imshow(P, cmap="RdYlGn", vmin=-8, vmax=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(ec, fontsize=10)
        ax.set_yticks(range(3)); ax.set_yticklabels(ef, fontsize=10)
        ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{P[i, j]:.0f}", ha="center", va="center", fontsize=15, fontweight="bold",
                        color="white" if abs(P[i, j]) >= 7 else "#1f2d3d")
        ax.set_title(t, fontsize=13, fontweight="bold", pad=42, color="#1f3a5f")
        ax.set_ylabel("Empresa A (elige fila)", fontsize=10)
    (i, j) = r1["sillas"][0]
    axs[0].add_patch(mp.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="black", lw=4))
    axs[0].set_xlabel(f"Recuadro negro = punto de silla. Valor del juego = {r1['maximin']:.0f}\n"
                      f"(maximin {r1['maximin']:.0f} = minimax {r1['minimax']:.0f})", fontsize=10, labelpad=10)
    r2 = punto_de_silla(P2)
    axs[1].set_xlabel(f"Maximin = {r2['maximin']:.0f} y minimax = {r2['minimax']:.0f}: no coinciden,\n"
                      "por eso hay que mezclar estrategias", fontsize=10, labelpad=10)
    axs[1].xaxis.set_label_position("bottom")
    axs[0].xaxis.set_label_position("bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(0.5, 0.02, "Ganancia de la empresa A (puntos de cuota de mercado). Verde: bueno para A. Rojo: bueno para B.",
             ha="center", fontsize=11, color="#555555")
    fig.savefig(FIG / "juegos_matrices.png", dpi=170)
    plt.close(fig)


def figura_estrategias(sol):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    datos = [("Empresa A: ¿con qué frecuencia usa cada jugada?", sol["x"], ["A1", "A2", "A3"], "#2b5c8a"),
             ("Empresa B: ¿con qué frecuencia usa cada jugada?", sol["y"], ["B1", "B2", "B3"], "#c0392b")]
    for ax, (t, v, et, col) in zip(axs, datos):
        ax.bar(et, v * 100, color=col, width=0.6)
        for i, val in enumerate(v):
            ax.text(i, val * 100 + 1.5, f"{val*100:.1f} %", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probabilidad de elegirla (%)")
        ax.set_title(t, fontsize=11, fontweight="bold", loc="left")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle(f"Estrategias mixtas óptimas. Valor del juego: A gana en promedio {sol['V']:.2f} puntos de cuota",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "juegos_mixtas.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 70)
    print("PARTE A - JUEGO CON PUNTO DE SILLA (ESTRATEGIAS PURAS)")
    print("=" * 70)
    r1 = punto_de_silla(JUEGO_SILLA)
    print("Matriz de pagos (A maximiza):\n", JUEGO_SILLA)
    print("Mínimo por fila  :", r1["min_filas"], "-> maximin =", r1["maximin"])
    print("Máximo por columna:", r1["max_cols"], "-> minimax =", r1["minimax"])
    if r1["existe"]:
        for i, j in r1["sillas"]:
            print(f"Punto de silla en ({ESTRATEGIAS_A[i]}, {ESTRATEGIAS_B[j]}); V = {JUEGO_SILLA[i, j]}")

    print("\n" + "=" * 70)
    print("PARTE B - ESTRATEGIAS MIXTAS POR PROGRAMACIÓN LINEAL")
    print("=" * 70)
    r2 = punto_de_silla(JUEGO_MIXTO)
    print("Matriz de pagos:\n", JUEGO_MIXTO)
    print(f"maximin = {r2['maximin']}  |  minimax = {r2['minimax']}  -> "
          f"{'punto de silla' if r2['existe'] else 'NO existe punto de silla'}")
    sol = resolver_mixto(JUEGO_MIXTO)
    print(f"\nDesplazamiento k = {sol['k']}")
    print("Primal (A): u* =", sol["u"], "| sum(u) =", round(sol["sum_u"], 6))
    print("Dual   (B): v* =", sol["v"], "| sum(v) =", round(sol["sum_v"], 6))
    print("Estrategia óptima de A (p):", sol["x"])
    print("Estrategia óptima de B (q):", sol["y"])
    print(f"Valor del juego: V (primal) = {sol['V_A']:.6f} | V (dual) = {sol['V_B']:.6f}")
    ver = verificar(JUEGO_MIXTO, sol["x"], sol["y"], sol["V"])
    print("Pago esperado de A frente a cada columna:", ver["pago_columnas"])
    print("Pago esperado de B frente a cada fila   :", ver["pago_filas"])
    print("Dualidad fuerte verificada:", np.isclose(sol["V_A"], sol["V_B"]),
          "| Condiciones de equilibrio:", ver["A_garantiza"] and ver["B_limita"])

    figura_matrices(JUEGO_SILLA, r1, JUEGO_MIXTO)
    figura_estrategias(sol)
    out = {"silla": {"P": JUEGO_SILLA.tolist(), "min_filas": r1["min_filas"].tolist(),
                     "max_cols": r1["max_cols"].tolist(), "maximin": r1["maximin"],
                     "minimax": r1["minimax"], "sillas": r1["sillas"]},
           "mixto": {"P": JUEGO_MIXTO.tolist(), "min_filas": r2["min_filas"].tolist(),
                     "max_cols": r2["max_cols"].tolist(), "maximin": r2["maximin"],
                     "minimax": r2["minimax"], "k": sol["k"], "u": sol["u"].tolist(),
                     "v": sol["v"].tolist(), "sum_u": sol["sum_u"], "sum_v": sol["sum_v"],
                     "x": sol["x"].tolist(), "y": sol["y"].tolist(), "V": sol["V"],
                     "pago_columnas": ver["pago_columnas"].tolist(),
                     "pago_filas": ver["pago_filas"].tolist()}}
    (RES / "juegos.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\nFiguras y resultados guardados.")
