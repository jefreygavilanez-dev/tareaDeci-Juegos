# -*- coding: utf-8 -*-
"""
Subtema 3: Decisión bajo incertidumbre
Criterios: Maximax, Maximin (Wald), Hurwicz, Laplace y Savage.

Uso:
  python 03_incertidumbre.py                 # matriz de ejemplo, alfa = 0.6
  python 03_incertidumbre.py pagos.csv 0.7   # matriz propia (CSV) y alfa

Formato del CSV: primera fila = nombres de estados (con celda inicial vacía);
primera columna = nombres de alternativas; el resto, pagos (a maximizar).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG = BASE / "figuras"
RES = BASE / "resultados"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Matriz de pagos de ejemplo (utilidad en miles de USD)
EJEMPLO = pd.DataFrame(
    [[120, 60, -20, -80],
     [90, 70, 10, -40],
     [60, 50, 30, 0],
     [25, 25, 20, 15]],
    index=["A1 Expandir capacidad", "A2 Nuevo producto", "A3 Alianza estratégica", "A4 Mantener"],
    columns=["Boom", "Crecimiento", "Estancamiento", "Recesión"])


def evaluar(pagos: pd.DataFrame, alfa: float = 0.6):
    """Recibe la matriz de pagos (maximización) y devuelve:
       (tabla comparativa, resumen de decisiones, matriz de arrepentimiento)."""
    M = pagos.to_numpy(dtype=float)
    alt = list(pagos.index)

    maximax = M.max(axis=1)
    maximin = M.min(axis=1)
    hurwicz = alfa * maximax + (1 - alfa) * maximin
    laplace = M.mean(axis=1)
    regret = M.max(axis=0) - M            # costo de oportunidad
    max_regret = regret.max(axis=1)       # Savage: minimizar el máximo arrepentimiento

    tabla = pd.DataFrame({
        "Maximax": maximax, "Maximin (Wald)": maximin,
        f"Hurwicz (α={alfa})": hurwicz, "Laplace": laplace,
        "Savage (máx. arrepentimiento)": max_regret}, index=alt)

    elegir = lambda s, f: alt[int(f(s))]
    resumen = pd.DataFrame({
        "Criterio": tabla.columns,
        "Regla": ["max de máximos", "max de mínimos", "max α·máx+(1-α)·mín",
                  "max promedio", "min de máx. arrepentimiento"],
        "Alternativa óptima": [elegir(maximax, np.argmax), elegir(maximin, np.argmax),
                               elegir(hurwicz, np.argmax), elegir(laplace, np.argmax),
                               elegir(max_regret, np.argmin)],
        "Valor": [maximax.max(), maximin.max(), hurwicz.max(), laplace.max(), max_regret.min()]})
    return tabla, resumen, pd.DataFrame(regret, index=alt, columns=pagos.columns)


def barrido_hurwicz(pagos, pasos=101):
    """Alternativa óptima de Hurwicz para cada alfa en [0,1]."""
    M = pagos.to_numpy(float)
    alfas = np.linspace(0, 1, pasos)
    valores = np.array([a * M.max(1) + (1 - a) * M.min(1) for a in alfas])
    return alfas, valores


def figura_hurwicz(pagos, alfa=0.6):
    alfas, vals = barrido_hurwicz(pagos, 401)
    colores = ["#1f3a5f", "#e08a1e", "#2e8b57", "#c0392b", "#7d5ba6"]
    mejor = vals.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ini = 0
    for k in range(1, len(alfas) + 1):          # franjas de color: alternativa ganadora
        if k == len(alfas) or mejor[k] != mejor[ini]:
            ax.axvspan(alfas[ini], alfas[min(k, len(alfas) - 1)], color=colores[mejor[ini]], alpha=0.10)
            ax.text((alfas[ini] + alfas[min(k, len(alfas) - 1)]) / 2, vals.max() * 1.07,
                    "Gana\n" + pagos.index[mejor[ini]].split(" ")[0], ha="center", fontsize=9,
                    color=colores[mejor[ini]], fontweight="bold")
            ini = k
    for k, nombre in enumerate(pagos.index):
        ax.plot(alfas, vals[:, k], lw=2.6, color=colores[k], label=nombre)
    ax.axvline(alfa, color="black", ls=":", lw=1.5)
    ax.text(alfa, vals.min() * 0.95, f" α = {alfa}\n (nuestro caso)", fontsize=9, va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(vals.min() * 1.05, vals.max() * 1.22)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0\n(pesimista)", "0.25", "0.5", "0.75", "1\n(optimista)"])
    ax.set_xlabel("Coeficiente de optimismo α", fontsize=11)
    ax.set_ylabel("Valor de Hurwicz (miles de USD)", fontsize=11)
    ax.set_title("¿Qué conviene elegir según qué tan optimistas seamos?", fontsize=14, fontweight="bold", loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "incertidumbre_hurwicz.png", dpi=170)
    plt.close(fig)


def figura_matrices(pagos, regret):
    import matplotlib.patches as mp
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    cfg = [("¿Cuánto ganamos?", pagos, "RdYlGn", "Verde = ganancia alta; recuadro = lo mejor de cada escenario"),
           ("¿Cuánto nos arrepentiríamos?", regret, "RdYlGn_r", "Rojo = mucho arrepentimiento; recuadro = el peor de cada alternativa")]
    for n, (ax, (t, D, cm, nota)) in enumerate(zip(axs, cfg)):
        V = D.to_numpy(float)
        ax.imshow(V, cmap=cm, aspect="auto")
        ax.set_xticks(range(D.shape[1])); ax.set_xticklabels(D.columns, fontsize=10)
        ax.set_yticks(range(D.shape[0])); ax.set_yticklabels(D.index, fontsize=10)
        ax.xaxis.tick_top()
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                rel = (V[i, j] - V.min()) / (V.max() - V.min())
                ax.text(j, i, f"{V[i, j]:.0f}", ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if (rel < 0.15 or rel > 0.85) else "#1f2d3d")
        if n == 0:
            for j in range(V.shape[1]):
                i = int(V[:, j].argmax())
                ax.add_patch(mp.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#1f3a5f", lw=3))
        else:
            for i in range(V.shape[0]):
                j = int(V[i].argmax())
                ax.add_patch(mp.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#1f3a5f", lw=3))
        ax.set_title(t, fontsize=13, fontweight="bold", pad=34, color="#1f3a5f")
        ax.set_xlabel(nota, fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(FIG / "incertidumbre_matrices.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pagos = pd.read_csv(sys.argv[1], index_col=0)
    else:
        pagos = EJEMPLO
    alfa = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6

    tabla, resumen, regret = evaluar(pagos, alfa)
    pd.set_option("display.width", 160)
    print("=" * 70)
    print("DECISIÓN BAJO INCERTIDUMBRE")
    print("=" * 70)
    print("\nMatriz de pagos:\n", pagos)
    print("\nMatriz de arrepentimiento:\n", regret)
    print("\nValores por criterio:\n", tabla.round(2))
    print("\nTabla comparativa de decisiones óptimas:\n", resumen.round(2).to_string(index=False))

    figura_matrices(pagos, regret)
    figura_hurwicz(pagos, alfa)
    out = {"alfa": alfa, "pagos": pagos.to_dict("split"),
           "regret": regret.to_dict("split"),
           "tabla": tabla.round(4).to_dict("split"),
           "resumen": resumen.round(4).to_dict("records")}
    (RES / "incertidumbre.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFiguras y resultados guardados.")
