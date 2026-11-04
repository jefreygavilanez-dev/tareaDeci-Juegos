# -*- coding: utf-8 -*-
"""
Subtema 1: Toma de decisiones bajo certidumbre
Proceso de Jerarquía Analítica (PJA / AHP) de T. L. Saaty

Problema: selección de la infraestructura tecnológica de una empresa.
  Meta       : Seleccionar la infraestructura TI más conveniente.
  Criterios  : Costo, Rendimiento, Seguridad, Soporte técnico.
  Alternativas: Nube pública, Nube híbrida, Infraestructura local.

Los juicios de comparación apareada son hipotéticos y se expresan en la
escala fundamental de Saaty (1-9).

Salidas: tablas en consola, figuras en ./figuras y resultados en
./resultados/ahp.json (consumidos por el informe).
Ejecución:  python 01_ahp_pja.py
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG = BASE / "figuras"
RES = BASE / "resultados"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Índice aleatorio de consistencia (Saaty) para n = 1..10
RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
      6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# --------------------------------------------------------------------------
# 1. Datos del problema: matrices de comparaciones apareadas
# --------------------------------------------------------------------------
CRITERIOS = ["Costo", "Rendimiento", "Seguridad", "Soporte"]
ALTERNATIVAS = ["Nube pública", "Nube híbrida", "Local (on-premise)"]

# Matriz de criterios (fila i respecto a columna j)
A_CRIT = np.array([
    [1,   2,   1/2, 4],
    [1/2, 1,   1/3, 3],
    [2,   3,   1,   5],
    [1/4, 1/3, 1/5, 1],
])

# Matrices de alternativas, una por criterio
A_ALT = {
    "Costo": np.array([[1, 3, 5], [1/3, 1, 3], [1/5, 1/3, 1]]),
    "Rendimiento": np.array([[1, 1/2, 1/3], [2, 1, 1/2], [3, 2, 1]]),
    "Seguridad": np.array([[1, 1/2, 1/5], [2, 1, 1/3], [5, 3, 1]]),
    "Soporte": np.array([[1, 3, 4], [1/3, 1, 2], [1/4, 1/2, 1]]),
}


# --------------------------------------------------------------------------
# 2. Funciones de cálculo
# --------------------------------------------------------------------------
def vector_propio(A):
    """Vector de prioridad (vector propio principal normalizado) y lambda_max."""
    valores, vectores = np.linalg.eig(A)
    k = np.argmax(valores.real)
    w = np.abs(vectores[:, k].real)
    return w / w.sum(), float(valores[k].real)


def media_geometrica(A):
    """Aproximación por media geométrica de filas (verificación)."""
    g = np.prod(A, axis=1) ** (1 / A.shape[0])
    return g / g.sum()


def consistencia(A):
    """Devuelve (w, lambda_max, CI, CR)."""
    n = A.shape[0]
    w, lam = vector_propio(A)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    cr = ci / RI[n] if RI[n] > 0 else 0.0
    return w, lam, ci, cr


def verificar_reciprocidad(A):
    return np.allclose(A * A.T, np.ones_like(A))


# --------------------------------------------------------------------------
# 3. Ejecución del modelo
# --------------------------------------------------------------------------
def resolver():
    assert verificar_reciprocidad(A_CRIT), "Matriz de criterios no recíproca"
    w_c, lam_c, ci_c, cr_c = consistencia(A_CRIT)
    out = {"criterios": CRITERIOS, "alternativas": ALTERNATIVAS,
           "A_crit": A_CRIT.tolist(),
           "crit": {"w": w_c.tolist(), "w_geom": media_geometrica(A_CRIT).tolist(),
                    "lambda_max": lam_c, "CI": ci_c, "CR": cr_c, "RI": RI[4]},
           "alt": {}}
    M = np.zeros((len(ALTERNATIVAS), len(CRITERIOS)))   # prioridades locales
    for j, c in enumerate(CRITERIOS):
        A = A_ALT[c]
        assert verificar_reciprocidad(A), f"Matriz '{c}' no recíproca"
        w, lam, ci, cr = consistencia(A)
        M[:, j] = w
        out["alt"][c] = {"A": A.tolist(), "w": w.tolist(),
                         "lambda_max": lam, "CI": ci, "CR": cr, "RI": RI[3]}
    # Síntesis jerárquica: prioridad global = M · w_criterios
    global_ = M @ w_c
    out["M_local"] = M.tolist()
    out["global"] = global_.tolist()
    out["ranking"] = [ALTERNATIVAS[i] for i in np.argsort(-global_)]
    out["todas_consistentes"] = bool(
        cr_c < 0.10 and all(v["CR"] < 0.10 for v in out["alt"].values()))
    return out, M, w_c, global_


# --------------------------------------------------------------------------
# 4. Figuras
# --------------------------------------------------------------------------
def figura_matrices():
    """Mapas de calor de las comparaciones de a dos, con guía de lectura."""
    from fractions import Fraction
    import textwrap
    mats = [("¿Qué criterio importa más?", A_CRIT, CRITERIOS)] + \
           [(f"Según {c.lower()}: ¿qué alternativa es mejor?", A_ALT[c], ALTERNATIVAS) for c in CRITERIOS]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.6))
    for ax, (t, A, et) in zip(axs.flat, mats):
        ax.imshow(np.log(A), cmap="Blues", vmin=-2.2, vmax=2.2)
        etq = [textwrap.fill(e, 12, break_on_hyphens=False) for e in et]
        ax.set_xticks(range(len(et))); ax.set_yticks(range(len(et)))
        ax.set_xticklabels(etq, fontsize=9); ax.set_yticklabels(etq, fontsize=9)
        ax.xaxis.tick_top()
        for i in range(len(et)):
            for j in range(len(et)):
                v = Fraction(A[i, j]).limit_denominator(9)
                txt = str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
                oscuro = np.log(A[i, j]) > 0.9
                ax.text(j, i, txt, ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if oscuro else "#1f2d3d")
        ax.set_title(t, fontsize=11, pad=34, color="#1f3a5f", fontweight="bold")
    ax = axs.flat[5]; ax.axis("off")
    ax.text(0.0, 0.95, "Cómo leer estas tablas", fontsize=13, fontweight="bold", color="#1f3a5f", va="top")
    ax.text(0.0, 0.82,
            "Cada número compara la FILA con la COLUMNA:\n\n"
            "   1  →  son igual de importantes\n"
            "   3  →  la fila es algo mejor\n"
            "   5  →  la fila es bastante mejor\n"
            "   7  →  la fila es mucho mejor\n"
            "   9  →  la fila es extremadamente mejor\n\n"
            "Una fracción (por ejemplo 1/3) significa lo\ncontrario: la columna es la que gana.\n\n"
            "Cuanto más oscuro el azul, más se prefiere\nla fila sobre la columna.",
            fontsize=11, va="top", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.8", fc="#f3f7fb", ec="#9fb7d1"))
    fig.suptitle("Las comparaciones que hicimos, de a dos", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(h_pad=3.5)
    fig.savefig(FIG / "ahp_matrices.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figura_prioridades(M, w_c, global_):
    """Barras apiladas: cuánto aporta cada criterio al puntaje final."""
    orden = list(np.argsort(global_))            # la mejor queda arriba
    colores = ["#1f3a5f", "#4f86c6", "#f2a541", "#8cc7a1"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    izq = np.zeros(len(ALTERNATIVAS))
    for j, c in enumerate(CRITERIOS):
        contrib = np.array([M[i, j] * w_c[j] for i in orden]) * 100
        ax.barh([ALTERNATIVAS[i] for i in orden], contrib, left=izq, color=colores[j],
                edgecolor="white", label=f"{c} (importancia {w_c[j]*100:.0f} %)")
        for k, v in enumerate(contrib):
            if v > 2.5:
                ax.text(izq[k] + v / 2, k, f"{v:.1f}", ha="center", va="center",
                        color="white" if j < 2 else "#1f2d3d", fontsize=10, fontweight="bold")
        izq += contrib
    mejor = int(np.argmax(global_))
    for k, i in enumerate(orden):
        ax.text(izq[k] + 0.8, k, f"{global_[i]*100:.1f} %" + ("  ★ mejor opción" if i == mejor else ""),
                va="center", fontsize=11, fontweight="bold", color="#2e8b57" if i == mejor else "#333333")
    ax.set_xlim(0, max(izq) * 1.35)
    ax.set_xlabel("Puntaje final (%)", fontsize=11)
    ax.set_title("¿Qué infraestructura nos conviene más?", fontsize=14, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.02, "Cada color muestra cuánto aporta un criterio al puntaje final", transform=ax.transAxes,
            fontsize=10, color="#555555", va="bottom")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "ahp_prioridades.png", dpi=170)
    plt.close(fig)


def figura_jerarquia(w_c, global_):
    """Mapa de la decisión con Graphviz: flechas más gruesas = criterios más importantes."""
    if not shutil.which("dot"):
        print("Aviso: Graphviz no disponible; se omite el diagrama de la jerarquía.")
        return
    mejor = int(np.argmax(global_))
    L = ['digraph G { rankdir=TB; nodesep=0.5; ranksep=0.9; '
         'node [fontname="DejaVu Sans", fontsize=13, shape=box, style="rounded,filled", margin="0.25,0.15"];',
         'edge [color="#7a8899", arrowsize=0.8];',
         'meta [label="META\\nElegir la mejor infraestructura de TI", fillcolor="#1f3a5f", fontcolor=white, fontsize=15];']
    for j, c in enumerate(CRITERIOS):
        L.append(f'c{j} [label="{c}\\nimportancia: {w_c[j]*100:.1f} %", fillcolor="#cfe0f1", color="#4f86c6"];')
        L.append(f'meta -> c{j} [penwidth={1 + 9 * w_c[j]:.2f}, color="#4f86c6"];')
    for i, a in enumerate(ALTERNATIVAS):
        if i == mejor:
            L.append(f'a{i} [label="★ {a}\\npuntaje final: {global_[i]*100:.1f} %\\n(mejor opción)", '
                     'fillcolor="#d8f0e0", color="#2e8b57", penwidth=3];')
        else:
            L.append(f'a{i} [label="{a}\\npuntaje final: {global_[i]*100:.1f} %", fillcolor="#fff4d6", color="#d9a441"];')
        for j in range(len(CRITERIOS)):
            L.append(f'c{j} -> a{i};')
    L.append('label="Cuanto más gruesa la flecha, más importante es el criterio"; labelloc=b; '
             'fontname="DejaVu Sans"; fontsize=11; fontcolor="#555555";')
    L.append("}")
    dot = FIG / "ahp_jerarquia.dot"
    dot.write_text("\n".join(L), encoding="utf-8")
    subprocess.run(["dot", "-Tpng", "-Gdpi=170", str(dot), "-o", str(FIG / "ahp_jerarquia.png")], check=True)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    out, M, w_c, global_ = resolver()

    print("=" * 70)
    print("PROCESO DE JERARQUÍA ANALÍTICA (PJA)")
    print("=" * 70)
    print("\nCriterios:", {k: round(float(v), 4) for k, v in zip(CRITERIOS, w_c)})
    c = out["crit"]
    print(f"  lambda_max = {c['lambda_max']:.4f} | CI = {c['CI']:.4f} | "
          f"RI = {c['RI']} | CR = {c['CR']:.4f} "
          f"-> {'CONSISTENTE' if c['CR'] < 0.10 else 'REVISAR JUICIOS'}")
    print("  Verificación (media geométrica):", np.round(c["w_geom"], 4))
    for crit, d in out["alt"].items():
        print(f"\nAlternativas según {crit}:", {k: round(float(v), 4) for k, v in zip(ALTERNATIVAS, d['w'])})
        print(f"  lambda_max = {d['lambda_max']:.4f} | CI = {d['CI']:.4f} | "
              f"CR = {d['CR']:.4f} -> {'CONSISTENTE' if d['CR'] < 0.10 else 'REVISAR'}")
    print("\nPrioridades globales:")
    for a, g in sorted(zip(ALTERNATIVAS, global_), key=lambda t: -t[1]):
        print(f"  {a:<22s} {g:.4f}")
    print("\nDecisión recomendada:", out["ranking"][0])

    figura_matrices()
    figura_prioridades(M, w_c, global_)
    figura_jerarquia(w_c, global_)
    (RES / "ahp.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print("\nFiguras y resultados guardados.")
