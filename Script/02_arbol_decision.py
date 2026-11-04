# -*- coding: utf-8 -*-
"""
Subtema 2: Toma de decisiones en condiciones de riesgo
Árbol de decisiones multietapa, valor esperado (EV), EVPI, EVSI y utilidad esperada.

Problema (cifras en miles de USD):
  Etapa 1: contratar (costo 20) o no un estudio de mercado.
  Etapa 2: según el resultado del estudio, construir planta grande,
           planta pequeña o no invertir.
  Estados de la naturaleza: demanda Alta (P=0.45) o Baja (P=0.55).
  Fiabilidad del estudio: P(Fav|Alta)=0.80 ; P(Fav|Baja)=0.25.

Salidas: consola, figuras en ./figuras y ./resultados/arbol.json
Ejecución:  python 02_arbol_decision.py
"""
import json
import math
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

# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
P_ALTA = 0.45
P_FAV_ALTA, P_FAV_BAJA = 0.80, 0.25
COSTO_ESTUDIO = 20.0
PAGOS = {  # (demanda alta, demanda baja)
    "Planta grande": (250.0, -150.0),
    "Planta pequeña": (90.0, -10.0),
    "No invertir": (0.0, 0.0),
}


# --------------------------------------------------------------------------
# Estructura de árbol mediante clases (rollback / inducción hacia atrás)
# --------------------------------------------------------------------------
class Nodo:
    def __init__(self, nombre):
        self.nombre = nombre
        self.valor = None      # valor (EV o EU) tras el rollback
        self.mejor = None      # rama óptima (solo nodos de decisión)


class Terminal(Nodo):
    def __init__(self, nombre, pago):
        super().__init__(nombre)
        self.pago = pago


class Azar(Nodo):
    def __init__(self, nombre, ramas):          # ramas: [(etiqueta, prob, hijo)]
        super().__init__(nombre)
        self.ramas = ramas
        assert abs(sum(p for _, p, _ in ramas) - 1) < 1e-9, "Probabilidades no suman 1"


class Decision(Nodo):
    def __init__(self, nombre, ramas):          # ramas: [(etiqueta, hijo)]
        super().__init__(nombre)
        self.ramas = ramas


def rollback(nodo, u=lambda x: x):
    """Inducción hacia atrás. u = función de utilidad (identidad => valor esperado)."""
    if isinstance(nodo, Terminal):
        nodo.valor = u(nodo.pago)
    elif isinstance(nodo, Azar):
        nodo.valor = sum(p * rollback(h, u) for _, p, h in nodo.ramas)
    else:
        vals = [rollback(h, u) for _, h in nodo.ramas]
        k = int(np.argmax(vals))
        nodo.mejor = nodo.ramas[k][0]
        nodo.valor = vals[k]
    return nodo.valor


def construir_arbol():
    # Probabilidades de Bayes
    p_fav = P_FAV_ALTA * P_ALTA + P_FAV_BAJA * (1 - P_ALTA)
    p_alta_fav = P_FAV_ALTA * P_ALTA / p_fav
    p_alta_des = (1 - P_FAV_ALTA) * P_ALTA / (1 - p_fav)

    def invertir(nombre, pa, costo):
        ramas = []
        for alt, (alta, baja) in PAGOS.items():
            azar = Azar(f"Demanda ({alt})", [
                ("Alta", pa, Terminal("", alta - costo)),
                ("Baja", 1 - pa, Terminal("", baja - costo))])
            ramas.append((alt, azar))
        return Decision(nombre, ramas)

    sin = invertir("D2: Invertir (sin estudio)", P_ALTA, 0.0)
    con = Azar("Resultado del estudio", [
        ("Favorable", p_fav, invertir("D2: Invertir | Favorable", p_alta_fav, COSTO_ESTUDIO)),
        ("Desfavorable", 1 - p_fav, invertir("D2: Invertir | Desfavorable", p_alta_des, COSTO_ESTUDIO))])
    raiz = Decision("D1: ¿Estudio de mercado?", [("Sin estudio", sin), ("Con estudio", con)])
    return raiz, dict(p_fav=p_fav, p_alta_fav=p_alta_fav, p_alta_des=p_alta_des)


# --------------------------------------------------------------------------
# Variantes del criterio del valor esperado
# --------------------------------------------------------------------------
def ev_sin_informacion():
    return {a: P_ALTA * h + (1 - P_ALTA) * b for a, (h, b) in PAGOS.items()}


def evpi():
    """Valor esperado de la información perfecta."""
    ev = ev_sin_informacion()
    ev_mejor = max(ev.values())
    ev_con_ip = P_ALTA * max(h for h, _ in PAGOS.values()) + \
        (1 - P_ALTA) * max(b for _, b in PAGOS.values())
    return ev_con_ip, ev_mejor, ev_con_ip - ev_mejor


def u_aversion(R):      # cóncava: aversión al riesgo
    return lambda x: 1 - math.exp(-x / R)


def u_propension(R):    # convexa: propensión al riesgo
    return lambda x: math.exp(x / R) - 1


def equivalente_cierto(u, eu, R, tipo):
    if tipo == "aversion":
        return -R * math.log(1 - eu)
    if tipo == "propension":
        return R * math.log(1 + eu)
    return eu


def analisis_utilidad(raiz):
    out = {}
    casos = [("Neutral al riesgo (EV)", lambda x: x, None, "neutral"),
             ("Aversión (R=100)", u_aversion(100), 100, "aversion"),
             ("Aversión leve (R=300)", u_aversion(300), 300, "aversion"),
             ("Propensión (R=200)", u_propension(200), 200, "propension")]
    for nombre, u, R, tipo in casos:
        v = rollback(raiz, u)
        ce = v if tipo == "neutral" else equivalente_cierto(u, v, R, tipo)
        # decisión en D1 y en D2 de cada rama
        con = raiz.ramas[1][1]
        sin = raiz.ramas[0][1]
        out[nombre] = {"valor": v, "EC": ce, "D1": raiz.mejor,
                       "D2_sin": sin.mejor,
                       "D2_fav": con.ramas[0][2].mejor,
                       "D2_des": con.ramas[1][2].mejor}
    rollback(raiz)  # restaurar EV
    return out


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
def figura_arbol(raiz):
    """Árbol de decisiones con Graphviz y etiquetas en lenguaje natural."""
    nombres = {"D1: ¿Estudio de mercado?": "¿Contratamos el estudio\\nde mercado?",
               "D2: Invertir (sin estudio)": "Sin estudio:\\n¿qué planta construimos?",
               "D2: Invertir | Favorable": "Estudio favorable:\\n¿qué planta construimos?",
               "D2: Invertir | Desfavorable": "Estudio desfavorable:\\n¿qué planta construimos?"}
    VERDE, ROJO = "#2e8b57", "#c0392b"
    lineas = ['digraph T { rankdir=LR; nodesep=0.25; ranksep=0.45;',
              'node [fontname="DejaVu Sans", fontsize=15]; edge [fontname="DejaVu Sans", fontsize=13];']
    cont = [0]

    def nuevo():
        cont[0] += 1
        return f"n{cont[0]}"

    def dibujar(nodo):
        nid = nuevo()
        if isinstance(nodo, Terminal):
            color = VERDE if nodo.pago > 0 else (ROJO if nodo.pago < 0 else "#555555")
            lineas.append(f'{nid} [shape=plaintext, fontsize=16, fontcolor="{color}", '
                          f'label=<<B>{nodo.pago:+.0f}</B>>];')
        elif isinstance(nodo, Azar):
            txt = "Demanda" if nodo.nombre.startswith("Demanda") else "Resultado\\ndel estudio"
            lineas.append(f'{nid} [shape=circle, style=filled, fillcolor="#fff4d6", color="#d9a441", '
                          f'label="{txt}\\nVE = {nodo.valor:.1f}"];')
            es_dem = nodo.nombre.startswith("Demanda")
            for et, p, h in nodo.ramas:
                hid = dibujar(h)
                lab = f"{et} {p:.0%}" if es_dem else f"{et} {p:.1%}"
                lineas.append(f'{nid} -> {hid} [label="{lab}"];')
        else:
            lineas.append(f'{nid} [shape=box, style="rounded,filled", fillcolor="#cfe0f1", color="#4f86c6", '
                          f'label="{nombres.get(nodo.nombre, nodo.nombre)}\\nMejor: {nodo.mejor}\\nVE = {nodo.valor:.1f}"];')
            for et, h in nodo.ramas:
                hid = dibujar(h)
                est = f'color="{VERDE}", penwidth=3.2, fontcolor="{VERDE}", ' if et == nodo.mejor else 'color="#9aa5b1", '
                lineas.append(f'{nid} -> {hid} [{est}label="{et}"];')
        return nid

    rollback(raiz)
    dibujar(raiz)
    lineas.append("}")
    (FIG / "arbol_decision.dot").write_text("\n".join(lineas), encoding="utf-8")
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tpng", "-Gdpi=150", str(FIG / "arbol_decision.dot"),
                        "-o", str(FIG / "arbol_decision.png")], check=True)
    else:
        print("Aviso: Graphviz no disponible; se omite el diagrama del árbol.")


def figura_utilidad():
    x = np.linspace(-170, 250, 300)
    norm = lambda f: (f(x) - f(x).min()) / (f(x).max() - f(x).min())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, norm(lambda t: t), "--", color="#555555", lw=2)
    ax.plot(x, norm(np.vectorize(u_aversion(100))), color="#1f3a5f", lw=3)
    ax.plot(x, norm(np.vectorize(u_propension(200))), color="#c0392b", lw=3)
    ax.text(-60, 0.62, "Le teme a las pérdidas\n(aversión al riesgo)", color="#1f3a5f", fontsize=11, fontweight="bold", ha="center")
    ax.text(185, 0.30, "Le atrae el riesgo\n(propensión)", color="#c0392b", fontsize=11, fontweight="bold", ha="center")
    ax.annotate("Neutral:\ncada dólar\nvale lo mismo", xy=(60, 0.54), xytext=(10, 0.60), color="#555555", fontsize=10,
                va="bottom", arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.axvline(0, color="#bbbbbb", lw=1)
    ax.set_xlabel("Ganancia o pérdida (miles de USD)", fontsize=11)
    ax.set_ylabel("Satisfacción (utilidad, de 0 a 1)", fontsize=11)
    ax.set_title("¿Cómo valora el dinero cada persona?", fontsize=14, fontweight="bold", loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / "arbol_utilidades.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    raiz, bayes = construir_arbol()
    ev_raiz = rollback(raiz)

    print("=" * 70)
    print("ÁRBOL DE DECISIONES - CRITERIO DEL VALOR ESPERADO")
    print("=" * 70)
    print("\nProbabilidades de Bayes:")
    print(f"  P(Favorable)            = {bayes['p_fav']:.4f}")
    print(f"  P(Alta | Favorable)     = {bayes['p_alta_fav']:.4f}")
    print(f"  P(Alta | Desfavorable)  = {bayes['p_alta_des']:.4f}")

    print("\nEV sin información adicional:")
    for a, v in ev_sin_informacion().items():
        print(f"  {a:<16s} {v:8.2f}")

    con = raiz.ramas[1][1]
    print("\nEV en los nodos de decisión (con estudio, neto de costo):")
    for et, _, h in con.ramas:
        print(f"  {et:<13s} -> {h.mejor:<15s} EV = {h.valor:8.2f}")
    print(f"\nValor del nodo raíz = {ev_raiz:.2f} | Decisión óptima D1: {raiz.mejor}")

    ev_ip, ev_mejor, v_ip = evpi()
    evsi = ev_raiz + COSTO_ESTUDIO - ev_mejor
    print(f"\nEV con información perfecta = {ev_ip:.2f}")
    print(f"EVPI  = {v_ip:.2f}  (cota superior del valor de cualquier estudio)")
    print(f"EVSI  = {evsi:.2f}  (costo del estudio = {COSTO_ESTUDIO:.1f}; "
          f"ENGS = {evsi - COSTO_ESTUDIO:.2f})")
    print(f"Eficiencia de la información = {evsi / v_ip:.1%}")

    util = analisis_utilidad(raiz)
    print("\nUTILIDAD ESPERADA")
    print(f"{'Perfil':<26s}{'Valor':>9s}{'EC':>9s}  D1            D2|Fav          D2|Des          D2|Sin")
    for k, d in util.items():
        print(f"{k:<26s}{d['valor']:9.3f}{d['EC']:9.2f}  {d['D1']:<13s} "
              f"{d['D2_fav']:<15s} {d['D2_des']:<15s} {d['D2_sin']}")

    rollback(raiz)
    figura_arbol(raiz)
    figura_utilidad()
    out = {"bayes": bayes, "ev_sin_info": ev_sin_informacion(), "ev_raiz": ev_raiz,
           "decision_D1": raiz.mejor, "ev_info_perfecta": ev_ip, "ev_mejor_sin_info": ev_mejor,
           "EVPI": v_ip, "EVSI": evsi, "costo_estudio": COSTO_ESTUDIO,
           "D2_fav": con.ramas[0][2].mejor, "D2_des": con.ramas[1][2].mejor,
           "EV_fav": con.ramas[0][2].valor, "EV_des": con.ramas[1][2].valor,
           "utilidad": util}
    (RES / "arbol.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFiguras y resultados guardados.")
