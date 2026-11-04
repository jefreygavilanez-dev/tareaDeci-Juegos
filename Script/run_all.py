# Ejecuta los cuatro módulos en orden.
import subprocess, sys
from pathlib import Path
base = Path(__file__).resolve().parent
for s in ["01_ahp_pja.py", "02_arbol_decision.py", "03_incertidumbre.py", "04_teoria_juegos.py"]:
    print(f"\n>>> {s}")
    subprocess.run([sys.executable, str(base / s)], check=True)
