"""
Genera la Figura 3.2 del TFM: ley de Gutenberg-Richter comparada entre
California (régimen transformante, San Andreas) y Alaska (subducción
Aleutiana) sobre el catálogo USGS 2010-2024.

Para cada región:
  - se carga el catálogo cacheado (M >= 2.5)
  - se calcula el b-value con el estimador máximo verosímil de Aki (1965)
    corregido por el binning (Utsu 1965; Bender 1983).
  - se ajusta el a-value anclando la recta G-R al punto (Mc, N(>=Mc)).

Se superponen tres curvas teóricas de referencia tomadas de la literatura:
  b = 0.8 (Schorlemmer & Wiemer 2005, alto stress)
  b = 1.0 (Frohlich & Davis 1993, canónico global)
  b = 1.2 (Wiemer & Wyss 2002, bajo stress / volcánico)

Salida: TFM_GUION/figuras/figura_gr_california.pdf a 300 DPI (vector)
        TFM_GUION/figuras/figura_gr_california.png a 300 DPI (rasterizado)

Uso:
    python figura_gr_california.py

Requiere: pandas, numpy, matplotlib.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Config ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "TFM_GUION" / "figuras" / "figura_gr_california.pdf"

MC = 2.5            # magnitud de completitud (Wiemer & Wyss 2000)
DELTA_M = 0.1       # paso de discretización
N_ANOS = 15         # 2010-01-01 a 2024-12-31

REGIONES = {
    "California": {
        "csv": ROOT / "USGS" / "california_2010_2024_M2.5.csv",
        "color": "#c0392b",   # rojo borgoña - régimen transformante
        "tectonica": "transformante (San Andreas)",
    },
    "Alaska": {
        "csv": ROOT / "USGS" / "alaska_2010_2024_M2.5.csv",
        "color": "#2471a3",   # azul profundo - régimen de subducción
        "tectonica": "subducción (Aleutianas)",
    },
}

# Valores de referencia del b-value
B_REF = {
    0.8: ("#7f8c8d", r"$b=0.8$ — alto $\mathit{stress}$ (Schorlemmer & Wiemer 2005)"),
    1.0: ("#34495e", r"$b=1.0$ — canónico (Frohlich & Davis 1993)"),
    1.2: ("#95a5a6", r"$b=1.2$ — bajo $\mathit{stress}$ (Wiemer & Wyss 2002)"),
}


# ----------------------------- Funciones ------------------------------------
def cargar_catalogo(csv):
    df = pd.read_csv(csv)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
    mag = df["mag"].dropna().values
    return mag[mag >= MC]


def ajustar_gr(mag, mc=MC, dm=DELTA_M, n_anos=N_ANOS):
    """Devuelve (a, b) ajustados al catálogo."""
    b = np.log10(np.e) / (mag.mean() - (mc - dm / 2))
    n_acum_mc = len(mag) / n_anos
    a = np.log10(n_acum_mc) + b * mc
    return a, b, n_acum_mc


def curva_gr(M, a, b):
    return 10.0 ** (a - b * M)


# ----------------------------- Carga + ajuste -------------------------------
resultados = {}
for nombre, cfg in REGIONES.items():
    mag = cargar_catalogo(cfg["csv"])
    a, b, n_mc = ajustar_gr(mag)
    resultados[nombre] = {"mag": mag, "a": a, "b": b, "n_mc": n_mc, **cfg}
    print(f"{nombre:10s}: N={len(mag):>6,}  a={a:.2f}  b={b:.3f}")


# ----------------------------- Figura ---------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "text.usetex": False,
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.0,
    "grid.alpha": 0.85,
    "axes.facecolor": "#f5efe3",   # crema cálido suave
    "figure.facecolor": "#faf6ee", # fondo de la figura ligeramente más claro
    "axes.edgecolor": "#7f6c4d",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#3b2f1f",
    "xtick.color": "#3b2f1f",
    "ytick.color": "#3b2f1f",
})

fig, ax = plt.subplots(figsize=(8.0, 5.8))

M_grid = np.linspace(MC, 8.5, 250)

# --- Curvas de referencia (al fondo, gris) ---
# Ancladas al promedio de los dos n_mc para que queden "en medio" visualmente.
n_mc_medio = np.exp(np.mean([np.log(r["n_mc"]) for r in resultados.values()]))
for b_val, (color, label) in B_REF.items():
    a_ref = np.log10(n_mc_medio) + b_val * MC
    ax.plot(M_grid, curva_gr(M_grid, a_ref, b_val),
            color=color, lw=1.2, ls=(0, (4, 3)), alpha=0.55, zorder=2,
            label=label)

# --- Datos empíricos por región ---
for nombre, r in resultados.items():
    bins = np.arange(MC, r["mag"].max() + DELTA_M, DELTA_M)
    n_acum = np.array([(r["mag"] >= m).sum() / N_ANOS for m in bins])

    ax.scatter(bins, n_acum, s=22, color=r["color"], zorder=6,
               edgecolor="white", linewidth=0.5,
               label=(f"{nombre} — {len(r['mag']):,} eventos · "
                      f"{r['tectonica']}"))

    ax.plot(M_grid, curva_gr(M_grid, r["a"], r["b"]),
            color=r["color"], lw=2.4, alpha=0.85, zorder=5,
            label=(f"Ajuste {nombre} (Aki 1965): "
                   f"$a={r['a']:.2f}$, $b={r['b']:.2f}$"))

# --- Líneas guía ---
ax.axhline(1, color="#7f6c4d", lw=0.6, ls=":", alpha=0.7)
ax.text(8.3, 1.18, "1 evento/año", fontsize=8.5, color="#7f6c4d",
        ha="right", style="italic")
ax.axvline(4.5, color="#7f6c4d", lw=0.6, ls=":", alpha=0.7)
ax.text(4.55, 6e3, "Umbral TFM\n$M\\geq 4.5$",
        fontsize=8.5, color="#7f6c4d", style="italic")

# --- Ejes ---
ax.set_yscale("log")
ax.set_xlim(MC, 8.5)
ax.set_ylim(5e-2, 5e4)
ax.set_xlabel("Magnitud $M$", fontsize=12)
ax.set_ylabel(r"Número de eventos $N(M' \geq M)$ por año", fontsize=12)
ax.set_title("Ley de Gutenberg–Richter: California vs Alaska (USGS, 2010–2024)",
             fontsize=13, color="#3b2f1f", pad=10)

# Leyenda con fondo crema más claro
legend = ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5,
                   facecolor="#fbf7ee", edgecolor="#bfa97a")
legend.get_frame().set_linewidth(0.6)

plt.tight_layout()

# ----------------------------- Guardado -------------------------------------
SALIDA.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SALIDA, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig(SALIDA.with_suffix(".png"), dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Guardado: {SALIDA}")
print(f"Guardado: {SALIDA.with_suffix('.png')}")
