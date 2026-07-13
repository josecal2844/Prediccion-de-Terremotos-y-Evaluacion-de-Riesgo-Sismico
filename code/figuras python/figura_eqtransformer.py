"""
Genera un esquema simplificado de la arquitectura EQTransformer
(Mousavi et al., 2020) para la sección 3.7.1 del marco teórico del TFM.

Es una versión resumida de la Figura 1 del paper original: agrupa las 7
capas convolucionales del encoder en un solo bloque mostrando 3 capas
representativas con MaxPool=2, 3, 5; agrupa también los 5 bloques
residuales en una sola caja; y mantiene visibles los componentes
distintivos (BiLSTM, Transformer/self-attention) y las 3 ramas de salida
(detección, P-pick, S-pick).

Salida: TFM_GUION/figuras/figura_eqtransformer.pdf (vector)
        TFM_GUION/figuras/figura_eqtransformer.png (rasterizado, 300 DPI)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "TFM_GUION" / "figuras" / "figura_eqtransformer.pdf"

# Estilo coherente con el resto de figuras del TFM
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.facecolor": "#f5efe3",
    "figure.facecolor": "#faf6ee",
})

# Paleta de colores por tipo de bloque
COL_INPUT = "#d9c9a3"      # crema oscuro - entrada
COL_CONV = "#a7c4d6"       # azul claro - convolucional
COL_RES = "#c7b5d6"        # lavanda - residual
COL_LSTM = "#f3c98b"       # naranja claro - recurrente
COL_TRANS = "#e8a09a"      # coral - transformer
COL_OUT = "#a4d4a8"        # verde claro - salida
COL_EDGE = "#7f6c4d"
COL_TEXT = "#3b2f1f"


def caja(ax, x, y, w, h, color, titulo, subtitulo=None, fs_titulo=10, fs_sub=8.5):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         linewidth=1.0, edgecolor=COL_EDGE,
                         facecolor=color, zorder=2)
    ax.add_patch(box)
    if subtitulo is None:
        ax.text(x + w / 2, y + h / 2, titulo,
                ha="center", va="center", fontsize=fs_titulo,
                color=COL_TEXT, fontweight="bold", zorder=3)
    else:
        ax.text(x + w / 2, y + h * 0.70, titulo,
                ha="center", va="center", fontsize=fs_titulo,
                color=COL_TEXT, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.30, subtitulo,
                ha="center", va="center", fontsize=fs_sub,
                color=COL_TEXT, style="italic", zorder=3)


def flecha(ax, x1, y1, x2, y2, lw=1.4):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle="-|>", mutation_scale=14,
                          color=COL_EDGE, linewidth=lw, zorder=1)
    ax.add_patch(arr)


fig, ax = plt.subplots(figsize=(7.5, 11.5))

# Layout: bloques centrados horizontalmente, apilados verticalmente
CX = 5.0          # centro horizontal
W_main = 7.4      # ancho de los bloques principales
H_block = 0.85    # alto estándar
GAP = 0.30        # espacio entre cajas

# ---- 1. Entrada ----
y = 13.5
caja(ax, CX - W_main / 2, y, W_main, H_block, COL_INPUT,
     "Sismograma 3-componentes",
     "E, N, Z   ·   6000 muestras × 3 canales   (60 s @ 100 Hz)")

# ---- 2. Encoder convolucional (caja con sub-cajas) ----
y_enc_top = y - GAP - 0.10
H_enc = 3.40
y_enc = y_enc_top - H_enc
caja(ax, CX - W_main / 2, y_enc, W_main, H_enc, "#eadcc4",
     "", subtitulo=None)
# título del bloque
ax.text(CX, y_enc + H_enc - 0.30, "Encoder convolucional",
        ha="center", va="center", fontsize=11, color=COL_TEXT,
        fontweight="bold", zorder=3)
ax.text(CX, y_enc + H_enc - 0.60,
        "extracción jerárquica de características locales",
        ha="center", va="center", fontsize=8.5, color=COL_TEXT,
        style="italic", zorder=3)

# 3 sub-cajas representativas + "..."
sub_w = 5.8
sub_h = 0.42
pool_values = [2, 3, 5]
for i, p in enumerate(pool_values):
    sy = y_enc + H_enc - 1.05 - i * (sub_h + 0.10)
    caja(ax, CX - sub_w / 2, sy, sub_w, sub_h, COL_CONV,
         f"Conv1D    ·    MaxPool = {p}",
         fs_titulo=9.5)
ax.text(CX, y_enc + 0.25,
        r"$\vdots$    (7 capas en total, $\mathrm{MaxPool}\in\{2,3,5,5,4,5,3\}$)",
        ha="center", va="center", fontsize=9,
        color=COL_TEXT, style="italic", zorder=3)

# ---- 3. Bloque residual CNN ----
y_res = y_enc - GAP - H_block
caja(ax, CX - W_main / 2, y_res, W_main, H_block, COL_RES,
     "Bloque residual CNN",
     "5 bloques convolucionales con skip-connections")

# ---- 4. BiLSTM ----
y_lstm = y_res - GAP - H_block
caja(ax, CX - W_main / 2, y_lstm, W_main, H_block, COL_LSTM,
     "BiLSTM",
     "contexto temporal bidireccional")

# ---- 5. Transformer / atención ----
y_trans = y_lstm - GAP - H_block
caja(ax, CX - W_main / 2, y_trans, W_main, H_block, COL_TRANS,
     "Transformer · self-attention",
     "qué partes del sismograma son más informativas")

# ---- 6. Tres ramas de salida ----
y_out = y_trans - GAP - 0.4 - H_block
w_out = 2.10
gap_out = 0.30
total_w = 3 * w_out + 2 * gap_out
x0 = CX - total_w / 2
etiquetas = [("Detección", "$\\sigma(\\cdot)$  evento sí / no"),
             ("P-pick", "$\\sigma(\\cdot)$  llegada onda P"),
             ("S-pick", "$\\sigma(\\cdot)$  llegada onda S")]
for i, (t, s) in enumerate(etiquetas):
    caja(ax, x0 + i * (w_out + gap_out), y_out, w_out, H_block, COL_OUT,
         t, subtitulo=s, fs_titulo=10, fs_sub=8)

# ---- Flechas verticales entre bloques principales ----
arrow_pairs = [
    (y, y_enc_top),                          # entrada -> encoder
    (y_enc, y_res + H_block),                # encoder -> residual
    (y_res, y_lstm + H_block),               # residual -> bilstm
    (y_lstm, y_trans + H_block),             # bilstm -> transformer
]
for y_from, y_to in arrow_pairs:
    flecha(ax, CX, y_from, CX, y_to + 0.02)

# Bifurcación del transformer a las tres ramas
y_bifurc = y_trans - 0.20
flecha(ax, CX, y_trans, CX, y_bifurc)
for i in range(3):
    x_branch = x0 + w_out / 2 + i * (w_out + gap_out)
    # línea horizontal hasta encima de cada caja, luego vertical
    ax.plot([CX, x_branch], [y_bifurc, y_bifurc],
            color=COL_EDGE, linewidth=1.4, zorder=1)
    flecha(ax, x_branch, y_bifurc, x_branch, y_out + H_block + 0.02)

# Limites + ocultar ejes
ax.set_xlim(0.8, 9.2)
ax.set_ylim(y_out - 0.5, 14.7)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()

SALIDA.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SALIDA, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig(SALIDA.with_suffix(".png"), dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Guardado: {SALIDA}")
print(f"Guardado: {SALIDA.with_suffix('.png')}")
