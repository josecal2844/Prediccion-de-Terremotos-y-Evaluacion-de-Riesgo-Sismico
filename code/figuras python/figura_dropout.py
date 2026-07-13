"""
Genera una figura ilustrativa del mecanismo de dropout para la sección 3.3.3
del marco teórico del TFM.

Muestra dos paneles lado a lado:
  Izquierda: red neuronal completa (todas las neuronas activas)
  Derecha: la misma red durante un paso de entrenamiento con dropout, donde
           un subconjunto aleatorio de neuronas (y sus conexiones) queda
           desactivado.

Inspirada en la figura canónica de Srivastava et al. (2014).

Salida: TFM_GUION/figuras/figura_dropout.pdf (vector)
        TFM_GUION/figuras/figura_dropout.png (rasterizado, 300 DPI)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Config ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "TFM_GUION" / "figuras" / "figura_dropout.pdf"

LAYER_SIZES = [4, 6, 6, 3]
LAYER_NAMES = ["Entrada", "Oculta 1", "Oculta 2", "Salida"]
DROP_PROB = 0.4
SEED = 42

COLOR_ACTIVE = "#2471a3"      # azul profundo - neurona activa
COLOR_DROPPED = "#bdc3c7"     # gris claro - neurona desactivada
COLOR_CONNECTION = "#7f6c4d"  # marrón tostado - conexión activa
COLOR_DROPPED_CONN = "#d5cab5"

# ----------------------------- Estilo ---------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.facecolor": "#f5efe3",
    "figure.facecolor": "#faf6ee",
})


# ----------------------------- Funciones ------------------------------------
def neuron_positions(layer_sizes):
    positions = []
    max_size = max(layer_sizes)
    for i, size in enumerate(layer_sizes):
        y_start = (max_size - size) / 2
        positions.append([(i * 1.4, y_start + j) for j in range(size)])
    return positions


def dropout_mask(layer_sizes, drop_prob, seed=SEED):
    """True = neurona activa, False = desactivada.
    Las capas de entrada y salida no se hacen dropout en la práctica."""
    rng = np.random.default_rng(seed)
    mask = []
    for li, size in enumerate(layer_sizes):
        if li == 0 or li == len(layer_sizes) - 1:
            mask.append([True] * size)
        else:
            mask.append([bool(rng.random() > drop_prob) for _ in range(size)])
    return mask


def draw_network(ax, layer_sizes, mask=None, title=""):
    positions = neuron_positions(layer_sizes)

    # Conexiones
    for li in range(len(positions) - 1):
        for i, (x1, y1) in enumerate(positions[li]):
            for j, (x2, y2) in enumerate(positions[li + 1]):
                if mask is not None and (not mask[li][i] or not mask[li + 1][j]):
                    color, alpha, lw = COLOR_DROPPED_CONN, 0.5, 0.4
                else:
                    color, alpha, lw = COLOR_CONNECTION, 0.55, 0.7
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                        linewidth=lw, zorder=1)

    # Neuronas
    for li, layer in enumerate(positions):
        for i, (x, y) in enumerate(layer):
            if mask is not None and not mask[li][i]:
                ax.add_patch(plt.Circle((x, y), 0.26, facecolor=COLOR_DROPPED,
                                        edgecolor="white", linewidth=1.0, zorder=3))
                ax.plot([x - 0.14, x + 0.14], [y - 0.14, y + 0.14],
                        color="white", linewidth=2.0, zorder=4)
                ax.plot([x - 0.14, x + 0.14], [y + 0.14, y - 0.14],
                        color="white", linewidth=2.0, zorder=4)
            else:
                ax.add_patch(plt.Circle((x, y), 0.26, facecolor=COLOR_ACTIVE,
                                        edgecolor="white", linewidth=1.2, zorder=3))

    # Etiquetas de capas
    for i, name in enumerate(LAYER_NAMES):
        ax.text(i * 1.4, -1.2, name, ha="center", va="top",
                fontsize=10, color="#3b2f1f", style="italic")

    ax.set_xlim(-0.8, (len(layer_sizes) - 1) * 1.4 + 0.8)
    ax.set_ylim(-1.8, max(layer_sizes))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, color="#3b2f1f", pad=12)


# ----------------------------- Figura ---------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))

draw_network(ax1, LAYER_SIZES, mask=None,
             title="(a) Red completa (evaluación)")

mask = dropout_mask(LAYER_SIZES, DROP_PROB)
draw_network(ax2, LAYER_SIZES, mask=mask,
             title=f"(b) Paso de entrenamiento con dropout ($p={DROP_PROB}$)")

plt.tight_layout()

# ----------------------------- Guardado -------------------------------------
SALIDA.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SALIDA, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig(SALIDA.with_suffix(".png"), dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Guardado: {SALIDA}")
print(f"Guardado: {SALIDA.with_suffix('.png')}")
