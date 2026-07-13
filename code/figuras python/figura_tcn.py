"""
Genera una figura ilustrativa de una Temporal Convolutional Network (TCN) 
para la sección 3.4.2 del marco teórico del TFM.

Muestra convoluciones causales y dilatadas, inspirada en la figura de Bai et al. (2018).

Salida: TFM_GUION/figuras/figura_tcn.pdf
        TFM_GUION/figuras/figura_tcn.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------- Config ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "TFM_GUION" / "figuras" / "figura_tcn.pdf"

# Configuración de red
TIME_STEPS = 10
LAYERS = 4  # 0: entrada, 1: d=1, 2: d=2, 3: d=4
K = 2  # Tamaño del kernel

COLOR_NODE_ACTIVE = "#2471a3"      # Azul
COLOR_NODE_INACTIVE = "#bdc3c7"    # Gris
COLOR_CONN_ACTIVE = "#7f6c4d"      # Marrón
COLOR_CONN_INACTIVE = "#d5cab5"    # Marrón claro

# ----------------------------- Estilo ---------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.facecolor": "#f5efe3",
    "figure.facecolor": "#faf6ee",
})

# ----------------------------- Figura ---------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

# Guardamos posiciones de los nodos
# nodes[layer][time_step] = (x, y)
nodes = {}
for l in range(LAYERS):
    nodes[l] = {}
    for t in range(TIME_STEPS):
        nodes[l][t] = (t, l * 1.5)

# Dibujar conexiones
# Para resaltar, vamos a hacer que un nodo de salida (última capa, último t)
# resalte sus dependencias (receptive field).
target_t = TIME_STEPS - 1

def is_in_receptive_field(l, t):
    """Devuelve True si el nodo (l, t) forma parte del receptive field de la salida final."""
    if l == LAYERS - 1:
        return t == target_t
    
    # Calcular las dependencias desde arriba hacia abajo
    # Un nodo en la capa l, tiempo t es dependiente si...
    # Verificamos si podemos llegar desde la capa final
    needed = {target_t}
    for current_l in range(LAYERS - 1, l, -1):
        dilation = 2**(current_l - 1)
        next_needed = set()
        for nt in needed:
            next_needed.add(nt)
            if nt - dilation >= 0:
                next_needed.add(nt - dilation)
        needed = next_needed
    return t in needed

for l in range(1, LAYERS):
    dilation = 2**(l - 1)
    for t in range(TIME_STEPS):
        # Conexión recta (causal, t -> t)
        active = is_in_receptive_field(l, t) and is_in_receptive_field(l-1, t)
        color, alpha, lw = (COLOR_CONN_ACTIVE, 0.8, 1.5) if active else (COLOR_CONN_INACTIVE, 0.4, 0.8)
        
        ax.plot([nodes[l-1][t][0], nodes[l][t][0]], 
                [nodes[l-1][t][1], nodes[l][t][1]], 
                color=color, alpha=alpha, linewidth=lw, zorder=1)
        
        # Conexión dilatada (causal, t-dilation -> t)
        if t - dilation >= 0:
            active = is_in_receptive_field(l, t) and is_in_receptive_field(l-1, t-dilation)
            color, alpha, lw = (COLOR_CONN_ACTIVE, 0.8, 1.5) if active else (COLOR_CONN_INACTIVE, 0.4, 0.8)
            
            ax.plot([nodes[l-1][t-dilation][0], nodes[l][t][0]], 
                    [nodes[l-1][t-dilation][1], nodes[l][t][1]], 
                    color=color, alpha=alpha, linewidth=lw, zorder=1)

# Dibujar nodos
for l in range(LAYERS):
    for t in range(TIME_STEPS):
        x, y = nodes[l][t]
        active = is_in_receptive_field(l, t)
        facecolor = COLOR_NODE_ACTIVE if active else COLOR_NODE_INACTIVE
        edgecolor = "white"
        
        circle = patches.Circle((x, y), 0.2, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=3)
        ax.add_patch(circle)

# Etiquetas
ax.set_xticks(range(TIME_STEPS))
ax.set_xticklabels([f"$t-{TIME_STEPS - 1 - i}$" if i < TIME_STEPS - 1 else "$t$" for i in range(TIME_STEPS)])
ax.set_yticks([l * 1.5 for l in range(LAYERS)])
ax.set_yticklabels(["Entrada", "Oculta 1\n($d=1$)", "Oculta 2\n($d=2$)", "Salida\n($d=4$)"])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)

plt.title("Convoluciones causales dilatadas en TCN ($K=2$)", fontsize=13, color="#3b2f1f", pad=20)
plt.tight_layout()

# ----------------------------- Guardado -------------------------------------
SALIDA.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SALIDA, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(SALIDA.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Guardado: {SALIDA}")
print(f"Guardado: {SALIDA.with_suffix('.png')}")
