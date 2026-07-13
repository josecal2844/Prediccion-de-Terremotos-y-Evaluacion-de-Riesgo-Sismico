"""
Genera una figura ilustrativa de un mapa continuo de peligrosidad (probabilidad) sísmica,
para la sección 3.7 del TFM.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "TFM_GUION" / "figuras" / "figura_mapa_riesgo.pdf"

# Estilo
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.facecolor": "#f5efe3",
    "figure.facecolor": "#faf6ee",
})

# Generar un grid de datos sintético (probabilidades)
x = np.linspace(-124, -114, 40) # Longitudes simuladas (ej. California, baja resolución para simular grid)
y = np.linspace(32, 42, 40)    # Latitudes simuladas
X, Y = np.meshgrid(x, y)

# Crear algunos puntos calientes (fallas sintéticas)
Z = np.zeros_like(X)
fallas = [
    (-122, 38, 1.5, 2.0),
    (-118, 34, 1.2, 2.5),
    (-116, 33, 0.8, 1.5)
]

for lon, lat, rx, ry in fallas:
    Z += np.exp(-(((X - lon)**2) / rx + ((Y - lat)**2) / ry))

# Añadir algo de ruido para simular predicciones brutas de un modelo
np.random.seed(42)
ruido = np.random.rand(*Z.shape) * 0.4
Z_raw = Z + ruido

# Suavizado para simular el proceso Gaussiano / interpolación espacial
# Aumentamos resolución para el mapa continuo
x_fine = np.linspace(-124, -114, 200)
y_fine = np.linspace(32, 42, 200)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

Z_fine = np.zeros_like(X_fine)
for lon, lat, rx, ry in fallas:
    Z_fine += np.exp(-(((X_fine - lon)**2) / rx + ((Y_fine - lat)**2) / ry))
Z_fine += np.random.rand(*Z_fine.shape) * 0.1 # Menos ruido estructural
Z_smooth = gaussian_filter(Z_fine, sigma=6)

# Normalizar como probabilidad (0 a 1)
Z_raw_prob = Z_raw / np.max(Z_raw)
Z_smooth_prob = Z_smooth / np.max(Z_smooth)

# Graficar
fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0.3})

# Subplot 1: Celda bruta (Probabilidad por celda)
# Usamos un grid visible
im1 = axes[0].pcolormesh(X, Y, Z_raw_prob, cmap='YlOrRd', shading='nearest', edgecolors='white', linewidth=0.1)
axes[0].set_title("Probabilidades crudas por celda\n(Salida discreta del modelo)", fontsize=12, pad=15)
axes[0].set_xlabel("Longitud")
axes[0].set_ylabel("Latitud")

# Subplot 2: Mapa continuo interpolado (GP / Kriging)
im2 = axes[1].contourf(X_fine, Y_fine, Z_smooth_prob, levels=20, cmap='YlOrRd')
axes[1].set_title("Mapa continuo interpolado\n(Post-procesado / GP)", fontsize=12, pad=15)
axes[1].set_xlabel("Longitud")
axes[1].set_ylabel("Latitud")

# Añadir isolíneas al segundo mapa
axes[1].contour(X_fine, Y_fine, Z_smooth_prob, levels=6, colors='black', linewidths=0.5, alpha=0.5)

# Barra de color compartida
cbar = fig.colorbar(im2, ax=axes, orientation='horizontal', fraction=0.06, pad=0.18)
cbar.set_label("Probabilidad espacial de sismicidad ($P$)", fontsize=11)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle("Transición de predicciones discretas a mapas continuos de peligrosidad", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()

SALIDA.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SALIDA, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(SALIDA.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Guardado: {SALIDA}")
