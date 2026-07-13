"""
Genera un histograma de magnitudes para el catálogo STEAD.
Sección 3.8 del TFM.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "STEAD" / "features.csv"
SALIDA = ROOT / "TFM_GUION" / "figuras" / "figura_histograma_stead.pdf"

# Estilo
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.facecolor": "#f5efe3",
    "figure.facecolor": "#faf6ee",
})

print(f"Leyendo metadatos desde {CSV_PATH}...")
# Leemos solo las columnas necesarias para ahorrar memoria
df = pd.read_csv(CSV_PATH, usecols=["trace_category", "source_magnitude"], low_memory=False)

# Filtramos solo terremotos y eliminamos NaN en magnitud
df_eq = df[(df["trace_category"] == "earthquake_local") & (df["source_magnitude"].notna())]
magnitudes = pd.to_numeric(df_eq["source_magnitude"], errors='coerce').dropna()

# Creamos la figura
fig, ax = plt.subplots(figsize=(8, 5))

# Dibujar el histograma
bins = np.arange(-0.5, 8.0, 0.1)
ax.hist(magnitudes, bins=bins, color="#2471a3", edgecolor="white", linewidth=0.5, alpha=0.9)

ax.set_xlabel("Magnitud", fontsize=12, color="#3b2f1f")
ax.set_ylabel("Frecuencia (número de sismogramas)", fontsize=12, color="#3b2f1f")
ax.set_title("Distribución de magnitudes en el catálogo STEAD", fontsize=13, color="#3b2f1f", pad=15)

# Escala logarítmica en el eje Y (Ley de Gutenberg-Richter)
ax.set_yscale("log")

# Mejorar el estilo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5, color="#7f6c4d")

# Poner ticks cada 1 magnitud
ax.set_xticks(np.arange(0, 9, 1))

plt.tight_layout()

# Guardado
SALIDA.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SALIDA, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(SALIDA.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Guardado: {SALIDA}")
print(f"Guardado: {SALIDA.with_suffix('.png')}")
