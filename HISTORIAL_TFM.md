# Historial del TFM — Predicción de terremotos con Deep Learning

Registro técnico del proyecto: contenido de cada notebook, decisiones de datos y
modelado, resultados principales y pruebas descartadas.

---

## Datos y región de estudio

- Catálogo USGS (web service FDSN). Región: California (32–42 °N, −124 a −114 °W) y
  Alaska (51–72 °N, −180 a −130 °W), periodo 2000–2024.
- Magnitud de completitud Mc = 2.5. Objetivo: M ≥ 4.5. Ventana de predicción: 30 días.
  Historia (lookback): 60 días.
- ~135.000 eventos M ≥ 2.5 en el periodo.
- Declustering de Gardner–Knopoff aplicado **solo al objetivo** (mainshocks
  independientes); las características se calculan sobre el catálogo completo.
- Rejilla de 0.5°×0.5°. Se conservan las celdas con ≥30 eventos (~626 activas). El
  objetivo por celda usa expansión 3×3 (≈55 km).
- Particiones temporales: train 2000–2020, validación 2021–2022, test 2023–2024.
  Normalización ajustada solo con train.
- 5 características causales por celda y día: log_count, log_energy (Kanamori),
  max_mag, log_count_M35 y b-value de 30 días (calculado por región).
- Semillas comunes en los modelos multi-semilla: {13, 42, 123, 256, 1024}.

STEAD se usa únicamente para la detección (nb 02). Para el pronóstico se usa USGS; la
justificación empírica está en el nb 07.

---

## Notebooks

### 01 — Exploración de datos

EDA del catálogo USGS de California+Alaska: eventos por año, distribución de
magnitudes (ley de Gutenberg–Richter) y de profundidades. Sirve para fijar Mc y
comprobar la coherencia del catálogo.

### 02 — Detección y picking (PhaseNet, EQTransformer)

- Modelos preentrenados sobre STEAD (vía SeisBench), aplicados en inferencia sobre un
  subconjunto de trazas no usadas en su entrenamiento. No se reentrenan.
- Evaluación por fase (P y S) con varias tolerancias temporales; multi-semilla.
- Resultados (tolerancia 0.1 s): PhaseNet F1 = 0.930 (P) / 0.788 (S); EQTransformer
  F1 = 0.832 (P) / 0.750 (S). Errores de localización de pocas centésimas de segundo.
- Sensibilidad a la tolerancia (PhaseNet, fase P): F1 = 0.930 / 0.951 / 0.963 a
  0.1 / 0.3 / 0.5 s.

### 03 — TCN temporal global

- Red convolucional temporal sobre características agregadas de toda la región (sin
  dimensión espacial). Objetivo binario por día.
- La ventana de 30 días degenera para la región global (objetivo casi siempre
  positivo); se usa una ventana de 14 días.
- Resultado test (multi-semilla): AUC = 0.512 ± 0.052. No supera a la regresión
  logística (0.544) ni a la persistencia (0.513).
- Conclusión: la dimensión temporal global no contiene señal aprovechable. Motiva el
  paso a una formulación espacial (nb 04 y 05).

### 04 — GNN espacial (GCN, con variante GAT)

- TCN compartida por celda → embedding por celda → 2 capas GCN sobre el grafo de
  vecindad → cabeza lineal por celda.
- Grafo: aristas por distancia de Haversine ≤100 km, restringidas a la misma región,
  con fallback KNN (k=2) para evitar nodos aislados.
- Resultado test (multi-semilla): AUC = 0.699 ± 0.008. Queda ligeramente por debajo de
  la climatología de Poisson (0.726).
- Variante GAT (atención sobre vecinos, Veličković et al. 2018): AUC = 0.697. Mejora
  marginal sobre GCN, sin salto cualitativo; no validada multi-semilla.

### 05 — Transformer geoespacial

- Misma TCN compartida + atención multi-cabeza enmascarada por región + codificación
  posicional sinusoidal de las coordenadas (índices de rejilla). D_MODEL = 128, 4
  cabezas, 2 capas.
- Resultado test (multi-semilla): AUC = 0.728 ± 0.002 (ensemble 0.729). Supera al
  Poisson (0.726): t-test 1 muestra t = 2.6, p = 0.029 (1 cola); Wilcoxon pareado
  frente a la GNN p = 0.031.
- Métricas operativas: PR-AUC = 0.111, lift@1% = 4.17.
- AUC temporal por celda = 0.495 ≈ 0.5: el modelo no distingue qué ventanas de 30 días
  son peligrosas dentro de una celda. La capacidad procede de la dimensión espacial,
  no de la temporal. Es el resultado central del trabajo.

### 06 — Mapas de riesgo (Proceso Gaussiano)

- Calibración isotónica de las probabilidades del Transformer (Brier 0.28 → 0.09,
  validada sobre una mitad del test independiente de la usada para calibrar).
- Interpolación con Proceso Gaussiano (kernel Matérn), ajustado por separado en cada
  región. Salida: mapa de riesgo + mapa de incertidumbre para California y Alaska.
- Validación espacial: el mapa separa las celdas con mainshocks reales de las demás con
  AUC = 0.80 y un riesgo medio 2.2× superior en las primeras.

### 07 — Resultados variados (justificación de datos y generalización)

- 7.0 Completitud USGS en Filipinas: Mc ≈ 4.55 (vs 2.5 en California); el 95 % de los
  eventos ya está por encima de M4. El USGS no cubre la baja magnitud fuera de zonas
  con redes densas, por lo que no se pueden añadir regiones como Filipinas para el
  pronóstico (haría falta el catálogo local de PHIVOLCS).
- 7.1 STEAD vs USGS (California): a M ≥ 2.5, STEAD tiene ~3× menos eventos que USGS, su
  muestreo por año oscila ~100× y termina en 2018. Por eso el pronóstico usa USGS.
- 7.2 Transferencia del Transformer (entrenado en CA+Alaska) a 9 regiones, 2 por
  continente, en inferencia: AUC medio = 0.504 (≈ azar); en las 9 regiones queda por
  debajo de su Poisson local. Transfiere algo mejor a las subducciones activas de Asia
  (Japón 0.637, Sumatra 0.545), parecidas a Alaska. El modelo no generaliza: aprende un
  mapa espacial propio de la región de entrenamiento.

---

## Código auxiliar (`code/utils/`)

- `modelos.py`: TCN, GCNLayer / SpatioTemporalGNN, GATLayer / SpatioTemporalGAT,
  SpatioTemporalTransformer, geospatial_encoding, region_attention_mask.
- `evaluacion.py`: metricas_completas (ROC-AUC, PR-AUC, Precision/Recall/Lift@K),
  curva_molchan, auc_temporal_por_celda, baseline_poisson, resumen_multiseed,
  wilcoxon_pareado, ttest_una_muestra.
- `data_loaders.py`: STEADDataset (carga de formas de onda para el nb 02).

---

## Pruebas descartadas (ablaciones)

| Variación | Efecto | AUC temporal |
|---|---|---|
| Arquitectura GCN → GAT | mejora marginal | 0.476 |
| Pooling temporal [último, media, máx] | ligera mejora del AUC global | 0.495 |
| Ventana 30 d → 14 d | empeora el AUC global | 0.501 |
| Catálogo sin declustering | sube el AUC global (réplicas triviales) | 0.509 |
| Características de anomalía por celda | sin mejora | 0.483 |
| Filtro de profundidad ≤70 km | baja el AUC (la profundidad es señal) | 0.463 |
| Profundidad como 6ª característica | baja el AUC (≈0.65) | 0.463 |

En ninguna ablación la AUC temporal por celda se aparta del entorno de 0.5, lo que
confirma que no hay señal precursora temporal a 30 días en el catálogo. Se llegaron a
probar hasta 8 características sin superar a las 5 originales.

---

## Hiperparámetros principales

| | TCN (03) | GNN (04) | Transformer (05) |
|---|---|---|---|
| Canales TCN | (32,32,64,64) | (32,32,64) | (32,32,64) |
| Dim. espacial | — | 64 (GCN) | 128 (D_MODEL) |
| Cabezas / capas atención | — | — | 4 / 2 |
| Dropout | 0.20 | 0.15 | 0.20 |
| LR | 3e-4 | 3e-3 | 1e-4 |
| Batch | 64 | 48 | 32 |
| Épocas máx / paciencia | 80 / 12 | 200 / 6 | 200 / 12 |
| Ventana de predicción | 14 d | 30 d | 30 d |

Comunes: optimizador Adam, ReduceLROnPlateau, weight_decay 1e-4, grad_clip 1.0,
lookback 60 d, pérdida BCE con ponderación de clase, semillas {13,42,123,256,1024}.

---

## Referencias principales

- Zhu & Beroza (2018), *PhaseNet*. GJI 216(1).
- Mousavi et al. (2020), *Earthquake Transformer*. Nature Communications 11.
- Mousavi et al. (2019), *STEAD*. IEEE Access.
- Bai, Kolter & Koltun (2018), *TCN*. arXiv:1803.01271.
- Kipf & Welling (2017), *GCN*. ICLR.
- Veličković et al. (2018), *GAT*. ICLR.
- Vaswani et al. (2017), *Attention is all you need*. NeurIPS.
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*. MIT Press.
- Gardner & Knopoff (1974), declustering. BSSA 64(5).
- van Stiphout et al. (2012), ventanas G-K continuas. CORSSA.
- Aki (1965), estimación del b-value. Bull. Earthq. Res. Inst. 43.
- Kanamori (1977), energía sísmica. JGR 82(20).
- Field et al. (2014), UCERF3. BSSA 104(3).
- Wiemer & Wyss (2000), magnitud de completitud. BSSA 90(4).
- Lin et al. (2017), Focal Loss. ICCV.
- UN (2015), Sendai Framework for Disaster Risk Reduction.
