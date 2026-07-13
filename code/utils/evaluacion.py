"""
Módulo de evaluación para el pipeline de pronóstico sísmico.

Métricas de evaluación reunidas en un único módulo. Incluye:

  - metricas_completas : ROC-AUC, PR-AUC y Precision@K / Recall@K / Lift@K.
  - curva_molchan      : diagrama de Molchan (τ, ν) + skill (área vs azar).
  - plot_molchan       : figura del diagrama de Molchan a 300 DPI.
  - resumen_multiseed  : media ± std + IC95% sobre varias semillas y ensemble.
  - wilcoxon_pareado   : test de Wilcoxon pareado entre dos modelos.
  - ttest_una_muestra  : t-test de una muestra (AUCs de seeds vs un baseline).

Convención: y_score son logits o probabilidades (sólo importa el orden relativo
para las métricas basadas en ranking; mayor = más riesgo).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# --------------------------------------------------------------------------- #
#  Baseline de Poisson / climatología                                          #
# --------------------------------------------------------------------------- #
def baseline_poisson(y_full, t_train, t_test):
    """
    Baseline de Poisson (tasa base constante por celda, sin información temporal).

    Estima sobre TRAIN la frecuencia empírica de ventanas positivas de cada celda
    --es decir, P(mainshock en los próximos T días | celda), independiente del
    día, consistente con un proceso de Poisson de tasa constante-- y la asigna a
    cada (día, celda) del test. Es la referencia climatológica mínima: un modelo
    solo aporta si supera la tasa de fondo de cada celda.

    El aplanado respeta el orden (día, celda) row-major que usa el bucle de
    evaluación del notebook, de modo que score y test_y quedan alineados.

    :param y_full:  array (n_dias, n_celdas) con el target binario.
    :param t_train: índices temporales del split de train.
    :param t_test:  índices temporales del split de test.
    :return: (y_test_flat, score_flat, p_cell) listos para metricas_completas.
    """
    p_cell      = y_full[t_train].mean(axis=0)        # (n_celdas,) tasa base
    y_test_flat = y_full[t_test].reshape(-1)
    score_flat  = np.tile(p_cell, len(t_test))        # constante por celda
    return y_test_flat, score_flat, p_cell


# --------------------------------------------------------------------------- #
#  Métricas por modelo                                                         #
# --------------------------------------------------------------------------- #
def metricas_completas(y_true, y_score, ks=(0.005, 0.01, 0.05)):
    """
    Calcula el paquete de métricas sobre un conjunto (test) ya predicho.

    :param y_true:  array (N,) con 0/1 reales.
    :param y_score: array (N,) con el score de riesgo (logit o prob).
    :param ks:      fracciones del top a evaluar (0.01 = top 1%).
    :return: dict con roc_auc, pr_auc, base_rate y, por cada k,
             'precision@k', 'recall@k', 'lift@k'.
    """
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n          = len(y_true)
    total_pos  = int(y_true.sum())
    base_rate  = total_pos / n if n else 0.0

    out = {
        "roc_auc":   roc_auc_score(y_true, y_score),
        "pr_auc":    average_precision_score(y_true, y_score),
        "base_rate": base_rate,
    }

    # Ranking de mayor a menor riesgo (una sola vez)
    y_sorted = y_true[np.argsort(-y_score)]
    for k in ks:
        topk        = max(1, int(round(k * n)))
        capturados  = int(y_sorted[:topk].sum())
        precision_k = capturados / topk
        recall_k    = capturados / total_pos if total_pos else 0.0
        lift_k      = precision_k / base_rate if base_rate else 0.0
        tag = f"{k:.1%}".replace(".0%", "%")
        out[f"precision@{tag}"] = precision_k
        out[f"recall@{tag}"]    = recall_k
        out[f"lift@{tag}"]      = lift_k
    return out


# --------------------------------------------------------------------------- #
#  AUC temporal por celda (aísla la señal precursora de la climatología)       #
# --------------------------------------------------------------------------- #
def auc_temporal_por_celda(y_true_flat, y_score_flat, n_celdas, min_pos=5):
    """
    AUC 'temporal' por celda: mide si el score ordena bien los DÍAS de cada celda
    (peligrosos vs tranquilos), aislando la dimensión temporal de la espacial.

    El AUC global sobre todos los pares (celda, día) está dominado por la
    discriminación ENTRE celdas (activas vs tranquilas), que un baseline de tasa
    base ya resuelve. Esta métrica, en cambio, evalúa SÓLO el ranking de días
    DENTRO de cada celda y promedia sobre las celdas. Un modelo de tasa constante
    (Poisson) da exactamente 0.5 aquí, porque asigna el mismo score a todos los
    días de una celda. Por tanto, AUC temporal > 0.5 es evidencia de señal
    precursora real (no climatológica).

    :param y_true_flat:  (n_dias*n_celdas,) target aplanado row-major (día, celda).
    :param y_score_flat: (n_dias*n_celdas,) score aplanado igual.
    :param n_celdas:     número de celdas (para des-aplanar).
    :param min_pos:      mínimo de días positivos en test para evaluar la celda
                         (filtra celdas con muy pocos positivos, AUC ruidoso).
    :return: (auc_media, aucs_por_celda) — sólo celdas con >= min_pos positivos
             y al menos un negativo.
    """
    y = np.asarray(y_true_flat).reshape(-1, n_celdas)
    s = np.asarray(y_score_flat).reshape(-1, n_celdas)
    aucs = []
    for c in range(n_celdas):
        yc = y[:, c]
        npos = int(yc.sum())
        if min_pos <= npos < len(yc):
            aucs.append(roc_auc_score(yc, s[:, c]))
    aucs = np.array(aucs)
    return (float(aucs.mean()) if len(aucs) else float("nan")), aucs


# --------------------------------------------------------------------------- #
#  Diagrama de Molchan                                                          #
# --------------------------------------------------------------------------- #
def curva_molchan(y_true, y_score):
    """
    Curva de Molchan: para cada nivel de alarma, fracción de espacio-tiempo en
    alarma (τ) frente a tasa de fallos (ν = 1 − recall).

    El azar queda sobre la diagonal ν = 1 − τ; un modelo con skill cae por
    debajo. El 'skill' devuelto es el área entre la diagonal y la curva
    (>0 = mejor que el azar, máximo ≈ 0.5).

    :return: (tau, nu, skill)
    """
    y_true = np.asarray(y_true).astype(int)
    y_sorted = y_true[np.argsort(-np.asarray(y_score, dtype=float))]
    n = len(y_true)
    P = int(y_sorted.sum())

    cum_tp = np.concatenate(([0], np.cumsum(y_sorted)))   # (n+1,)
    tau = np.arange(0, n + 1) / n
    nu  = 1.0 - cum_tp / P if P else np.ones(n + 1)
    # Área entre la diagonal (1 − τ) y la curva, por regla del trapecio.
    y_area = (1.0 - tau) - nu
    skill = float(np.sum((y_area[:-1] + y_area[1:]) / 2.0 * np.diff(tau)))
    return tau, nu, skill


def plot_molchan(y_true, y_score, ax=None, label=None, color="C0"):
    """
    Dibuja el diagrama de Molchan (τ vs ν) con la diagonal del azar.
    Devuelve (ax, skill).
    """
    import matplotlib.pyplot as plt
    tau, nu, skill = curva_molchan(y_true, y_score)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    lbl = (label or "modelo") + f"  (skill = {skill:.3f})"
    ax.plot(tau, nu, color=color, lw=2, label=lbl)
    ax.plot([0, 1], [1, 0], "k--", lw=1, label="azar (ν = 1 − τ)")
    ax.set_xlabel(r"$\tau$  (fracción de espacio-tiempo en alarma)")
    ax.set_ylabel(r"$\nu$  (tasa de fallos = 1 − recall)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Diagrama de Molchan")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return ax, skill


# --------------------------------------------------------------------------- #
#  Agregación multi-seed y tests estadísticos                                  #
# --------------------------------------------------------------------------- #
def resumen_multiseed(metricas_por_seed, ic=0.95):
    """
    Agrega una lista de dicts de métricas (uno por seed) en media ± std + IC.

    :param metricas_por_seed: lista de dicts (salidas de metricas_completas).
    :param ic: nivel del intervalo de confianza (0.95 por defecto).
    :return: dict {metrica: {'mean','std','ic_low','ic_high','vals'}}.
    """
    from scipy import stats
    claves = metricas_por_seed[0].keys()
    n = len(metricas_por_seed)
    z = stats.t.ppf(0.5 + ic / 2, df=max(n - 1, 1))   # t de Student
    res = {}
    for k in claves:
        vals = np.array([m[k] for m in metricas_por_seed], dtype=float)
        mean, std = vals.mean(), vals.std(ddof=1) if n > 1 else 0.0
        half = z * std / np.sqrt(n) if n > 1 else 0.0
        res[k] = {"mean": mean, "std": std,
                  "ic_low": mean - half, "ic_high": mean + half,
                  "vals": vals}
    return res


def wilcoxon_pareado(scores_a, scores_b):
    """
    Test de Wilcoxon pareado entre dos modelos evaluados con las MISMAS seeds.
    Compara, por ejemplo, el AUC de dos modelos semilla a semilla.

    :return: dict con statistic, pvalue y la mediana de la diferencia (a − b).
    """
    from scipy.stats import wilcoxon
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    res = wilcoxon(a, b)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
            "mediana_dif": float(np.median(a - b))}


def ttest_una_muestra(scores, baseline):
    """
    t-test de una muestra: contrasta si los AUC de las semillas superan un baseline
    escalar (p. ej. el AUC del modelo frente al de la regresión logística).

    :return: dict con statistic, pvalue (una cola, H1: media > baseline) y media.
    """
    from scipy.stats import ttest_1samp
    x = np.asarray(scores, float)
    res = ttest_1samp(x, baseline)
    # scipy tipa TtestResult.statistic como genérico (_T_co); en runtime es escalar.
    stat  = float(res.statistic)  # type: ignore[arg-type]
    p_two = float(res.pvalue)     # type: ignore[arg-type]
    p_one = p_two / 2 if stat > 0 else 1.0 - p_two / 2
    return {"statistic": stat, "pvalue": float(p_one),
            "media": float(x.mean()), "baseline": float(baseline)}
