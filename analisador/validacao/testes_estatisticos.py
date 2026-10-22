"""Testes de significância adequados a rótulos sobrepostos."""
from __future__ import annotations

import numpy as np
from scipy import stats


def variancia_hac(d: np.ndarray, lags: int) -> float:
    """Variância de longo prazo de Newey-West (núcleo de Bartlett)."""
    d = np.asarray(d, dtype=float) - np.mean(d)
    n = len(d)
    v = float(d @ d) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * float(d[k:] @ d[:-k]) / n
    return max(v, 1e-18)


def diebold_mariano(perda_modelo: np.ndarray, perda_ref: np.ndarray, h: int) -> dict[str, float]:
    """H0: mesma perda esperada. H1 (unicaudal): o modelo tem perda MENOR que a referência.

    Usa variância HAC com h-1 defasagens (rótulos de horizonte h se sobrepõem) e a correção
    de amostra pequena de Harvey-Leybourne-Newbold, com distribuição t de n-1 graus de liberdade.
    """
    d = np.asarray(perda_ref, dtype=float) - np.asarray(perda_modelo, dtype=float)  # > 0 = modelo melhor
    n = len(d)
    if n < 10:
        return {"estatistica": float("nan"), "p_valor": 1.0, "melhora_media": float(np.mean(d)) if n else 0.0}
    var = variancia_hac(d, max(h - 1, 0))
    dm = float(np.mean(d)) / np.sqrt(var / n)
    correcao = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    estat = dm * correcao
    return {"estatistica": float(estat), "p_valor": float(stats.t.sf(estat, df=n - 1)), "melhora_media": float(np.mean(d))}


def binomial_efetivo(acertos: np.ndarray, h: int, p0: float) -> dict[str, float]:
    """Teste binomial unicaudal (acurácia > p0) numa subamostra SEM sobreposição (1 a cada h pregões)."""
    sub = np.asarray(acertos, dtype=bool)[::max(h, 1)]
    k, n = int(sub.sum()), len(sub)
    if n == 0:
        return {"acertos": 0, "n_efetivo": 0, "acuracia": float("nan"), "p0": p0, "p_valor": 1.0}
    res = stats.binomtest(k, n, p0, alternative="greater")
    return {"acertos": k, "n_efetivo": n, "acuracia": k / n, "p0": float(p0), "p_valor": float(res.pvalue)}
