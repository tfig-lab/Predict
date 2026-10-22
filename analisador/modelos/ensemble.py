"""Ensemble com pesos recalculados no tempo a partir do desempenho FORA DA AMOSTRA.

Peso do modelo m no pregão t:  w_m ∝ exp(-n_ef · (L_m - min L)),
onde L_m é a perda média (log loss ou pinball) nas previsões cujos rótulos já se
realizaram até t (linhas s <= t - h), numa janela móvel, e n_ef = nº dessas linhas / h
(observações efetivamente independentes). É uma média bayesiana de modelos aproximada:
diferenças pequenas de perda com pouca evidência mantêm os pesos quase iguais.
"""
from __future__ import annotations

import numpy as np


def pesos_online(perdas: np.ndarray, h: int, janela: int, minimo: int, elegivel: np.ndarray | None = None) -> np.ndarray:
    """perdas: (M, n) perda de cada modelo em cada linha fora da amostra (ordem temporal).
    elegivel: (n, M) opcional; modelo não elegível em t recebe peso zero.
    Devolve pesos (n, M) usando só linhas já realizadas."""
    M, n = perdas.shape
    elegivel = np.ones((n, M), dtype=bool) if elegivel is None else elegivel
    acum = np.concatenate([np.zeros((M, 1)), np.cumsum(perdas, axis=1)], axis=1)
    pesos = np.where(elegivel, 1.0, 0.0)
    for t in range(n):
        fim = t - h + 1                     # linhas [ini, fim) realizadas
        ini = max(0, fim - janela)
        cont = fim - ini
        if fim > 0 and cont >= minimo:
            L = (acum[:, fim] - acum[:, ini]) / cont
            n_ef = cont / max(h, 1)
            pesos[t] = np.exp(-n_ef * (L - L[elegivel[t]].min())) * elegivel[t]
        pesos[t] /= pesos[t].sum()
    return pesos


def elegibilidade_online(perdas: np.ndarray, candidato: int, baselines: list[int], h: int, janela: int,
                         minimo: int) -> np.ndarray:
    """Máscara (n, M): o `candidato` só é elegível em t se sua perda realizada na janela for
    menor que a do MELHOR baseline. Os demais modelos são sempre elegíveis."""
    M, n = perdas.shape
    acum = np.concatenate([np.zeros((M, 1)), np.cumsum(perdas, axis=1)], axis=1)
    mask = np.ones((n, M), dtype=bool)
    for t in range(n):
        fim = t - h + 1
        ini = max(0, fim - janela)
        if fim <= 0 or fim - ini < minimo:
            mask[t, candidato] = False
            continue
        L = acum[:, fim] - acum[:, ini]
        mask[t, candidato] = L[candidato] < L[baselines].min()
    return mask


def combinar_probabilidades(P: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """P: (M, n); pesos: (n, M)."""
    return np.einsum("mn,nm->n", P, pesos)


def combinar_quantis(Q: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Média ponderada de quantis (vincentização). Q: (M, n, nq); pesos: (n, M)."""
    return np.sort(np.einsum("mnq,nm->nq", Q, pesos), axis=1)


def concordancia(P: np.ndarray, pesos: np.ndarray, p_ens: np.ndarray) -> np.ndarray:
    """Fração ponderada dos modelos que apontam a mesma direção (vs. 50%) que o ensemble."""
    mesmo = np.sign(P - 0.5) == np.sign(p_ens - 0.5)[None, :]
    return np.einsum("mn,nm->n", mesmo.astype(float), pesos)
