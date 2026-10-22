"""Intervalos conformes: CQR (conformalized quantile regression) + ACI (Gibbs & Candès, 2021).

Para cada par de quantis (ex.: P10/P90 -> 80%), a pontuação de não conformidade é
E_s = max(q_lo(s) - y_s, y_s - q_hi(s)). O intervalo em t é alargado (ou estreitado)
pelo quantil (1 - α_t) das pontuações realizadas recentes. O α_t se adapta online:
α ← α + γ (α_nominal - erro), o que corrige a cobertura após mudanças de regime.
O erro da linha s só é conhecido em s + h, então a atualização é atrasada em h.
"""
from __future__ import annotations

import numpy as np

PARES = {0.50: (0.25, 0.75), 0.80: (0.10, 0.90), 0.90: (0.05, 0.95)}


def conformal_online(
    Q: np.ndarray,
    y: np.ndarray,
    quantis: list[float],
    h: int,
    gamma: float,
    janela: int,
    minimo: int = 100,
) -> tuple[np.ndarray, dict[float, float]]:
    """Q: (n, nq) quantis brutos em ordem temporal; y: realizado (NaN se ainda não realizado).
    Devolve (Q_ajustado, α final por cobertura)."""
    n = len(Q)
    Qa = Q.copy()
    pos = {round(q, 2): i for i, q in enumerate(quantis)}
    alfas_finais = {}
    for cob, (ql, qh) in PARES.items():
        il, ih = pos[ql], pos[qh]
        alfa_nom = 1 - cob
        alfa = alfa_nom
        E = np.maximum(Q[:, il] - y, y - Q[:, ih])         # NaN onde y não realizado
        ajuste = np.zeros(n)
        for t in range(n):
            s = t - h                                        # linha cujo rótulo acaba de se realizar
            if s >= 0 and np.isfinite(E[s]):
                erro = float(E[s] > ajuste[s])
                alfa = alfa + gamma * (alfa_nom - erro)
            fim = t - h + 1
            ini = max(0, fim - janela)
            if fim - ini >= minimo:
                janela_E = E[ini:fim]
                janela_E = janela_E[np.isfinite(janela_E)]
                nivel = np.clip(np.ceil((len(janela_E) + 1) * (1 - alfa)) / len(janela_E), 0.0, 1.0)
                ajuste[t] = np.quantile(janela_E, nivel) if alfa > 0 else janela_E.max()
            Qa[t, il] = Q[t, il] - ajuste[t]
            Qa[t, ih] = Q[t, ih] + ajuste[t]
        alfas_finais[cob] = float(alfa)
    return np.sort(Qa, axis=1), alfas_finais
