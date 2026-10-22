"""Simulação Monte Carlo: GARCH-t paramétrico + bootstrap em blocos dos resíduos (FHS)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analisador.modelos.garch import GarchFinal, simular

QUANTIS_LEQUE = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


@dataclass
class ResultadoMC:
    dias: np.ndarray                   # 1..H
    leque: dict[float, np.ndarray]     # quantil -> log-retorno acumulado por dia
    eventos: dict[int, dict[str, float]]
    n_trajetorias: int


def monte_carlo(final: GarchFinal, horizontes: list[int], n_sim: int, bloco: int, semente: int) -> ResultadoMC:
    rng = np.random.default_rng(semente)
    H = max(horizontes)
    parametrico = simular(final.params, final.s2_1, H, n_sim, rng)
    bootstrap = simular(final.params, final.s2_1, H, n_sim, rng, residuos=final.residuos, bloco=bloco)
    trajetorias = np.vstack([parametrico, bootstrap])
    leque = {q: np.quantile(trajetorias, q, axis=0) for q in QUANTIS_LEQUE}
    eventos = {}
    for h in horizontes:
        r = trajetorias[:, h - 1]
        eventos[h] = {
            "p_alta": float(np.mean(r > 0)),
            "p_mais_10": float(np.mean(r > np.log(1.10))),
            "p_menos_15": float(np.mean(r < np.log(0.85))),
            "p_alta_param": float(np.mean(parametrico[:, h - 1] > 0)),
            "p_alta_bootstrap": float(np.mean(bootstrap[:, h - 1] > 0)),
        }
    return ResultadoMC(np.arange(1, H + 1), leque, eventos, len(trajetorias))
