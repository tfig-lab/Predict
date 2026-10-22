"""Interface comum dos modelos.

Todos recebem X (DataFrame) e y = log-retorno acumulado no horizonte. Colunas de X que
começam com "_" são contexto (ex.: `_sigma_h`, a vol GARCH prevista para o horizonte)
e não entram como feature nos modelos de ML.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

SIGMA = "_sigma_h"


class Modelo(Protocol):
    nome: str

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Modelo": ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(y > 0) para cada linha."""
        ...

    def predict_quantiles(self, X: pd.DataFrame, quantis: list[float]) -> np.ndarray:
        """Quantis de y, forma (n, len(quantis)), não decrescentes em cada linha."""
        ...


def colunas_features(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not c.startswith("_")]


def ecdf_acima(amostra_ordenada: np.ndarray, limiar: np.ndarray) -> np.ndarray:
    """P(amostra > limiar) pela distribuição empírica (amostra já ordenada)."""
    n = len(amostra_ordenada)
    pos = np.searchsorted(amostra_ordenada, limiar, side="right")
    return 1 - pos / n
