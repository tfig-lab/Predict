"""Baselines obrigatórios + modelo GARCH-FHS (vol GARCH com resíduos empíricos)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from analisador.modelos.base import SIGMA, ecdf_acima


class TaxaBase:
    """Fração histórica de janelas positivas; quantis = distribuição empírica incondicional."""
    nome = "taxa_base"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TaxaBase":
        self.y = np.sort(np.asarray(y, dtype=float))
        self.p = float(np.mean(self.y > 0))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.p)

    def predict_quantiles(self, X: pd.DataFrame, quantis: list[float]) -> np.ndarray:
        return np.tile(np.quantile(self.y, quantis), (len(X), 1))


class PasseioAleatorio:
    """Random walk com drift: retorno médio e volatilidade históricos, distribuição normal."""
    nome = "passeio_aleatorio"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PasseioAleatorio":
        self.mu, self.sd = float(np.mean(y)), float(np.std(y) + 1e-12)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), float(norm.cdf(self.mu / self.sd)))

    def predict_quantiles(self, X: pd.DataFrame, quantis: list[float]) -> np.ndarray:
        return np.tile(self.mu + self.sd * norm.ppf(quantis), (len(X), 1))


class MomentumIngenuo:
    """Sinal do retorno de 12 meses excluindo o último mês. A probabilidade é a frequência
    histórica de alta (no treino) condicionada a esse sinal."""
    nome = "momentum"
    coluna = "mom_12_1"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MomentumIngenuo":
        s = np.sign(X[self.coluna].to_numpy())
        y = np.asarray(y, dtype=float)
        self.grupos = {}
        for g in (1.0, -1.0):
            sel = (s == g) if g > 0 else (s <= 0)
            yy = y[sel] if sel.sum() >= 30 else y
            # suavização de Laplace: evita 0% / 100% com poucas observações
            self.grupos[g] = ((np.sum(yy > 0) + 1) / (len(yy) + 2), np.sort(yy))
        self.geral = ((np.sum(y > 0) + 1) / (len(y) + 2), np.sort(y))
        return self

    def _grupo(self, X: pd.DataFrame) -> np.ndarray:
        m = X[self.coluna].to_numpy()
        return np.where(np.isnan(m), 0.0, np.where(m > 0, 1.0, -1.0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        g = self._grupo(X)
        return np.select([g == 1, g == -1], [self.grupos[1.0][0], self.grupos[-1.0][0]], self.geral[0])

    def predict_quantiles(self, X: pd.DataFrame, quantis: list[float]) -> np.ndarray:
        g = self._grupo(X)
        out = np.empty((len(X), len(quantis)))
        for chave, (_, amostra) in [(1.0, self.grupos[1.0]), (-1.0, self.grupos[-1.0]), (0.0, self.geral)]:
            out[g == chave] = np.quantile(amostra, quantis)
        return out


class GarchFHS:
    """r_h = μ_h + σ_h · ε, com σ_h da estrutura a termo do GARCH (coluna `_sigma_h`) e ε da
    distribuição empírica dos resíduos padronizados do treino (filtered historical simulation)."""
    nome = "garch"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GarchFHS":
        sig = X[SIGMA].to_numpy()
        ok = np.isfinite(sig) & (sig > 0)
        y = np.asarray(y, dtype=float)
        self.mu = float(np.mean(y))
        self.eps = np.sort((y[ok] - self.mu) / sig[ok])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        sig = X[SIGMA].to_numpy()
        return ecdf_acima(self.eps, -self.mu / sig)

    def predict_quantiles(self, X: pd.DataFrame, quantis: list[float]) -> np.ndarray:
        sig = X[SIGMA].to_numpy()[:, None]
        return self.mu + sig * np.quantile(self.eps, quantis)[None, :]


BASELINES = (TaxaBase, PasseioAleatorio, MomentumIngenuo)
