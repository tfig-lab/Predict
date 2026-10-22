"""LightGBM: classificador de direção e regressões quantílicas, com regularização forte."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from analisador.config import CfgLgbm
from analisador.modelos.base import SIGMA, colunas_features
from analisador.validacao.walkforward import divisao_interna

warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")


@dataclass
class Hiper:
    params: dict
    n_estimators: int
    perda_validacao: float


def selecionar_hiper(X: pd.DataFrame, y_bin: np.ndarray, h: int, cfg: CfgLgbm, semente: int) -> Hiper:
    """CV aninhada SÓ no período de treino: subtreino -> purga de h -> validação temporal.
    Escolhe a configuração da grade com menor log loss e o nº de árvores por early stopping."""
    cols = colunas_features(X)
    fim_sub, ini_val = divisao_interna(len(X), h, cfg.fracao_validacao)
    Xs, ys = X.iloc[:fim_sub][cols], y_bin[:fim_sub]
    Xv, yv = X.iloc[ini_val:][cols], y_bin[ini_val:]
    melhor: Hiper | None = None
    if len(np.unique(ys)) < 2 or len(np.unique(yv)) < 2:
        return Hiper(cfg.grade[0], 50, float("nan"))
    for params in cfg.grade:
        m = lgb.LGBMClassifier(**cfg.comum, **params, random_state=semente, n_jobs=1)
        m.fit(Xs, ys, eval_X=(Xv,), eval_y=(yv,), eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
        perda = float(m.best_score_["valid_0"]["binary_logloss"])
        n_arv = max(int(m.best_iteration_ or cfg.comum["n_estimators"]), 10)
        if melhor is None or perda < melhor.perda_validacao:
            melhor = Hiper(params, n_arv, perda)
    return melhor


class LGBMDirecao:
    nome = "lgbm"

    def __init__(self, cfg: CfgLgbm, h: int, semente: int, hiper: Hiper | None = None):
        self.cfg, self.h, self.semente, self.hiper = cfg, h, semente, hiper
        self.constante: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LGBMDirecao":
        y_bin = (np.asarray(y) > 0).astype(int)
        if len(np.unique(y_bin)) < 2:
            self.constante = float(y_bin.mean())
            return self
        if self.hiper is None:
            self.hiper = selecionar_hiper(X, y_bin, self.h, self.cfg, self.semente)
        comum = {**self.cfg.comum, "n_estimators": self.hiper.n_estimators}
        self.cols = colunas_features(X)
        self.modelo = lgb.LGBMClassifier(**comum, **self.hiper.params, random_state=self.semente, n_jobs=1)
        self.modelo.fit(X[self.cols], y_bin)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.constante is not None:
            return np.full(len(X), self.constante)
        return self.modelo.predict_proba(X[self.cols])[:, 1]

    def predict_quantiles(self, X: pd.DataFrame, quantis: list[float]) -> np.ndarray:
        raise NotImplementedError("use LGBMQuantis para quantis")


class LGBMQuantis:
    """Regressão quantílica do retorno PADRONIZADO pela vol GARCH: ε = (y - μ) / σ_h.
    Padronizar torna o alvo quase estacionário; o quantil final é μ + σ_h · q̂(ε)."""
    nome = "lgbm"

    def __init__(self, cfg: CfgLgbm, quantis: list[float], semente: int):
        self.cfg, self.quantis, self.semente = cfg, quantis, semente

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LGBMQuantis":
        sig = X[SIGMA].to_numpy()
        ok = np.isfinite(sig) & (sig > 0)
        y = np.asarray(y, dtype=float)
        self.mu = float(np.mean(y))
        eps = (y[ok] - self.mu) / sig[ok]
        self.cols = colunas_features(X)
        Xo = X.loc[ok, self.cols]
        self.modelos = []
        for q in self.quantis:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.cfg.quantil, random_state=self.semente, n_jobs=1)
            m.fit(Xo, eps)
            self.modelos.append(m)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("use LGBMDirecao para probabilidades")

    def predict_quantiles(self, X: pd.DataFrame, quantis: list[float] | None = None) -> np.ndarray:
        eps = np.column_stack([m.predict(X[self.cols]) for m in self.modelos])
        eps = np.sort(eps, axis=1)  # elimina cruzamento de quantis
        return self.mu + X[SIGMA].to_numpy()[:, None] * eps
