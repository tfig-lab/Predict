"""Regimes de mercado com Hidden Markov Model gaussiano (hmmlearn).

Ponto crítico: `GaussianHMM.predict_proba` devolve probabilidades SUAVIZADAS (usam a série
inteira, inclusive o futuro). Para features e backtest usamos probabilidades FILTRADAS,
calculadas aqui pelo algoritmo forward com os parâmetros ajustados só no passado.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from analisador.config import CfgHmm

NOMES_REGIMES = {2: ["Queda", "Alta"], 3: ["Queda", "Lateral", "Alta"]}
logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # avisos de convergência de EM são esperados


def observacoes(close: pd.Series) -> pd.DataFrame:
    """Retorno de 5 dias e log da volatilidade realizada de 21 dias."""
    r = np.log(close).diff()
    return pd.DataFrame({"r5": np.log(close / close.shift(5)), "lvol": np.log(r.rolling(21).std() + 1e-8)}).dropna()


@dataclass
class AjusteHmm:
    modelo: GaussianHMM
    ordem: np.ndarray          # estados reordenados do menor para o maior retorno médio
    media: np.ndarray          # padronização (do treino)
    desvio: np.ndarray

    @property
    def k(self) -> int:
        return self.modelo.n_components

    def _padronizar(self, X: np.ndarray) -> np.ndarray:
        return (X - self.media) / self.desvio

    @property
    def transicao(self) -> np.ndarray:
        A = self.modelo.transmat_
        return A[np.ix_(self.ordem, self.ordem)]

    @property
    def medias_originais(self) -> np.ndarray:
        """Médias por estado (ordenado) na escala original: colunas r5, lvol."""
        return self.modelo.means_[self.ordem] * self.desvio + self.media

    def filtrar(self, X: np.ndarray) -> np.ndarray:
        """P(estado_t | x_1..x_t), estados na ordem crescente de retorno. Forma (n, k)."""
        Z = self._padronizar(X)
        m = self.modelo
        var = np.diagonal(m.covars_, axis1=1, axis2=2)  # covars_ sempre devolve matrizes (k, d, d)
        logp = -0.5 * (((Z[:, None, :] - m.means_[None]) ** 2) / var[None] + np.log(2 * np.pi * var)[None]).sum(axis=2)
        # Verossimilhanças reescaladas por linha (estável); a normalização do filtro cancela a escala
        b = np.exp(logp - logp.max(axis=1, keepdims=True)) + 1e-300
        A = m.transmat_
        alfa = np.empty_like(b)
        atual = m.startprob_ * b[0]
        alfa[0] = atual / atual.sum()
        for t in range(1, len(b)):
            atual = (alfa[t - 1] @ A) * b[t]
            alfa[t] = atual / atual.sum()
        return alfa[:, self.ordem]


def reajustar(X: np.ndarray, anterior: AjusteHmm, cfg: CfgHmm) -> AjusteHmm:
    """Reajuste rápido com o mesmo nº de estados, partindo dos parâmetros anteriores (warm start)."""
    media, desvio = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = (X - media) / desvio
    a = anterior.modelo
    m = GaussianHMM(n_components=a.n_components, covariance_type="diag", n_iter=cfg.iteracoes,
                    init_params="", tol=1e-4, random_state=0)
    m.startprob_, m.transmat_ = a.startprob_.copy(), a.transmat_.copy()
    m.means_ = (a.means_ * anterior.desvio + anterior.media - media) / desvio
    m.covars_ = np.diagonal(a.covars_, axis1=1, axis2=2) * (anterior.desvio / desvio) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m.fit(Z)
        except Exception:
            return anterior
    return AjusteHmm(m, np.argsort(m.means_[:, 0]), media, desvio)


def ajustar(X: np.ndarray, cfg: CfgHmm, semente: int) -> AjusteHmm:
    """Escolhe o nº de estados por BIC; várias inicializações por nº de estados."""
    media, desvio = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = (X - media) / desvio
    melhor, melhor_bic = None, np.inf
    for k in cfg.estados:
        for i in range(cfg.inicializacoes):
            m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=cfg.iteracoes,
                            random_state=semente + 97 * i + k, tol=1e-4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m.fit(Z)
                    bic = m.bic(Z)
                except Exception:
                    continue
            if np.isfinite(bic) and bic < melhor_bic:
                melhor, melhor_bic = m, bic
    if melhor is None:
        raise RuntimeError("HMM não convergiu")
    ordem = np.argsort(melhor.means_[:, 0])
    return AjusteHmm(melhor, ordem, media, desvio)


def resumo_regime(probs: np.ndarray, ajuste: AjusteHmm) -> dict[str, np.ndarray]:
    A = ajuste.transicao
    r5 = ajuste.medias_originais[:, 0]
    return {
        "hmm_p_alta": probs[:, -1],
        "hmm_p_queda": probs[:, 0],
        "hmm_ret_esperado": probs @ r5,
        "hmm_persistencia": probs @ np.diag(A),
        "hmm_estado": np.argmax(probs, axis=1).astype(float),
        "hmm_k": np.full(len(probs), float(ajuste.k)),
    }


def hmm_online(close: pd.Series, cfg: CfgHmm, semente: int) -> pd.DataFrame:
    """Regime filtrado em cada pregão, com o HMM reajustado a cada `reajuste` pregões no passado."""
    obs = observacoes(close)
    X = obs.to_numpy()
    n = len(X)
    colunas = ["hmm_p_alta", "hmm_p_queda", "hmm_ret_esperado", "hmm_persistencia", "hmm_estado", "hmm_k"]
    out = {c: np.full(n, np.nan) for c in colunas}
    ajuste: AjusteHmm | None = None
    ultima_selecao = -10**9
    for k in range(cfg.janela_minima, n, cfg.reajuste):
        if ajuste is None or k - ultima_selecao >= cfg.reespecificacao:
            ajuste, ultima_selecao = ajustar(X[:k], cfg, semente), k
        else:
            ajuste = reajustar(X[:k], ajuste, cfg)
        fim = min(k + cfg.reajuste, n)
        resumo = resumo_regime(ajuste.filtrar(X[:fim]), ajuste)
        for c in colunas:
            out[c][k:fim] = resumo[c][k:fim]
    return pd.DataFrame(out, index=obs.index).reindex(close.index)


@dataclass
class HmmFinal:
    ajuste: AjusteHmm
    nomes: list[str]
    probs_atuais: np.ndarray
    transicao: np.ndarray
    medias: np.ndarray            # r5 e vol anualizada por estado


def ajustar_final(close: pd.Series, cfg: CfgHmm, semente: int) -> HmmFinal:
    obs = observacoes(close).to_numpy()
    ajuste = ajustar(obs, cfg, semente)
    probs = ajuste.filtrar(obs)[-1]
    med = ajuste.medias_originais
    medias = np.column_stack([med[:, 0], np.exp(med[:, 1]) * np.sqrt(252)])
    return HmmFinal(ajuste, NOMES_REGIMES.get(ajuste.k, [f"Estado {i}" for i in range(ajuste.k)]),
                    probs, ajuste.transicao, medias)
