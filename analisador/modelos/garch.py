"""Família GARCH (arch): seleção por BIC, volatilidade online sem look-ahead e simulação.

Convenção: os retornos entram no `arch` em PORCENTAGEM (escala numericamente estável);
as saídas públicas estão em log-retorno decimal.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.special import gammaln

from analisador.config import CfgGarch
from analisador.log import obter_logger

log = obter_logger(__name__)
RAIZ_2_PI = np.sqrt(2 / np.pi)


def media_abs_t(nu: float) -> float:
    """E|z| para z ~ t de Student PADRONIZADA (variância 1) com ν graus de liberdade."""
    if not np.isfinite(nu) or nu > 300:
        return RAIZ_2_PI
    nu = max(nu, 2.01)
    return float(np.sqrt(nu - 2) * np.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi))


@dataclass(frozen=True)
class ParamsGarch:
    tipo: str          # GARCH | GJR | EGARCH
    mu: float          # % ao dia
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float          # graus de liberdade da t
    bic: float = float("nan")

    @property
    def persistencia(self) -> float:
        if self.tipo == "EGARCH":
            return min(self.beta, 0.9995)
        return min(self.alpha + self.gamma / 2 + self.beta, 0.9995)

    @property
    def var_longo_prazo(self) -> float:
        """Variância diária incondicional (%²). No EGARCH, aproximação exp(E[ln σ²])."""
        if self.tipo == "EGARCH":
            # E[ln σ²] = (ω + α·(E|z| - √(2/π))) / (1 - β). O `arch` centra |z| em √(2/π) (valor
            # da normal); com inovações t, E|z| é menor e esse termo NÃO é desprezível quando β ≈ 1.
            # Limita a log-variância a um intervalo plausível.
            centro = self.omega + self.alpha * (media_abs_t(self.nu) - RAIZ_2_PI)
            return float(np.exp(np.clip(centro / (1 - self.persistencia), -10, 10)))
        return self.omega / (1 - self.persistencia)


def _modelo(r_pct: np.ndarray, tipo: str, dist: str):
    o = 0 if tipo == "GARCH" else 1
    vol = "EGARCH" if tipo == "EGARCH" else "GARCH"
    return arch_model(r_pct, mean="Constant", vol=vol, p=1, o=o, q=1, dist=dist, rescale=False)


def ajustar(r_pct: np.ndarray, tipo: str, dist: str = "t") -> ParamsGarch:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _modelo(r_pct, tipo, dist).fit(disp="off", show_warning=False)
    p = res.params
    return ParamsGarch(
        tipo=tipo, mu=float(p["mu"]), omega=float(p["omega"]), alpha=float(p["alpha[1]"]),
        gamma=float(p.get("gamma[1]", 0.0)), beta=float(p["beta[1]"]), nu=float(p.get("nu", 1e6)), bic=float(res.bic),
    )


def selecionar(r_pct: np.ndarray, variantes: list[str], dist: str = "t") -> tuple[ParamsGarch, dict[str, float]]:
    """Ajusta cada variante e escolhe a de menor BIC."""
    ajustes = {}
    for tipo in variantes:
        try:
            ajustes[tipo] = ajustar(r_pct, tipo, dist)
        except Exception as e:  # convergência ruim numa variante não deve derrubar a análise
            log.debug("variante GARCH falhou", extra={"tipo": tipo, "erro": str(e)})
    if not ajustes:
        raise RuntimeError("nenhuma variante GARCH convergiu")
    melhor = min(ajustes.values(), key=lambda a: a.bic)
    return melhor, {k: v.bic for k, v in ajustes.items()}


def variancia_condicional(r_pct: np.ndarray, p: ParamsGarch, s2_inicial: float | None = None) -> np.ndarray:
    """s2[t] = variância prevista para o dia t+1 com informação até t (inclusive)."""
    n = len(r_pct)
    s2 = np.empty(n)
    atual = float(np.var(r_pct[: min(n, 250)])) if s2_inicial is None else s2_inicial
    for t in range(n):
        e = r_pct[t] - p.mu
        if p.tipo == "EGARCH":
            z = e / np.sqrt(atual)
            ln = p.omega + p.alpha * (abs(z) - RAIZ_2_PI) + p.gamma * z + p.beta * np.log(atual)
            atual = float(np.exp(np.clip(ln, -20, 20)))
        else:
            assim = p.gamma if (p.tipo == "GJR" and e < 0) else 0.0
            atual = p.omega + (p.alpha + assim) * e * e + p.beta * atual
        s2[t] = atual
    return s2


def garch_online(retornos: pd.Series, cfg: CfgGarch) -> pd.DataFrame:
    """Previsões de variância sem look-ahead: parâmetros reestimados a cada `reajuste` pregões
    usando só o passado; a especificação (GARCH/GJR/EGARCH) é reescolhida a cada `reespecificacao`."""
    r_pct = 100 * retornos.fillna(0.0).to_numpy()
    n = len(r_pct)
    cols = {c: np.full(n, np.nan) for c in ("s2_1", "persistencia", "var_lp", "egarch", "mu")}
    params: ParamsGarch | None = None
    ultima_espec = -10**9
    for k in range(cfg.janela_minima, n, cfg.reajuste):
        treino = r_pct[:k]
        if params is None or k - ultima_espec >= cfg.reespecificacao:
            params, _ = selecionar(treino, cfg.variantes, cfg.distribuicao)
            ultima_espec = k
        else:
            try:
                params = ajustar(treino, params.tipo, cfg.distribuicao)
            except Exception:
                pass  # mantém os parâmetros anteriores
        fim = min(k + cfg.reajuste, n)
        s2 = variancia_condicional(r_pct[:fim], params)
        cols["s2_1"][k:fim] = s2[k:fim]
        cols["persistencia"][k:fim] = params.persistencia
        cols["var_lp"][k:fim] = params.var_longo_prazo
        cols["egarch"][k:fim] = float(params.tipo == "EGARCH")
        cols["mu"][k:fim] = params.mu
    return pd.DataFrame(cols, index=retornos.index)


def sigma_horizonte(online: pd.DataFrame, h: int) -> pd.Series:
    """Desvio do log-retorno acumulado em h pregões (decimal), pela estrutura a termo do GARCH:
    sum_{k=1..h} E[σ²_{t+k}], com reversão geométrica à variância de longo prazo."""
    s2, phi, lp, eg = (online[c].to_numpy() for c in ("s2_1", "persistencia", "var_lp", "egarch"))
    with np.errstate(invalid="ignore", divide="ignore"):
        linear = h * lp + (s2 - lp) * (1 - phi**h) / (1 - phi)
        # EGARCH: reversão em log-variância
        ln_lp, ln_s2 = np.log(lp), np.log(s2)
        k = np.arange(h)[None, :]
        log_caminho = ln_lp[:, None] + (phi[:, None] ** k) * (ln_s2 - ln_lp)[:, None]
        exponencial = np.exp(log_caminho).sum(axis=1)
    total = np.where(eg == 1, exponencial, linear)
    return pd.Series(np.sqrt(np.maximum(total, 0)) / 100, index=online.index)


def simular(
    p: ParamsGarch,
    s2_1: float,
    horizonte: int,
    n_sim: int,
    rng: np.random.Generator,
    residuos: np.ndarray | None = None,
    bloco: int = 21,
) -> np.ndarray:
    """Trajetórias de log-retorno ACUMULADO (decimal), forma (n_sim, horizonte).

    residuos=None: inovações t de Student padronizadas (GARCH-t paramétrico).
    residuos=array: bootstrap em blocos dos resíduos padronizados históricos (FHS),
    preservando caudas e dependência de curto prazo observadas.
    """
    if residuos is None:
        nu = max(p.nu, 2.1)
        z = rng.standard_t(nu, size=(n_sim, horizonte)) * np.sqrt((nu - 2) / nu)
    else:
        m = len(residuos)
        nb = int(np.ceil(horizonte / bloco))
        inicios = rng.integers(0, m - bloco, size=(n_sim, nb))
        idx = (inicios[:, :, None] + np.arange(bloco)[None, None, :]).reshape(n_sim, -1)[:, :horizonte]
        z = residuos[idx]
    s2 = np.full(n_sim, s2_1)
    r = np.empty((n_sim, horizonte))
    for k in range(horizonte):
        e = np.sqrt(s2) * z[:, k]
        r[:, k] = p.mu + e
        if p.tipo == "EGARCH":
            ln = p.omega + p.alpha * (np.abs(z[:, k]) - RAIZ_2_PI) + p.gamma * z[:, k] + p.beta * np.log(s2)
            s2 = np.exp(np.clip(ln, -20, 20))
        else:
            assim = np.where(e < 0, p.gamma, 0.0) if p.tipo == "GJR" else 0.0
            s2 = p.omega + (p.alpha + assim) * e * e + p.beta * s2
    return np.cumsum(r, axis=1) / 100


@dataclass
class GarchFinal:
    params: ParamsGarch
    bics: dict[str, float]
    s2_1: float                    # variância (%²) prevista para o próximo pregão
    residuos: np.ndarray           # resíduos padronizados in-sample (para o FHS)


def ajustar_final(retornos: pd.Series, cfg: CfgGarch) -> GarchFinal:
    r_pct = 100 * retornos.dropna().to_numpy()
    params, bics = selecionar(r_pct, cfg.variantes, cfg.distribuicao)
    s2 = variancia_condicional(r_pct, params)
    s2_atual = np.concatenate([[np.var(r_pct[:250])], s2[:-1]])  # variância de cada dia t, conhecida em t-1
    residuos = (r_pct - params.mu) / np.sqrt(s2_atual)
    return GarchFinal(params=params, bics=bics, s2_1=float(s2[-1]), residuos=residuos[50:])
