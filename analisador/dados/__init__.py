"""Carregamento de todos os dados de um ativo: preço, índice, macro, BCB, fundamentos."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd

from analisador.config import RAIZ, Config
from analisador.dados.bcb import IPCA_MENSAL, SELIC_META, baixar_sgs, ipca_12m_disponivel
from analisador.dados.cache import CacheParquet, obter_com_cache
from analisador.dados.validacao import validar_ohlcv
from analisador.dados.yahoo import baixar_yahoo, info_e_fundamentos, mercado_do_ticker, normalizar_ticker
from analisador.log import obter_logger

log = obter_logger(__name__)


@dataclass
class DadosAtivo:
    ticker: str
    mercado: str
    px: pd.DataFrame                       # OHLCV do ativo
    indice: pd.DataFrame | None            # OHLCV do índice de referência
    nome_indice: str
    macro: dict[str, pd.Series]            # Close de cada série macro (câmbio, VIX, juros, Selic, IPCA 12m)
    caixa_diario: pd.Series                # rendimento diário do caixa (backtest)
    info: dict
    fundamentos: dict
    avisos: list[str] = field(default_factory=list)


def _sgs(codigo: int, inicio: date | None) -> pd.DataFrame:
    return baixar_sgs(codigo, inicio)


def carregar_dados(
    ticker_digitado: str,
    cfg: Config,
    inicio: str | None = None,
    offline: bool = False,
    pasta_cache: str | Path | None = None,
) -> DadosAtivo:
    ticker = normalizar_ticker(ticker_digitado)
    mercado = mercado_do_ticker(ticker)
    mcfg = cfg.dados.mercados[mercado]
    cache = CacheParquet(pasta_cache or RAIZ / cfg.dados.pasta_cache)
    tol = cfg.dados.tolerancia_reajuste
    avisos: list[str] = []

    def yahoo(t: str) -> pd.DataFrame:
        return obter_com_cache(cache, t, partial(baixar_yahoo, t), tol, offline=offline)

    px, av = validar_ohlcv(yahoo(ticker), ticker, cfg.dados.salto_sigma, cfg.dados.lacuna_dias_corridos)
    avisos += av

    indice = None
    try:
        indice, _ = validar_ohlcv(yahoo(mcfg.indice), mcfg.indice, cfg.dados.salto_sigma,
                                  cfg.dados.lacuna_dias_corridos, checar_volume=False)
    except Exception as e:
        avisos.append(f"índice {mcfg.indice} indisponível ({e}); features relativas desativadas")

    macro: dict[str, pd.Series] = {}
    for t in mcfg.macro:
        try:
            df, _ = validar_ohlcv(yahoo(t), t, cfg.dados.salto_sigma, 30, checar_volume=False)
            macro[t] = df["Close"]
        except Exception as e:
            avisos.append(f"série macro {t} indisponível ({e})")

    if mcfg.bcb:
        try:
            selic = obter_com_cache(cache, "SGS_432", partial(_sgs, SELIC_META), 1e-9, offline=offline)["Close"]
            macro["SELIC"] = selic
        except Exception as e:
            avisos.append(f"Selic (SGS 432) indisponível ({e})")
        try:
            ipca = obter_com_cache(cache, "SGS_433", partial(_sgs, IPCA_MENSAL), 1e-9, offline=offline)["Close"]
            macro["IPCA12M"] = ipca_12m_disponivel(ipca, cfg.dados.defasagem_ipca_dias)
        except Exception as e:
            avisos.append(f"IPCA (SGS 433) indisponível ({e})")

    caixa = pd.Series(0.0, index=px.index)
    try:
        if mcfg.caixa == "selic" and "SELIC" in macro:
            taxa = macro["SELIC"].reindex(px.index, method="ffill").fillna(0) / 100
            caixa = (1 + taxa) ** (1 / 252) - 1
        elif mcfg.caixa.startswith("^"):
            taxa = yahoo(mcfg.caixa)["Close"].reindex(px.index, method="ffill").fillna(0) / 100
            caixa = taxa / 252
    except Exception as e:
        avisos.append(f"taxa do caixa indisponível ({e}); backtest assume caixa a 0%")

    inicio = inicio or cfg.dados.inicio
    if inicio:
        px = px[px.index >= pd.Timestamp(inicio)]
    else:
        px = px[px.index >= px.index[-1] - pd.DateOffset(years=cfg.dados.anos_maximos)]
    anos = (px.index[-1] - px.index[0]).days / 365.25
    if anos < cfg.dados.anos_minimos:
        avisos.append(f"apenas {anos:.1f} anos de histórico (recomendado: {cfg.dados.anos_minimos}+); validação menos confiável")
    if len(px) < cfg.dados.pregoes_minimos:
        raise ValueError(f"{ticker}: {len(px)} pregões; mínimo exigido é {cfg.dados.pregoes_minimos}")

    info, fundamentos = ({"nome": ticker, "setor": "", "moeda": "BRL" if mercado == "B3" else "USD"}, {}) \
        if offline else info_e_fundamentos(ticker)

    for a in avisos:
        log.warning(a, extra={"ticker": ticker})
    return DadosAtivo(
        ticker=ticker, mercado=mercado, px=px, indice=indice,
        nome_indice="Ibovespa" if mercado == "B3" else "S&P 500",
        macro=macro, caixa_diario=caixa.reindex(px.index).fillna(0.0),
        info=info, fundamentos=fundamentos, avisos=avisos,
    )
