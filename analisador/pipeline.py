"""Orquestra a análise completa de um ticker."""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analisador.config import Config
from analisador.dados import DadosAtivo, carregar_dados
from analisador.features import construir_features
from analisador.horizonte import avaliar_horizonte
from analisador.log import fixar_sementes, obter_logger
from analisador.modelos import garch, hmm
from analisador.modelos.montecarlo import monte_carlo
from analisador.tipos import ResultadoAnalise

log = obter_logger(__name__)


def preparar_matriz(dados: DadosAtivo, cfg: Config, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Features técnicas/mercado/macro + volatilidade GARCH e regime HMM calculados ONLINE
    (reajustados só com o passado), para poderem entrar como features sem look-ahead."""
    close = dados.px["Close"]
    retornos = np.log(close).diff()
    X = construir_features(dados.px, dados.indice, dados.macro)
    g_on, h_on = Parallel(n_jobs=min(2, n_jobs), backend="loky")([
        delayed(garch.garch_online)(retornos, cfg.garch),
        delayed(hmm.hmm_online)(close, cfg.hmm, cfg.semente),
    ])
    g = g_on.reindex(X.index)
    X["garch_vol_1"] = np.sqrt(g["s2_1"]) / 100 * np.sqrt(252)
    X["garch_vol_razao"] = np.sqrt(g["s2_1"] / g["var_lp"])
    for c in ("s2_1", "persistencia", "var_lp", "egarch"):
        X[f"_{c}"] = g[c]
    hm = h_on.reindex(X.index)
    for c in ("hmm_p_alta", "hmm_p_queda", "hmm_ret_esperado", "hmm_persistencia"):
        X[c] = hm[c]
    X["_hmm_estado"] = hm["hmm_estado"]
    return X, g_on, h_on


def _tecnico(px: pd.DataFrame, X: pd.DataFrame) -> dict[str, float]:
    u = X.iloc[-1]
    return {
        "ret_1m": float(np.expm1(u["ret_21"])), "ret_3m": float(np.expm1(u["ret_63"])),
        "ret_12m": float(np.expm1(u["ret_252"])), "dist_mm200": float(u["dist_mm200"]),
        "rsi": float(u["rsi_14"] * 100), "vol_21": float(u["vol_21"]), "drawdown_52s": float(u["drawdown_52s"]),
    }


def analisar(
    ticker: str,
    cfg: Config,
    inicio: str | None = None,
    offline: bool = False,
    horizontes: list[int] | None = None,
    pasta_cache: str | None = None,
    ao_concluir_horizonte: Callable[[int, int], None] | None = None,
) -> ResultadoAnalise:
    t0 = time.time()
    fixar_sementes(cfg.semente)
    horizontes = horizontes or cfg.horizontes
    n_jobs = (os.cpu_count() or 1) if cfg.n_jobs <= 0 else cfg.n_jobs

    dados = carregar_dados(ticker, cfg, inicio=inicio, offline=offline, pasta_cache=pasta_cache)
    log.info(f"{dados.ticker}: {len(dados.px)} pregões de {dados.px.index[0]:%d/%m/%Y} a {dados.px.index[-1]:%d/%m/%Y}")
    X, g_on, h_on = preparar_matriz(dados, cfg, n_jobs)
    log.info("volatilidade GARCH e regimes HMM online calculados", extra={"s": round(time.time() - t0, 1)})

    close = dados.px["Close"]
    idx_close = None if dados.indice is None else dados.indice["Close"]
    tarefas = (delayed(avaliar_horizonte)(h, X, close, idx_close, dados.caixa_diario, cfg) for h in horizontes)
    resultados = []
    for r in Parallel(n_jobs=min(n_jobs, len(horizontes)), backend="loky", return_as="generator")(tarefas):
        resultados.append(r)
        log.info(f"horizonte {r.nome} validado", extra={"s": round(r.tempo_s, 1), "n_oos": r.n_oos})
        if ao_concluir_horizonte:
            ao_concluir_horizonte(len(resultados), len(horizontes))

    retornos = np.log(close).diff()
    g_final = garch.ajustar_final(retornos, cfg.garch)
    h_final = hmm.ajustar_final(close, cfg.hmm, cfg.semente)
    mc = monte_carlo(g_final, horizontes, cfg.montecarlo["simulacoes"], cfg.montecarlo["bloco"], cfg.semente)

    preco = float(close.iloc[-1])
    datas_futuras = pd.bdate_range(close.index[-1], periods=len(mc.dias) + 1)[1:]
    leque = pd.DataFrame({q: preco * np.exp(v) for q, v in mc.leque.items()}, index=datas_futuras)
    anos = cfg.relatorio["anos_grafico"]
    janela = close.index >= close.index[-1] - pd.DateOffset(years=anos)
    p = g_final.params
    graficos = {
        "preco": pd.DataFrame({"close": close, "mm50": close.rolling(50).mean(), "mm200": close.rolling(200).mean()})[janela],
        "regime": h_on[["hmm_estado", "hmm_k"]][janela],
        "vol": pd.DataFrame({
            "garch": np.sqrt(g_on["s2_1"]) / 100 * np.sqrt(252),
            "realizada_21": retornos.rolling(21).std() * np.sqrt(252),
        })[janela],
        "vol_futura": pd.Series(
            [np.sqrt((p.var_longo_prazo + p.persistencia ** k * (g_final.s2_1 - p.var_longo_prazo))) / 100 * np.sqrt(252)
             for k in range(len(mc.dias))], index=datas_futuras),
        "leque": leque,
    }

    res = ResultadoAnalise(
        ticker=dados.ticker, mercado=dados.mercado, nome_indice=dados.nome_indice, info=dados.info,
        fundamentos=dados.fundamentos, avisos=dados.avisos, data_ultima=close.index[-1], preco_atual=preco,
        tecnico=_tecnico(dados.px, X), horizontes=sorted(resultados, key=lambda r: r.h),
        garch={"tipo": p.tipo, "bics": g_final.bics, "nu": p.nu, "persistencia": p.persistencia,
               "vol_atual": float(np.sqrt(g_final.s2_1) / 100 * np.sqrt(252)),
               "vol_longo_prazo": float(np.sqrt(p.var_longo_prazo) / 100 * np.sqrt(252))},
        hmm={"nomes": h_final.nomes, "probs": h_final.probs_atuais.tolist(), "transicao": h_final.transicao.tolist(),
             "medias": h_final.medias.tolist(), "atual": h_final.nomes[int(np.argmax(h_final.probs_atuais))]},
        monte_carlo={"eventos": mc.eventos, "n": mc.n_trajetorias},
        graficos=graficos,
        config_resumo={"passo": cfg.passo, "rapido": cfg.rapido, "horizontes": horizontes},
    )
    res.tempo_total_s = time.time() - t0
    log.info("análise concluída", extra={"ticker": dados.ticker, "s": round(res.tempo_total_s, 1)})
    return res
