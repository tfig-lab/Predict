"""Estruturas de resultado da análise."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

SEM_VANTAGEM = "Sem vantagem estatística sobre o acaso neste horizonte"


@dataclass
class PrevisaoHoje:
    p_alta: float                     # calibrada
    p_alta_bruta: float               # ensemble antes da calibração
    p_taxa_base: float                # frequência histórica de alta neste horizonte
    p_supera_indice: float | None
    quantis: dict[float, float]       # log-retorno acumulado (conformal)
    precos: dict[float, float]        # quantis convertidos em preço
    retorno_mediano: float            # retorno simples mediano
    vol_periodo: float                # desvio do log-retorno no horizonte (GARCH)
    vol_anualizada: float
    score: float
    rotulo: str
    vantagem: bool
    componentes: dict[str, float]
    pesos: dict[str, float]
    mensagem: str


@dataclass
class ResultadoHorizonte:
    h: int
    nome: str
    hoje: PrevisaoHoje
    periodo_oos: tuple[str, str]
    n_oos: int
    metricas_direcao: dict[str, dict[str, float]]
    metricas_quantis: dict[str, dict[str, float]]
    metricas_vol: dict[str, Any]
    metricas_relativo: dict[str, float] | None
    testes: dict[str, dict[str, float]]
    acerto_faixas: dict[str, dict[str, float]]
    vantagem_historica: dict[str, float]
    calibracao: list[dict[str, float]]
    backtest: dict[str, Any]
    shap_global: list[tuple[str, float]]
    shap_local: list[tuple[str, float, float]]
    serie_oos: pd.DataFrame = field(repr=False)
    tempo_s: float = 0.0
    # Previsões fora da amostra de CADA modelo (q10, q50, q90 do log-retorno e p_alta), por data de origem
    previsoes: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)


@dataclass
class ResultadoAnalise:
    ticker: str
    mercado: str
    nome_indice: str
    info: dict
    fundamentos: dict
    avisos: list[str]
    data_ultima: pd.Timestamp
    preco_atual: float
    tecnico: dict[str, float]
    horizontes: list[ResultadoHorizonte]
    garch: dict[str, Any]
    hmm: dict[str, Any]
    monte_carlo: dict[str, Any]
    graficos: dict[str, pd.DataFrame | pd.Series] = field(repr=False)
    config_resumo: dict[str, Any] = field(default_factory=dict)
    tempo_total_s: float = 0.0
