"""Monta a matriz de features e os alvos por horizonte."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analisador.features.mercado import features_macro, features_mercado
from analisador.features.tecnicas import features_tecnicas

NOMES = {
    "ret_1": "Retorno de 1 dia", "ret_5": "Retorno de 1 semana", "ret_21": "Retorno de 1 mês",
    "ret_63": "Retorno de 3 meses", "ret_126": "Retorno de 6 meses", "ret_252": "Retorno de 12 meses",
    "mom_12_1": "Momentum 12-1 meses",
    "vol_5": "Volatilidade de 1 semana", "vol_21": "Volatilidade de 1 mês", "vol_63": "Volatilidade de 3 meses",
    "vol_parkinson_21": "Volatilidade Parkinson (máx/mín)", "vol_gk_21": "Volatilidade Garman-Klass",
    "vol_razao_5_63": "Vol. 1 semana / 3 meses", "vol_razao_21_252": "Vol. 1 mês / 12 meses",
    "dist_mm20": "Distância da média de 20 dias", "dist_mm50": "Distância da média de 50 dias",
    "dist_mm200": "Distância da média de 200 dias", "mm50_mm200": "Média 50 vs. média 200",
    "rsi_14": "RSI (14)", "macd_hist": "MACD (histograma)", "bollinger_pct": "Posição nas Bandas de Bollinger",
    "drawdown_52s": "Queda desde a máxima de 52 semanas", "dist_minimo_52s": "Alta desde a mínima de 52 semanas",
    "volume_z": "Volume (z-score)", "mes": "Mês do ano", "dia_semana": "Dia da semana",
    "idx_ret_21": "Índice: retorno de 1 mês", "idx_ret_63": "Índice: retorno de 3 meses",
    "idx_dist_mm200": "Índice vs. média de 200 dias", "idx_vol_21": "Índice: volatilidade de 1 mês",
    "beta_126": "Beta vs. índice", "correl_63": "Correlação com o índice",
    "forca_rel_21": "Força relativa (1 mês)", "forca_rel_63": "Força relativa (3 meses)",
    "usdbrl_ret_21": "Dólar: variação de 1 mês", "usdbrl_ret_63": "Dólar: variação de 3 meses",
    "vix_z_252": "VIX (z-score)", "vix_var_21": "VIX: variação de 1 mês",
    "juros_eua_var_63": "Juros EUA 10a: variação de 3 meses",
    "selic_var_126": "Selic: variação de 6 meses", "ipca12m_var_63": "IPCA 12m: variação de 3 meses",
    "garch_vol_1": "Volatilidade prevista (GARCH)", "garch_vol_h": "Volatilidade prevista no horizonte (GARCH)",
    "garch_vol_razao": "Vol. GARCH vs. de longo prazo",
    "hmm_p_alta": "Prob. regime de alta (HMM)", "hmm_p_queda": "Prob. regime de queda (HMM)",
    "hmm_ret_esperado": "Retorno esperado do regime (HMM)", "hmm_persistencia": "Persistência do regime (HMM)",
}

AQUECIMENTO = 252  # pregões necessários para as janelas mais longas


def construir_features(px: pd.DataFrame, indice: pd.DataFrame | None, macro: dict[str, pd.Series]) -> pd.DataFrame:
    """Features na linha t usam somente informação disponível no fechamento de t."""
    f = pd.concat([features_tecnicas(px), features_mercado(px, indice), features_macro(px, macro)], axis=1)
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.iloc[AQUECIMENTO:]
    return f.loc[:, f.notna().mean() > 0.6]


def alvo_log_retorno(close: pd.Series, h: int) -> pd.Series:
    """y_t = log(P_{t+h} / P_t). NaN nas últimas h linhas (ainda não realizado)."""
    return np.log(close.shift(-h) / close)


def alvo_relativo(close: pd.Series, indice_close: pd.Series | None, h: int) -> pd.Series | None:
    if indice_close is None:
        return None
    m = indice_close.reindex(close.index).ffill()
    return alvo_log_retorno(close, h) - alvo_log_retorno(m, h)
