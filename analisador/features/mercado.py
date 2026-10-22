"""Features relativas ao índice de referência e ao contexto macro."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _alinhar(serie: pd.Series, indice: pd.DatetimeIndex) -> pd.Series:
    """Último valor conhecido em cada pregão do ativo (nunca um valor posterior)."""
    s = serie[~serie.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(indice)).ffill().reindex(indice)


def features_mercado(px: pd.DataFrame, indice: pd.DataFrame | None) -> pd.DataFrame:
    f = pd.DataFrame(index=px.index)
    if indice is None or indice.empty:
        return f
    c = px["Close"]
    m = _alinhar(indice["Close"], px.index)
    r, rm = np.log(c).diff(), np.log(m).diff()
    f["idx_ret_21"] = np.log(m / m.shift(21))
    f["idx_ret_63"] = np.log(m / m.shift(63))
    f["idx_dist_mm200"] = m / m.rolling(200).mean() - 1
    f["idx_vol_21"] = rm.rolling(21).std() * np.sqrt(252)
    f["beta_126"] = r.rolling(126).cov(rm) / rm.rolling(126).var()
    f["correl_63"] = r.rolling(63).corr(rm)
    f["forca_rel_21"] = np.log(c / c.shift(21)) - f["idx_ret_21"]
    f["forca_rel_63"] = np.log(c / c.shift(63)) - f["idx_ret_63"]
    return f


def features_macro(px: pd.DataFrame, macro: dict[str, pd.Series]) -> pd.DataFrame:
    f = pd.DataFrame(index=px.index)
    for nome, serie in macro.items():
        s = _alinhar(serie, px.index)
        if nome == "BRL=X":
            f["usdbrl_ret_21"] = np.log(s / s.shift(21))
            f["usdbrl_ret_63"] = np.log(s / s.shift(63))
        elif nome == "^VIX":
            f["vix_z_252"] = (s - s.rolling(252).mean()) / s.rolling(252).std()
            f["vix_var_21"] = np.log(s / s.shift(21))
        elif nome == "^TNX":
            f["juros_eua_var_63"] = s - s.shift(63)          # variação em pontos percentuais
        elif nome == "SELIC":
            f["selic_var_126"] = s - s.shift(126)
        elif nome == "IPCA12M":
            f["ipca12m_var_63"] = s - s.shift(63)
    return f
