"""Indicadores técnicos do próprio ativo (somente dados até t)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANUAL = np.sqrt(252)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ganho / perda.replace(0, np.nan))


def vol_parkinson(px: pd.DataFrame, n: int) -> pd.Series:
    hl = np.log(px["High"] / px["Low"]) ** 2
    return np.sqrt(hl.rolling(n).mean() / (4 * np.log(2))) * ANUAL


def vol_garman_klass(px: pd.DataFrame, n: int) -> pd.Series:
    hl = 0.5 * np.log(px["High"] / px["Low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(px["Close"] / px["Open"]) ** 2
    return np.sqrt((hl - co).clip(lower=0).rolling(n).mean()) * ANUAL


def features_tecnicas(px: pd.DataFrame) -> pd.DataFrame:
    c = px["Close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=px.index)

    for n in (1, 5, 21, 63, 126, 252):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    f["mom_12_1"] = np.log(c.shift(21) / c.shift(252))       # momentum clássico, sem o último mês

    for n in (5, 21, 63):
        f[f"vol_{n}"] = r.rolling(n).std() * ANUAL
    f["vol_parkinson_21"] = vol_parkinson(px, 21)
    f["vol_gk_21"] = vol_garman_klass(px, 21)
    f["vol_razao_5_63"] = f["vol_5"] / f["vol_63"]
    f["vol_razao_21_252"] = f["vol_21"] / (r.rolling(252).std() * ANUAL)

    medias = {n: c.rolling(n).mean() for n in (20, 50, 200)}
    for n, m in medias.items():
        f[f"dist_mm{n}"] = c / m - 1
    f["mm50_mm200"] = medias[50] / medias[200] - 1

    f["rsi_14"] = rsi(c) / 100
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    dp20 = c.rolling(20).std()
    f["bollinger_pct"] = (c - (medias[20] - 2 * dp20)) / (4 * dp20)

    f["drawdown_52s"] = c / c.rolling(252).max() - 1
    f["dist_minimo_52s"] = c / c.rolling(252).min() - 1

    vol = np.log(px["Volume"].where(px["Volume"] > 0))
    f["volume_z"] = (vol.rolling(5).mean() - vol.rolling(126).mean()) / vol.rolling(126).std()

    f["mes"] = px.index.month
    f["dia_semana"] = px.index.dayofweek
    return f
