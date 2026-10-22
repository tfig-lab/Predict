"""Checagens de qualidade das séries de preço."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analisador.log import obter_logger

log = obter_logger(__name__)


def _cortar_preco_congelado(df: pd.DataFrame, minimo: int) -> tuple[pd.DataFrame, list[str]]:
    """Trechos longos com preço idêntico e volume nulo não são negociação real (ex.: ticker
    antigo preenchido pela fonte). Eles criam retornos zero artificiais, que um modelo aprende
    a "prever" e que inflam qualquer métrica. O histórico passa a começar após o último trecho."""
    parado = (df["Close"].diff() == 0) & (df["Volume"].fillna(0) <= 0)
    grupo = (~parado).cumsum()
    tamanhos = parado.groupby(grupo).transform("sum").where(parado, 0)
    longos = tamanhos >= minimo
    if not longos.any():
        return df, []
    fim = longos[longos].index[-1]
    n_parados = int(parado[: fim].sum())
    restante = df[df.index > fim]
    aviso = (f"preço congelado com volume nulo em {n_parados} pregões até {fim:%d/%m/%Y} "
             f"(dado artificial da fonte); histórico considerado a partir de "
             f"{restante.index[0]:%d/%m/%Y}" if len(restante) else "")
    return restante, [aviso] if aviso else ["série inteira com preço congelado"]


def validar_ohlcv(
    df: pd.DataFrame,
    nome: str,
    salto_sigma: float = 8.0,
    lacuna_dias: int = 7,
    checar_volume: bool = True,
    congelado_min: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove linhas inválidas e devolve (df_limpo, avisos). Os avisos vão para o log em DEBUG."""
    avisos: list[str] = []
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])

    precos = [c for c in ("Open", "High", "Low", "Close") if c in df]
    ruins = (df[precos] <= 0).any(axis=1)
    if ruins.any():
        avisos.append(f"{int(ruins.sum())} pregões com preço zerado ou negativo removidos")
        df = df[~ruins]

    if {"High", "Low"} <= set(df):
        inconsistente = (df["High"] < df["Low"]) | (df["Close"] > df["High"] * 1.01) | (df["Close"] < df["Low"] * 0.99)
        if inconsistente.any():
            avisos.append(f"{int(inconsistente.sum())} pregões com máxima/mínima incoerentes (ajustados)")
            df = df.copy()
            df.loc[:, "High"] = df[["High", "Close", "Open", "Low"]].max(axis=1)
            df.loc[:, "Low"] = df[["Low", "Close", "Open", "High"]].min(axis=1)

    lacunas = df.index.to_series().diff().dt.days
    grandes = lacunas[lacunas > lacuna_dias]
    if len(grandes):
        avisos.append(f"{len(grandes)} lacunas de mais de {lacuna_dias} dias sem pregão "
                      f"(maior: {int(grandes.max())} dias, em {grandes.idxmax():%d/%m/%Y})")

    r = np.log(df["Close"]).diff()
    sigma = 1.4826 * (r - r.median()).abs().median()
    if sigma > 0:
        extremo = r.abs() > salto_sigma * sigma
        # Pico isolado: salto seguido de reversão quase total no pregão seguinte = erro de dado
        prox = r.shift(-1)
        pico = extremo & (prox.abs() > salto_sigma * sigma) & (np.sign(r) != np.sign(prox)) & ((r + prox).abs() < 0.25 * r.abs())
        if pico.any():
            avisos.append(f"{int(pico.sum())} picos isolados (salto e reversão no dia seguinte) removidos como erro de dado")
            df = df[~pico]
        restantes = int((extremo & ~pico).sum())
        if restantes:
            avisos.append(f"{restantes} retornos diários acima de {salto_sigma:g} desvios robustos (mantidos; confira eventos)")

    if checar_volume and "Volume" in df:
        df, av = _cortar_preco_congelado(df, congelado_min)
        avisos += av
        nulo = (df["Volume"].fillna(0) <= 0).mean()
        if nulo > 0.05:
            avisos.append(f"volume nulo em {nulo:.0%} dos pregões")

    for a in avisos:
        log.debug(a, extra={"serie": nome})  # quem chama decide o que vira aviso visível
    return df, avisos
