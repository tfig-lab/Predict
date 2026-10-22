"""Séries do SGS do Banco Central (Selic meta = 432, IPCA mensal = 433)."""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
SELIC_META = 432
IPCA_MENSAL = 433
INICIO_PADRAO = date(1995, 1, 1)


def _get_com_retentativas(url: str, params: dict, tentativas: int = 4) -> requests.Response:
    """A API do SGS oscila bastante; tenta de novo com espera crescente."""
    for i in range(tentativas):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code < 500:
                return resp
        except requests.RequestException:
            if i == tentativas - 1:
                raise
        time.sleep(2 ** i)
    resp.raise_for_status()
    return resp


def baixar_sgs(codigo: int, inicio: date | None = None, fim: date | None = None) -> pd.DataFrame:
    """Baixa em blocos de 10 anos (limite da API para séries diárias). Devolve coluna 'Close'."""
    inicio = inicio or INICIO_PADRAO
    fim = fim or date.today()
    partes = []
    atual = inicio
    while atual <= fim:
        bloco_fim = min(date(atual.year + 9, 12, 31), fim)
        resp = _get_com_retentativas(
            URL.format(codigo=codigo),
            {"formato": "json", "dataInicial": atual.strftime("%d/%m/%Y"), "dataFinal": bloco_fim.strftime("%d/%m/%Y")},
        )
        if resp.status_code == 404:  # bloco sem observações
            atual = bloco_fim + timedelta(days=1)
            continue
        resp.raise_for_status()
        dados = resp.json()
        if dados:
            df = pd.DataFrame(dados)
            df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
            df["Close"] = pd.to_numeric(df["valor"], errors="coerce")
            partes.append(df.set_index("data")[["Close"]])
        atual = bloco_fim + timedelta(days=1)
    if not partes:
        raise ValueError(f"SGS {codigo} sem dados")
    out = pd.concat(partes).sort_index()
    return out[~out.index.duplicated(keep="last")]


def ipca_12m_disponivel(ipca_mensal: pd.Series, defasagem_dias: int) -> pd.Series:
    """IPCA acumulado em 12 meses indexado pela data em que ele passa a ser CONHECIDO.

    O dado do mês M (datado no dia 1º de M) só é divulgado por volta do dia 10 de M+1;
    indexar pela data de referência criaria look-ahead.
    """
    acum = (1 + ipca_mensal / 100).rolling(12).apply(lambda x: x.prod(), raw=True) - 1
    acum.index = acum.index + pd.Timedelta(days=defasagem_dias)
    return acum.dropna()
