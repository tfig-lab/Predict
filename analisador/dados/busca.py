"""Busca de ações por nome ou código (Yahoo Finance), restrita à B3 e às bolsas dos EUA."""
from __future__ import annotations

import re

import yfinance as yf

# Bolsas americanas principais no Yahoo (balcão/OTC fica de fora: dados ruins e pouca liquidez)
BOLSAS_EUA = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NAS", "NYS"}
TIPOS = {"EQUITY", "ETF"}
B3 = re.compile(r"^[A-Z0-9]{4}\d{1,2}\.SA$")         # PETR4.SA, BOVA11.SA (exclui fracionário PETR4F.SA)

POPULARES = [
    ("PETR4.SA", "Petrobras PN"), ("VALE3.SA", "Vale ON"), ("ITUB4.SA", "Itaú Unibanco PN"),
    ("BBAS3.SA", "Banco do Brasil ON"), ("BBDC4.SA", "Bradesco PN"), ("B3SA3.SA", "B3 ON"),
    ("WEGE3.SA", "WEG ON"), ("ABEV3.SA", "Ambev ON"), ("ELET3.SA", "Eletrobras ON"),
    ("PRIO3.SA", "PetroRio ON"), ("SUZB3.SA", "Suzano ON"), ("RENT3.SA", "Localiza ON"),
    ("MGLU3.SA", "Magazine Luiza ON"), ("BOVA11.SA", "ETF Ibovespa"),
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "Nvidia"), ("AMZN", "Amazon"),
    ("GOOGL", "Alphabet"), ("META", "Meta"), ("TSLA", "Tesla"), ("SPY", "ETF S&P 500"),
]


def filtrar_resultados(cotacoes: list[dict]) -> list[dict]:
    """Mantém ações/ETFs da B3 e das bolsas dos EUA, sem duplicatas, na ordem de relevância do Yahoo."""
    vistos, out = set(), []
    for c in cotacoes:
        simbolo = (c.get("symbol") or "").upper()
        if c.get("quoteType") not in TIPOS or simbolo in vistos:
            continue
        if simbolo.endswith(".SA"):
            if not B3.match(simbolo):
                continue
            bolsa = "B3"
        elif c.get("exchange") in BOLSAS_EUA and "." not in simbolo and "=" not in simbolo:
            bolsa = "EUA"
        else:
            continue
        vistos.add(simbolo)
        out.append({"simbolo": simbolo, "nome": c.get("longname") or c.get("shortname") or simbolo, "bolsa": bolsa})
    return out


def buscar_acoes(texto: str, limite: int = 15) -> list[dict]:
    texto = texto.strip()
    if len(texto) < 2:
        return []
    resultado = yf.Search(texto, max_results=limite * 2, news_count=0)
    achados = filtrar_resultados(resultado.quotes or [])
    # Código da B3 digitado sem sufixo (ex.: "bbas3") vai primeiro se existir
    exato = texto.upper() + ".SA"
    achados.sort(key=lambda a: a["simbolo"] not in (texto.upper(), exato))
    return achados[:limite]
