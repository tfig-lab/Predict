"""Cotações e fundamentos via yfinance."""
from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

from analisador.log import obter_logger

log = obter_logger(__name__)
COLUNAS = ["Open", "High", "Low", "Close", "Volume"]


def normalizar_ticker(ticker: str) -> str:
    """PETR4 -> PETR4.SA. Tickers com sufixo, índices (^), câmbio (=X) ou só letras ficam como estão."""
    t = ticker.strip().upper()
    if "." not in t and "=" not in t and not t.startswith("^") and any(c.isdigit() for c in t):
        t += ".SA"
    return t


def mercado_do_ticker(ticker: str) -> str:
    return "B3" if ticker.endswith(".SA") or ticker == "^BVSP" else "EUA"


def baixar_yahoo(ticker: str, inicio: date | None = None) -> pd.DataFrame:
    """OHLCV diário ajustado por desdobramentos e proventos. `inicio=None` = histórico máximo."""
    tk = yf.Ticker(ticker)
    if inicio is None:
        hist = tk.history(period="max", auto_adjust=True)
    else:
        hist = tk.history(start=inicio.isoformat(), auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"Yahoo Finance não retornou dados para '{ticker}'")
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    hist = hist[~hist.index.duplicated(keep="last")].sort_index()
    for c in COLUNAS:
        if c not in hist:
            hist[c] = float("nan")
    return hist[COLUNAS].astype(float)


def info_e_fundamentos(ticker: str) -> tuple[dict, dict]:
    """Nome/setor/moeda e fundamentos ATUAIS (não históricos: só servem de contexto)."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:  # yfinance falha com frequência neste endpoint
        log.warning("não foi possível obter info/fundamentos", extra={"ticker": ticker, "erro": str(e)})
        info = {}
    basico = {
        "nome": info.get("longName") or info.get("shortName") or ticker,
        "setor": info.get("sector") or "",
        "moeda": info.get("currency") or ("BRL" if ticker.endswith(".SA") else "USD"),
    }
    dy = info.get("dividendYield")
    if dy is not None and dy > 1:  # algumas versões do Yahoo devolvem em %, outras em fração
        dy = dy / 100
    fund = {
        "P/L": info.get("trailingPE"),
        "P/VP": info.get("priceToBook"),
        "Dividend yield": dy,
        "ROE": info.get("returnOnEquity"),
        "Margem líquida": info.get("profitMargins"),
    }
    return basico, {k: float(v) for k, v in fund.items() if isinstance(v, (int, float))}
