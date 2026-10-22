"""Backtest econômico simples: comprado quando P(alta) > limiar, caixa caso contrário."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _estatisticas(ret: pd.Series, caixa: pd.Series) -> dict[str, float]:
    anos = len(ret) / 252
    curva = (1 + ret).cumprod()
    excesso = ret - caixa
    neg = excesso[excesso < 0]
    dd = curva / curva.cummax() - 1
    return {
        "cagr": float(curva.iloc[-1] ** (1 / anos) - 1) if anos > 0 else float("nan"),
        "vol_anual": float(ret.std() * np.sqrt(252)),
        "sharpe": float(excesso.mean() / excesso.std() * np.sqrt(252)) if excesso.std() > 0 else 0.0,
        "sortino": float(excesso.mean() / np.sqrt((neg**2).mean()) * np.sqrt(252)) if len(neg) else float("nan"),
        "max_drawdown": float(dd.min()),
        "retorno_total": float(curva.iloc[-1] - 1),
    }


def backtest_limiar(
    p: pd.Series,
    close: pd.Series,
    caixa_diario: pd.Series,
    limiar: float,
    custo: float,
    defasagem: int = 1,
) -> dict:
    """p_t é conhecido no fechamento de t; a ordem é executada no fechamento de t+defasagem.

    A posição vale para o retorno do pregão seguinte à execução: pos_s = sinal_{s-1-defasagem}.
    O custo incide sobre cada mudança de posição (entrada ou saída).
    """
    p = p.dropna()
    idx = p.index
    r = close.pct_change().reindex(idx).fillna(0.0)
    cx = caixa_diario.reindex(idx).fillna(0.0)
    sinal = (p > limiar).astype(float)
    pos = sinal.shift(1 + defasagem).fillna(0.0)
    trocas = pos.diff().abs().fillna(pos.iloc[0])
    estrategia = pos * r + (1 - pos) * cx - custo * trocas
    # Buy-and-hold no MESMO período (paga uma entrada)
    bh = r.copy()
    bh.iloc[0] -= custo
    return {
        "estrategia": {**_estatisticas(estrategia, cx), "operacoes": int(trocas.sum()), "exposicao": float(pos.mean())},
        "buy_and_hold": {**_estatisticas(bh, cx), "operacoes": 1, "exposicao": 1.0},
        "curvas": pd.DataFrame({"estrategia": (1 + estrategia).cumprod(), "buy_and_hold": (1 + bh).cumprod()}),
        "inicio": idx[0], "fim": idx[-1],
    }
