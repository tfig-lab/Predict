"""Visão simplificada: aderência de cada modelo ao que de fato aconteceu.

Para cada data do passado, o modelo previu o preço h pregões à frente usando só dados
até aquela data (walk-forward). Aqui comparamos essa previsão com o preço realizado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analisador.tipos import ResultadoAnalise, ResultadoHorizonte
from analisador.validacao.testes_estatisticos import diebold_mariano

MODELOS = {
    "ensemble": ("Combinado (ensemble)", "Média ponderada de todos os modelos, com pesos pelo desempenho recente."),
    "passeio_aleatorio": ("Random walk (tendência histórica)", "Supõe que o preço segue a tendência e a volatilidade médias do passado."),
    "garch": ("Volatilidade GARCH", "Tendência histórica, com a incerteza ajustada à volatilidade do momento."),
    "lgbm": ("Machine learning (LightGBM)", "Aprende padrões de indicadores técnicos, de mercado e macro."),
    "momentum": ("Momentum 12 meses", "Usa o desempenho dos últimos 12 meses para projetar a direção."),
    "taxa_base": ("Distribuição histórica", "Repete a distribuição de retornos observada no passado, sem condicionar a nada."),
    "fundacional": ("Chronos-Bolt (IA)", "Modelo fundacional de séries temporais, sem treino específico."),
}
REFERENCIA = "passeio_aleatorio"


def nome(chave: str) -> str:
    return MODELOS.get(chave, (chave, ""))[0]


def _avaliaveis(h: ResultadoHorizonte) -> pd.Series:
    s = h.serie_oos
    return s["calibrado"].astype(bool) & s["y"].notna()


def tabela_aderencia(h: ResultadoHorizonte, alfa: float = 0.05) -> pd.DataFrame:
    """Uma linha por modelo, ordenada da maior para a menor aderência (menor erro médio do preço)."""
    m = _avaliaveis(h)
    y = h.serie_oos.loc[m, "y"].to_numpy()
    erros = {}
    linhas = []
    for chave, df in h.previsoes.items():
        d = df[m]
        erro = np.abs(np.exp(d["q50"].to_numpy() - y) - 1)       # |preço previsto / realizado - 1|
        erros[chave] = erro
        linhas.append({
            "chave": chave, "modelo": nome(chave),
            "erro_medio": float(erro.mean()),
            "dentro_da_faixa": float(np.mean((y >= d["q10"].to_numpy()) & (y <= d["q90"].to_numpy()))),
            "acerto_direcao": float(np.mean((d["p_alta"].to_numpy() >= 0.5) == (y > 0))),
            "n": int(m.sum()),
        })
    tab = pd.DataFrame(linhas)
    ref = erros.get(REFERENCIA)
    tab["p_vs_random_walk"] = [
        np.nan if c == REFERENCIA or ref is None else diebold_mariano(erros[c], ref, h.h)["p_valor"] for c in tab["chave"]
    ]
    tab["melhor_que_rw"] = tab["p_vs_random_walk"] < alfa
    return tab.sort_values("erro_medio").reset_index(drop=True)


def previsto_vs_realizado(h: ResultadoHorizonte, modelo: str) -> pd.DataFrame:
    """Indexado pela data-ALVO: preço realizado e o que o modelo previu h pregões antes."""
    s = h.serie_oos
    m = _avaliaveis(h) & s["data_alvo"].notna()
    d = h.previsoes[modelo][m]
    base = s.loc[m, "preco"]
    out = pd.DataFrame({
        "realizado": base * np.exp(s.loc[m, "y"]),
        "previsto": base * np.exp(d["q50"]),
        "faixa_min": base * np.exp(d["q10"]),
        "faixa_max": base * np.exp(d["q90"]),
        "data_previsao": s.index[m],
    })
    out.index = pd.DatetimeIndex(s.loc[m, "data_alvo"])
    return out


def projecao(r: ResultadoAnalise, modelo: str) -> pd.DataFrame:
    """Projeção de hoje para cada horizonte analisado: mediana e faixa de 80%, em preço."""
    hoje = r.data_ultima
    linhas = [{"data": hoje, "horizonte": "Hoje", "previsto": r.preco_atual, "faixa_min": r.preco_atual,
               "faixa_max": r.preco_atual, "p_alta": np.nan}]
    for h in r.horizontes:
        if modelo not in h.previsoes:
            continue
        u = h.previsoes[modelo].iloc[-1]
        linhas.append({
            "data": pd.bdate_range(hoje, periods=h.h + 1)[-1], "horizonte": h.nome,
            "previsto": r.preco_atual * np.exp(u["q50"]), "faixa_min": r.preco_atual * np.exp(u["q10"]),
            "faixa_max": r.preco_atual * np.exp(u["q90"]), "p_alta": float(u["p_alta"]),
        })
    return pd.DataFrame(linhas).set_index("data")
