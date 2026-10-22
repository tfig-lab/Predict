"""Relatório HTML autocontido (plotly.js embutido; funciona offline)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly

from analisador import __version__
from analisador.features import NOMES
from analisador.horizonte import NOMES_MODELOS
from analisador.relatorio import formato as fmt
from analisador.tipos import ResultadoAnalise

TEMPLATE = Path(__file__).with_name("template.html")
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


def _limpo(x):
    if isinstance(x, dict):
        return {str(k): _limpo(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_limpo(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    return x


def _datas(idx) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in idx]


def _serie(s: pd.Series, casas: int = 4) -> list:
    return [None if pd.isna(v) else round(float(v), casas) for v in s]


def _nome_feature(c: str) -> str:
    return NOMES.get(c, c)


def resumo_executivo(r: ResultadoAnalise) -> list[str]:
    moeda = "R$" if r.info.get("moeda") == "BRL" else r.info.get("moeda", "")
    com = [h for h in r.horizontes if h.hoje.vantagem]
    frases = []
    if not com:
        frases.append("Em nenhum dos horizontes analisados o modelo mostrou vantagem estatística sobre o acaso para "
                      "prever a DIREÇÃO do preço. Isso é o resultado esperado para a maioria das ações: o retorno é "
                      "quase imprevisível. As probabilidades de alta abaixo não devem ser lidas como sinal.")
    else:
        for h in com:
            fx = h.acerto_faixas.get(h.hoje.rotulo, {})
            frases.append(f"{h.nome}: {h.hoje.mensagem.lower()}, com confiança {h.hoje.rotulo.lower()} "
                          f"({h.hoje.score:.0f}/100). Nas vezes em que a confiança esteve nessa faixa, o modelo acertou "
                          f"{fmt.pct(fx.get('acerto'), 0)} (n = {fmt.inteiro(fx.get('n', 0))}).")
        sem = [h.nome for h in r.horizontes if not h.hoje.vantagem]
        if sem:
            frases.append(f"Nos demais horizontes ({', '.join(sem)}) não há vantagem estatística.")
    frases.append(f"O que é mais confiável aqui é a INCERTEZA: a volatilidade prevista pelo GARCH ({r.garch['tipo']}) é de "
                  f"{fmt.pct(r.garch['vol_atual'], 0)} ao ano, contra {fmt.pct(r.garch['vol_longo_prazo'], 0)} de longo prazo, "
                  f"e as faixas de preço passaram por validação de cobertura fora da amostra.")
    por_h = {h.h: h for h in r.horizontes}
    for hh in (63, 126):
        if hh in por_h:
            p = por_h[hh].hoje
            cob = por_h[hh].metricas_quantis["ensemble_conformal"]["cobertura_80"]
            frases.append(f"Em {por_h[hh].nome}, a faixa de 80% vai de {moeda} {fmt.num(p.precos[0.10])} a "
                          f"{moeda} {fmt.num(p.precos[0.90])} (no histórico, intervalos assim contiveram o resultado "
                          f"{fmt.pct(cob, 0)} das vezes).")
    ev = r.monte_carlo["eventos"].get(63)
    if ev:
        frases.append(f"Pela simulação Monte Carlo ({fmt.inteiro(r.monte_carlo['n'])} trajetórias), a chance de subir mais "
                      f"de 10% em 3 meses é {fmt.pct(ev['p_mais_10'], 0)} e a de cair mais de 15% é {fmt.pct(ev['p_menos_15'], 0)}.")
    frases.append(f"Regime de mercado atual (HMM): {r.hmm['atual'].lower()}, com probabilidade de {fmt.pct(max(r.hmm['probs']), 0)}.")
    return frases


def gerar_html(r: ResultadoAnalise) -> str:
    g = r.graficos
    preco = g["preco"]
    reg = g["regime"]
    vol = g["vol"]
    leque = g["leque"]
    horizontes = []
    for h in r.horizontes:
        p = h.hoje
        curvas = h.backtest["curvas"].resample("W").last()
        horizontes.append({
            "h": h.h, "nome": h.nome, "periodo": h.periodo_oos, "n": h.n_oos,
            "hoje": {
                "p_alta": p.p_alta, "p_bruta": p.p_alta_bruta, "p_base": p.p_taxa_base, "p_indice": p.p_supera_indice,
                "precos": {str(k): v for k, v in p.precos.items()}, "ret_mediano": p.retorno_mediano,
                "vol": p.vol_periodo, "vol_anual": p.vol_anualizada, "score": p.score, "rotulo": p.rotulo,
                "vantagem": p.vantagem, "componentes": p.componentes, "mensagem": p.mensagem,
                "pesos": {NOMES_MODELOS.get(k, k): v for k, v in p.pesos.items()},
            },
            "faixas": h.acerto_faixas,
            "modelos": [{"nome": NOMES_MODELOS.get(k, k), "chave": k, **v} for k, v in h.metricas_direcao.items()],
            "quantis": [{"nome": NOMES_MODELOS.get(k, k), "chave": k, **v} for k, v in h.metricas_quantis.items()],
            "vol": h.metricas_vol, "relativo": h.metricas_relativo, "testes": h.testes,
            "vantagem_hist": h.vantagem_historica, "calibracao": h.calibracao,
            "backtest": {"estrategia": h.backtest["estrategia"], "buy_and_hold": h.backtest["buy_and_hold"],
                         "datas": _datas(curvas.index), "estr": _serie(curvas["estrategia"]), "bh": _serie(curvas["buy_and_hold"])},
            "shap_global": [{"nome": _nome_feature(c), "valor": v} for c, v in h.shap_global],
            "shap_local": [{"nome": _nome_feature(c), "valor": v, "x": x} for c, v, x in h.shap_local],
        })
    dados = _limpo({
        "ticker": r.ticker, "info": r.info, "indice": r.nome_indice, "mercado": r.mercado,
        "data": r.data_ultima.strftime("%d/%m/%Y"), "preco": r.preco_atual, "tecnico": r.tecnico,
        "fundamentos": r.fundamentos, "avisos": r.avisos, "resumo": resumo_executivo(r),
        "garch": r.garch, "hmm": r.hmm, "mc": r.monte_carlo,
        "serie": {"datas": _datas(preco.index), "close": _serie(preco["close"]), "mm50": _serie(preco["mm50"]),
                  "mm200": _serie(preco["mm200"]), "estado": _serie(reg["hmm_estado"], 0), "k": _serie(reg["hmm_k"], 0)},
        "leque": {"datas": _datas(leque.index), **{str(q): _serie(leque[q], 3) for q in leque.columns}},
        "vol": {"datas": _datas(vol.index), "garch": _serie(vol["garch"]), "realizada": _serie(vol["realizada_21"]),
                "futuro_datas": _datas(g["vol_futura"].index), "futuro": _serie(g["vol_futura"])},
        "horizontes": horizontes,
        "tempo": r.tempo_total_s, "config": r.config_resumo,
        "gerado": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "versao": __version__,
    })
    html = TEMPLATE.read_text(encoding="utf-8")
    return (html.replace("__TITULO__", r.ticker)
                .replace("/*__PLOTLY__*/", PLOTLY_JS.read_text(encoding="utf-8"))
                .replace("__DADOS__", json.dumps(dados, ensure_ascii=False).replace("</", "<\\/")))


def salvar_html(r: ResultadoAnalise, pasta: str | Path) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / f"{r.ticker.replace('^', '')}_{r.data_ultima:%Y-%m-%d}.html"
    arq.write_text(gerar_html(r), encoding="utf-8")
    return arq
