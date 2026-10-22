"""Dashboard técnico completo: uv run streamlit run app_completo.py  (a tela simples é o app.py)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analisador import NOME_APP, __version__
from analisador.config import carregar_config
from analisador.dados.busca import POPULARES, buscar_acoes
from analisador.log import configurar_logging
from analisador.pipeline import analisar
from analisador.relatorio import formato as fmt
from analisador.relatorio.html import gerar_html

st.set_page_config(page_title=f"{NOME_APP} v{__version__} · técnico", layout="wide")
configurar_logging("WARNING")
cfg_base = carregar_config()


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _buscar(texto: str) -> list[dict]:
    return buscar_acoes(texto)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _rodar(ticker: str, horizontes: tuple[int, ...], rapido: bool) -> tuple[dict, str]:
    cfg = carregar_config(rapido=rapido)
    r = analisar(ticker, cfg, horizontes=list(horizontes))
    linhas = []
    for h in r.horizontes:
        p = h.hoje
        fx = h.acerto_faixas.get(p.rotulo, {})
        linhas.append({
            "Horizonte": h.nome,
            "P(alta)": fmt.pct(p.p_alta, 0) if p.vantagem else f"sem vantagem (base {fmt.pct(p.p_taxa_base, 0)})",
            "Retorno mediano": fmt.pct(p.retorno_mediano, 1, True),
            "Faixa P10–P90": f"{fmt.num(p.precos[0.10])} – {fmt.num(p.precos[0.90])}",
            "Vol. prevista": f"{fmt.pct(p.vol_periodo)} ({fmt.pct(p.vol_anualizada, 0)} a.a.)",
            "Confiança": f"{p.rotulo} ({p.score:.0f})",
            "Acerto na faixa": f"{fmt.pct(fx.get('acerto'), 0)} (n={fx.get('n', 0)})" if fx.get("n") else "—",
        })
    resumo = {"titulo": f"{r.info.get('nome', r.ticker)} ({r.ticker})", "preco": r.preco_atual,
              "moeda": "R$" if r.info.get("moeda") == "BRL" else r.info.get("moeda", ""),
              "data": r.data_ultima.strftime("%d/%m/%Y"), "tabela": linhas, "avisos": r.avisos, "tempo": r.tempo_total_s}
    return resumo, gerar_html(r)


def _rotulo(acao: dict) -> str:
    codigo = acao["simbolo"].removesuffix(".SA")
    return f"{codigo} · {acao['nome']} ({acao['bolsa']})"


st.title(f"{NOME_APP.replace('$', chr(92) + '$')} `v{__version__}`")
st.caption("Dashboard técnico: tendência por horizonte com grau de confiabilidade medido fora da amostra.")

busca = st.text_input("Buscar ação", placeholder="Nome ou código: Petrobras, WEG, BBAS3, Apple, NVDA…",
                      help="Busca no Yahoo Finance entre ações e ETFs da B3 e das bolsas dos EUA. Tecle Enter para buscar.")
if busca.strip():
    try:
        opcoes = _buscar(busca.strip())
    except Exception as e:
        st.error(f"A busca no Yahoo Finance falhou ({e}). Tente de novo ou digite o código exato.")
        opcoes = []
    if not opcoes:
        st.info("Nenhuma ação da B3 ou dos EUA encontrada para essa busca. Tente o nome da empresa ou o código (ex.: VALE3).")
else:
    opcoes = [{"simbolo": s, "nome": n, "bolsa": "B3" if s.endswith(".SA") else "EUA"} for s, n in POPULARES]

c1, c2, c3 = st.columns([3, 3, 1])
escolha = c1.selectbox("Ação", opcoes, format_func=_rotulo, index=0 if opcoes else None,
                       placeholder="Busque uma ação acima", disabled=not opcoes)
horizontes = c2.multiselect("Horizontes", cfg_base.horizontes, default=cfg_base.horizontes,
                            format_func=cfg_base.nome_horizonte)
rapido = c3.checkbox("Modo rápido", value=False, help="Retreina a cada 63 pregões em vez de 21 (≈2× mais rápido).")
analisar_agora = st.button("Analisar", type="primary", disabled=not (escolha and horizontes))

if analisar_agora and escolha:
    st.session_state["ultima"] = (escolha["simbolo"], tuple(sorted(horizontes)), rapido)

if "ultima" in st.session_state:
    ticker, hs, rap = st.session_state["ultima"]
    with st.spinner(f"Analisando {ticker.removesuffix('.SA')}: baixando dados, ajustando GARCH/HMM e validando "
                    f"cada horizonte (cerca de 1 minuto na primeira vez)…"):
        try:
            resumo, html = _rodar(ticker, hs, rap)
        except Exception as e:
            st.error(f"Não foi possível analisar {ticker}: {e}")
            st.stop()
    st.subheader(resumo["titulo"])
    st.write(f"Último preço: **{resumo['moeda']} {fmt.num(resumo['preco'])}** em {resumo['data']} · "
             f"análise em {resumo['tempo']:.0f} s")
    st.dataframe(pd.DataFrame(resumo["tabela"]), hide_index=True, width="stretch")
    for a in resumo["avisos"]:
        st.warning(a)
    st.download_button("Baixar relatório HTML", html, file_name=f"{ticker}.html", mime="text/html")
    # HTML gerado pelo próprio analisador (textos externos, como o nome da empresa, são escapados no template)
    st.iframe(html, height=7400)
