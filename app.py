"""Predict - Thomfig$420: escolha a ação, o prazo e o modelo; veja o previsto vs. o realizado.

Rodar: dois cliques em ABRIR_ANALISADOR.bat  (ou: uv run streamlit run app.py)
Dashboard técnico completo: uv run streamlit run app_completo.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analisador import NOME_APP, __version__
from analisador import aderencia as ad
from analisador.config import carregar_config
from analisador.dados.busca import POPULARES, buscar_acoes
from analisador.log import configurar_logging
from analisador.pipeline import analisar
from analisador.relatorio import formato as fmt
from analisador.relatorio.html import gerar_html

HORIZONTES = {21: "1 mês", 63: "3 meses", 126: "6 meses"}
PERIODOS = {"6 meses": 6, "1 ano": 12, "2 anos": 24, "4 anos": 48}
AZUL, AZUL_FAIXA, CINZA = "#2a78d6", "rgba(42,120,214,0.18)", "#898781"
# Sem zoom/arraste: no celular, o toque no gráfico rola a página em vez de ficar preso nele
CONFIG_GRAFICO = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "responsive": True}

NOME_MD = NOME_APP.replace("$", r"\$")  # em markdown, "$" abriria uma fórmula
st.set_page_config(page_title=f"{NOME_APP} v{__version__}", layout="wide")
configurar_logging("WARNING")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _buscar(texto: str) -> list[dict]:
    return buscar_acoes(texto)


@st.cache_resource(show_spinner=False, ttl=6 * 3600, max_entries=8)
def _analisar(ticker: str, rapido: bool):
    return analisar(ticker, carregar_config(rapido=rapido), horizontes=list(HORIZONTES))


def _rotulo_acao(a: dict) -> str:
    return f"{a['simbolo'].removesuffix('.SA')} · {a['nome']} ({a['bolsa']})"


def _layout(fig: go.Figure, prefixo: str) -> go.Figure:
    fig.update_layout(
        # margem inferior reserva espaço para a legenda (senão ela invade o elemento seguinte no celular)
        margin=dict(l=4, r=4, t=4, b=70), height=420, dragmode=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.1, x=0, font=dict(size=12)),
        yaxis=dict(tickprefix=prefixo, fixedrange=True), xaxis=dict(fixedrange=True),
    )
    return fig


def _grafico(fig: go.Figure, prefixo: str) -> None:
    st.plotly_chart(_layout(fig, prefixo), width="stretch", config=CONFIG_GRAFICO)


# ------------------------------------------------------------------ cabeçalho e escolha da ação
# Título em HTML: tamanho que se adapta à largura da tela (no celular o st.title ocupa meia tela).
# "&#36;" = "$" (literal "$" seria lido como fórmula pelo markdown)
st.markdown(
    f'<h1 style="font-size:clamp(1.7rem,7vw,2.6rem);line-height:1.15;margin:0 0 .2rem;padding:0">'
    f'{NOME_APP.replace("$", "&#36;")} <span style="font-size:.42em;font-weight:500;color:#898781;'
    f'white-space:nowrap">v{__version__}</span></h1>', unsafe_allow_html=True)
st.caption("Cada modelo é testado no passado como se estivesse prevendo de verdade, só com dados até aquela data. "
           "Depois a previsão é comparada com o preço que de fato aconteceu.")

busca = st.text_input("Buscar ação", placeholder="Nome ou código: Petrobras, WEG, BBAS3, Apple…")
if busca.strip():
    try:
        opcoes = _buscar(busca.strip())
    except Exception as e:
        st.error(f"A busca falhou ({e}). Tente de novo ou digite o código exato.")
        opcoes = []
    if not opcoes:
        st.info("Nenhuma ação da B3 ou dos EUA encontrada. Tente o nome da empresa ou o código (ex.: VALE3).")
else:
    opcoes = [{"simbolo": s, "nome": n, "bolsa": "B3" if s.endswith(".SA") else "EUA"} for s, n in POPULARES]
acao = st.selectbox("Ação", opcoes, format_func=_rotulo_acao, index=0 if opcoes else None, disabled=not opcoes)

b1, b2 = st.columns([1, 2])
rapido = b2.checkbox("Modo rápido", value=True,
                     help="Retreina os modelos a cada 3 meses em vez de todo mês: cerca de 2× mais rápido, "
                          "com previsões passadas um pouco menos atualizadas.")
if b1.button("Rodar modelos", type="primary", disabled=not acao, width="stretch"):
    st.session_state["ticker"] = (acao["simbolo"], rapido)
elif "ticker" not in st.session_state and st.query_params.get("acao"):
    # Link direto: ...?acao=PETR4.SA abre a análise sem precisar buscar
    st.session_state["ticker"] = (st.query_params["acao"].strip().upper(), True)
if "ticker" not in st.session_state:
    st.stop()

ticker, rapido_usado = st.session_state["ticker"]
st.query_params["acao"] = ticker
with st.spinner(f"Rodando os modelos em {ticker.removesuffix('.SA')} (de 1 a 3 minutos na primeira vez)…"):
    try:
        r = _analisar(ticker, rapido_usado)
    except Exception as e:
        st.error(f"Não foi possível analisar {ticker}: {e}")
        st.stop()
moeda = "R$ " if r.info.get("moeda") == "BRL" else f"{r.info.get('moeda', '')} "
st.divider()
st.subheader(f"{r.info.get('nome', r.ticker)} ({r.ticker.removesuffix('.SA')})")
st.caption(f"{moeda}{fmt.num(r.preco_atual)} em {r.data_ultima:%d/%m/%Y}")

# ------------------------------------------------------------------ prazo, veredito e modelo
por_h = {h.h: h for h in r.horizontes}
h_escolhido = st.segmented_control("Prazo da previsão", list(HORIZONTES), format_func=HORIZONTES.get,
                                   default=63, required=True, width="stretch")
hr = por_h[h_escolhido]
ranking = ad.tabela_aderencia(hr)
melhor = ranking.iloc[0]
rw = ranking.set_index("chave").loc[ad.REFERENCIA]

if melhor["chave"] == ad.REFERENCIA:
    veredito = "Nenhum modelo mais sofisticado superou a simples tendência histórica neste prazo."
elif melhor["melhor_que_rw"]:
    veredito = f"A vantagem sobre o random walk ({fmt.pct(rw['erro_medio'])} de erro) é estatisticamente significativa."
else:
    veredito = (f"Mas a diferença para o random walk ({fmt.pct(rw['erro_medio'])} de erro) **não** é estatisticamente "
                f"significativa: pode ser sorte.")
st.success(f"**Maior aderência em {HORIZONTES[h_escolhido]}: {melhor['modelo']}**, com erro médio de "
           f"{fmt.pct(melhor['erro_medio'])} no preço previsto. {veredito}")

modelo = st.selectbox("Modelo", list(ranking["chave"]), index=0,
                      format_func=lambda c: ("★ " if c == melhor["chave"] else "") + ad.nome(c),
                      help="Em ordem de aderência. ★ = maior aderência no prazo escolhido.")
st.caption(ad.MODELOS.get(modelo, ("", ""))[1])
linha = ranking.set_index("chave").loc[modelo]
m1, m2, m3 = st.columns(3)
m1.metric("Erro médio do preço", fmt.pct(linha["erro_medio"]),
          help="Média de |preço previsto ÷ preço realizado − 1|. Quanto menor, maior a aderência.")
m2.metric("Dentro da faixa prevista", fmt.pct(linha["dentro_da_faixa"], 0),
          help="Faixa de 80%: o ideal é ficar perto de 80%. Muito acima = faixa larga demais; muito abaixo = otimista demais.")
m3.metric("Acerto da direção", fmt.pct(linha["acerto_direcao"], 0),
          help="Sobe ou cai. Compare com os outros modelos no Ranking: num período de alta forte, todos acertam muito.")

# ------------------------------------------------------------------ abas
aba_hist, aba_proj, aba_rank = st.tabs(["Previsto × realizado", "Próximos meses", "Ranking"])

with aba_hist:
    periodo = st.segmented_control("Período", list(PERIODOS), default="2 anos", required=True,
                                   key="periodo_hist", width="stretch")
    pvr = ad.previsto_vs_realizado(hr, modelo)
    pvr = pvr[pvr.index >= pvr.index[-1] - pd.DateOffset(months=PERIODOS[periodo])]
    st.caption(f"Linha azul: preço que o modelo previa {HORIZONTES[h_escolhido]} antes. Cinza: o que aconteceu.")
    _grafico(go.Figure([
        go.Scatter(x=pvr.index, y=pvr["faixa_max"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
        go.Scatter(x=pvr.index, y=pvr["faixa_min"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=AZUL_FAIXA,
                   name="Faixa prevista (80%)", hoverinfo="skip"),
        go.Scatter(x=pvr.index, y=pvr["previsto"], name="Previsto", line=dict(color=AZUL, width=2)),
        go.Scatter(x=pvr.index, y=pvr["realizado"], name="Realizado", line=dict(color=CINZA, width=2)),
    ]), moeda)

with aba_proj:
    meses_hist = st.segmented_control("Histórico no gráfico", list(PERIODOS)[:3], default="6 meses", required=True,
                                      key="periodo_proj", width="stretch")
    proj = ad.projecao(r, modelo)
    hist = r.graficos["preco"]["close"]
    hist = hist[hist.index >= hist.index[-1] - pd.DateOffset(months=PERIODOS[meses_hist])]
    st.caption(f"Projeção de {ad.nome(modelo)} a partir de hoje.")
    _grafico(go.Figure([
        go.Scatter(x=proj.index, y=proj["faixa_max"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
        go.Scatter(x=proj.index, y=proj["faixa_min"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=AZUL_FAIXA,
                   name="Faixa provável (80%)", hoverinfo="skip"),
        go.Scatter(x=proj.index, y=proj["previsto"], name="Projeção", mode="lines+markers",
                   line=dict(color=AZUL, width=2, dash="dot"), marker=dict(size=8)),
        go.Scatter(x=hist.index, y=hist, name="Preço", line=dict(color=CINZA, width=2)),
    ]), moeda)
    tab_proj = proj.iloc[1:]
    st.dataframe(pd.DataFrame({
        "Prazo": tab_proj["horizonte"],
        "Previsto": tab_proj["previsto"].map(lambda v: moeda + fmt.num(v)),
        "Faixa 80%": [f"{fmt.num(a)} – {fmt.num(b)}" for a, b in zip(tab_proj["faixa_min"], tab_proj["faixa_max"])],
        "Acima de hoje": tab_proj["p_alta"].map(lambda v: fmt.pct(v, 0)),
    }), hide_index=True, width="stretch")
    st.caption("Faixa 80%: o preço deve terminar dentro dela em 8 de cada 10 casos. "
               "Acima de hoje: chance estimada de o preço estar acima do atual no prazo.")
    sem_vantagem = [h.nome for h in r.horizontes if not h.hoje.vantagem]
    if sem_vantagem:
        st.caption(f"Em {', '.join(sem_vantagem)}, nenhum modelo mostrou vantagem estatística para prever a DIREÇÃO: "
                   f"a chance de estar acima de hoje deve ser lida como incerteza, não como sinal.")

with aba_rank:
    st.caption(f"{HORIZONTES[h_escolhido]} · {fmt.inteiro(int(melhor['n']))} previsões testadas, "
               f"de {hr.periodo_oos[0]} a {hr.periodo_oos[1]}")
    st.dataframe(pd.DataFrame({
        "Modelo": [("★ " if c == melhor["chave"] else "") + n for c, n in zip(ranking["chave"], ranking["modelo"])],
        "Erro médio": ranking["erro_medio"].map(fmt.pct),
        "Na faixa 80%": ranking["dentro_da_faixa"].map(lambda v: fmt.pct(v, 0)),
        "Direção": ranking["acerto_direcao"].map(lambda v: fmt.pct(v, 0)),
        "Melhor que RW?": ["referência" if c == ad.REFERENCIA else ("sim" if s else "não")
                           for c, s in zip(ranking["chave"], ranking["melhor_que_rw"])],
    }), hide_index=True, width="stretch")
    with st.expander("Como ler"):
        st.markdown(
            "- **Erro médio**: diferença média entre o preço previsto (mediana) e o realizado. O ranking ordena por ele.\n"
            "- **Na faixa 80%**: quantas vezes o preço caiu dentro da faixa prevista; o ideal é perto de 80%.\n"
            "- **Melhor que RW?**: teste de Diebold-Mariano (5%) contra o random walk, considerando a sobreposição "
            "das previsões. \"Não\" = a diferença pode ser sorte do período.\n"
            "- As previsões do passado foram feitas sem olhar o futuro: modelos retreinados periodicamente só com "
            "dados anteriores, com intervalo de segurança do tamanho do prazo.")
    st.download_button("Baixar relatório técnico completo (HTML)", gerar_html(r), file_name=f"{r.ticker}.html",
                       mime="text/html", width="stretch")

st.caption(f"{NOME_MD} · v{__version__}")
