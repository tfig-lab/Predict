"""Tela simplificada: escolha a ação, o prazo e o modelo; veja o previsto vs. o realizado.

Rodar: dois cliques em ABRIR_ANALISADOR.bat  (ou: uv run streamlit run app.py)
Dashboard técnico completo: uv run streamlit run app_completo.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analisador import aderencia as ad
from analisador.config import carregar_config
from analisador.dados.busca import POPULARES, buscar_acoes
from analisador.log import configurar_logging
from analisador.pipeline import analisar
from analisador.relatorio import formato as fmt
from analisador.relatorio.html import gerar_html

HORIZONTES = {21: "1 mês", 63: "3 meses", 126: "6 meses"}
AZUL, AZUL_FAIXA, CINZA = "#2a78d6", "rgba(42,120,214,0.18)", "#898781"

st.set_page_config(page_title="Aderência dos modelos", layout="wide")
configurar_logging("WARNING")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _buscar(texto: str) -> list[dict]:
    return buscar_acoes(texto)


@st.cache_resource(show_spinner=False, ttl=6 * 3600, max_entries=8)
def _analisar(ticker: str):
    return analisar(ticker, carregar_config(), horizontes=list(HORIZONTES))


def _rotulo_acao(a: dict) -> str:
    return f"{a['simbolo'].removesuffix('.SA')} · {a['nome']} ({a['bolsa']})"


def _layout(fig: go.Figure, prefixo: str) -> go.Figure:
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis=dict(tickprefix=prefixo))
    return fig


# ------------------------------------------------------------------ 1. escolha da ação
st.title("Aderência dos modelos ao realizado")
st.caption("Cada modelo é testado no passado como se estivesse prevendo de verdade: só com dados até aquela data. "
           "Depois comparamos com o preço que de fato aconteceu. Ferramenta educacional; não é recomendação de investimento.")

c1, c2 = st.columns([2, 3])
busca = c1.text_input("Buscar ação", placeholder="Nome ou código: Petrobras, WEG, BBAS3, Apple…")
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
acao = c2.selectbox("Ação", opcoes, format_func=_rotulo_acao, index=0 if opcoes else None, disabled=not opcoes)

if st.button("Rodar modelos", type="primary", disabled=not acao):
    st.session_state["ticker"] = acao["simbolo"]
if "ticker" not in st.session_state:
    st.stop()

ticker = st.session_state["ticker"]
with st.spinner(f"Rodando os modelos em {ticker.removesuffix('.SA')} (cerca de 1 minuto na primeira vez)…"):
    try:
        r = _analisar(ticker)
    except Exception as e:
        st.error(f"Não foi possível analisar {ticker}: {e}")
        st.stop()
moeda = "R$ " if r.info.get("moeda") == "BRL" else f"{r.info.get('moeda', '')} "
st.divider()
st.subheader(f"{r.info.get('nome', r.ticker)} ({r.ticker.removesuffix('.SA')}) · {moeda}{fmt.num(r.preco_atual)} em {r.data_ultima:%d/%m/%Y}")

# ------------------------------------------------------------------ 2. prazo e ranking
por_h = {h.h: h for h in r.horizontes}
h_escolhido = st.radio("Prazo da previsão", list(HORIZONTES), format_func=HORIZONTES.get, horizontal=True, index=1)
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

# ------------------------------------------------------------------ 3. modelo escolhido: previsto x realizado
chaves = list(ranking["chave"])
modelo = st.selectbox("Modelo", chaves, index=0,
                      format_func=lambda c: ("★ " if c == melhor["chave"] else "") + ad.nome(c),
                      help="Em ordem de aderência. ★ = maior aderência no prazo escolhido.")
st.caption(ad.MODELOS.get(modelo, ("", ""))[1])
linha = ranking.set_index("chave").loc[modelo]
m1, m2, m3 = st.columns(3)
m1.metric("Erro médio do preço previsto", fmt.pct(linha["erro_medio"]),
          help="Média de |preço previsto ÷ preço realizado − 1|. Quanto menor, maior a aderência.")
m2.metric("Vezes dentro da faixa prevista", fmt.pct(linha["dentro_da_faixa"], 0),
          help="Faixa de 80%: o ideal é ficar perto de 80%. Muito acima = faixa larga demais; muito abaixo = otimista demais.")
m3.metric("Acerto da direção (sobe/cai)", fmt.pct(linha["acerto_direcao"], 0),
          help="Com 50% o modelo seria uma moeda; compare com os outros modelos na tabela abaixo.")

pvr = ad.previsto_vs_realizado(hr, modelo)
pvr = pvr[pvr.index >= pvr.index[-1] - pd.DateOffset(years=4)]
fig = go.Figure([
    go.Scatter(x=pvr.index, y=pvr["faixa_max"], line=dict(width=0), hoverinfo="skip", showlegend=False),
    go.Scatter(x=pvr.index, y=pvr["faixa_min"], line=dict(width=0), fill="tonexty", fillcolor=AZUL_FAIXA,
               name="Faixa prevista (80%)", hoverinfo="skip"),
    go.Scatter(x=pvr.index, y=pvr["previsto"], name=f"Previsto {HORIZONTES[h_escolhido]} antes", line=dict(color=AZUL, width=2)),
    go.Scatter(x=pvr.index, y=pvr["realizado"], name="Realizado", line=dict(color=CINZA, width=2)),
])
st.markdown(f"**Previsto × realizado.** Em cada data, a linha azul mostra o preço que o modelo tinha previsto "
            f"{HORIZONTES[h_escolhido]} antes; a cinza, o que aconteceu (últimos 4 anos).")
st.plotly_chart(_layout(fig, moeda), width="stretch")

# ------------------------------------------------------------------ 4. projeção para os próximos meses
proj = ad.projecao(r, modelo)
hist = r.graficos["preco"]["close"]
hist = hist[hist.index >= hist.index[-1] - pd.DateOffset(years=1)]
fig2 = go.Figure([
    go.Scatter(x=proj.index, y=proj["faixa_max"], line=dict(width=0), hoverinfo="skip", showlegend=False),
    go.Scatter(x=proj.index, y=proj["faixa_min"], line=dict(width=0), fill="tonexty", fillcolor=AZUL_FAIXA,
               name="Faixa provável (80%)", hoverinfo="skip"),
    go.Scatter(x=proj.index, y=proj["previsto"], name="Projeção (mediana)", mode="lines+markers",
               line=dict(color=AZUL, width=2, dash="dot"), marker=dict(size=8)),
    go.Scatter(x=hist.index, y=hist, name="Preço (último ano)", line=dict(color=CINZA, width=2)),
])
st.markdown(f"**Projeção para os próximos meses: {ad.nome(modelo)}**")
st.plotly_chart(_layout(fig2, moeda), width="stretch")
tab_proj = proj.iloc[1:].assign(
    **{"Preço previsto": lambda d: d["previsto"].map(lambda v: moeda + fmt.num(v)),
       "Faixa provável (80%)": lambda d: [f"{moeda}{fmt.num(a)} – {fmt.num(b)}" for a, b in zip(d["faixa_min"], d["faixa_max"])],
       "Chance de estar acima de hoje": lambda d: d["p_alta"].map(lambda v: fmt.pct(v, 0))})
st.dataframe(tab_proj[["horizonte", "Preço previsto", "Faixa provável (80%)", "Chance de estar acima de hoje"]]
             .rename(columns={"horizonte": "Prazo"}), hide_index=True, width="stretch")
sem_vantagem = [h.nome for h in r.horizontes if not h.hoje.vantagem]
if sem_vantagem:
    st.caption(f"Atenção: em {', '.join(sem_vantagem)}, nenhum modelo mostrou vantagem estatística para prever a "
               f"DIREÇÃO. A chance de estar acima de hoje deve ser lida como incerteza, não como sinal.")

# ------------------------------------------------------------------ 5. ranking completo
st.markdown(f"**Ranking de aderência em {HORIZONTES[h_escolhido]}** "
            f"({fmt.inteiro(int(melhor['n']))} previsões testadas, de {hr.periodo_oos[0]} a {hr.periodo_oos[1]})")
st.dataframe(pd.DataFrame({
    "Modelo": [("★ " if c == melhor["chave"] else "") + n for c, n in zip(ranking["chave"], ranking["modelo"])],
    "Erro médio do preço": ranking["erro_medio"].map(fmt.pct),
    "Dentro da faixa 80% (ideal ≈ 80%)": ranking["dentro_da_faixa"].map(lambda v: fmt.pct(v, 0)),
    "Acerto de direção": ranking["acerto_direcao"].map(lambda v: fmt.pct(v, 0)),
    "Melhor que o random walk?": [
        "referência" if c == ad.REFERENCIA else ("sim" if s else "não (pode ser sorte)")
        for c, s in zip(ranking["chave"], ranking["melhor_que_rw"])],
}), hide_index=True, width="stretch")

with st.expander("Como ler e detalhes técnicos"):
    st.markdown(
        "- **Aderência** = erro médio entre o preço previsto (mediana) e o realizado. O ranking ordena por ele.\n"
        "- **Melhor que o random walk?** Teste de Diebold-Mariano (5%), que leva em conta a sobreposição das previsões. "
        "\"Não\" significa que a diferença de erro pode ser só sorte do período.\n"
        "- Todas as previsões do passado foram feitas sem olhar o futuro: modelos retreinados mês a mês só com dados "
        "anteriores, com intervalo de segurança do tamanho do prazo.\n"
        "- Retornos de ações são quase imprevisíveis; é normal que os modelos fiquem muito próximos entre si.")
    st.download_button("Baixar relatório técnico completo (HTML)", gerar_html(r), file_name=f"{r.ticker}.html",
                       mime="text/html")
