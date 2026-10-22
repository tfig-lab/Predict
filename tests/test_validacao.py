import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analisador.validacao.backtest import backtest_limiar
from analisador.validacao.metricas import metricas_direcao, metricas_quantis, pinball_pontual
from analisador.validacao.testes_estatisticos import binomial_efetivo, diebold_mariano, variancia_hac
from analisador.validacao.walkforward import divisao_interna, gerar_janelas


@pytest.mark.parametrize("h", [1, 5, 21, 63, 126])
@pytest.mark.parametrize("embargo_igual_h", [True, False])
def test_walkforward_respeita_purga_e_embargo(h, embargo_igual_h):
    embargo = h if embargo_igual_h else 0
    janelas = gerar_janelas(n=3000, h=h, treino_minimo=756, passo=21, embargo=embargo)
    assert janelas, "deve haver janelas"
    for j in janelas:
        ultimo_treino = j.treino_fim - 1
        # O rótulo da última linha de treino usa o preço em ultimo_treino + h,
        # que precisa ser anterior ao início do teste, com `embargo` linhas de folga
        assert ultimo_treino + h + embargo < j.teste_ini
        assert j.teste_ini < j.teste_fim <= 3000
    # Os testes cobrem, sem sobreposição e sem buracos, do primeiro teste até o fim
    for a, b in zip(janelas, janelas[1:]):
        assert a.teste_fim == b.teste_ini
    assert janelas[-1].teste_fim == 3000


def test_divisao_interna_purgada():
    fim_sub, ini_val = divisao_interna(1000, h=21, fracao_validacao=0.2)
    assert ini_val == 800
    assert fim_sub - 1 + 21 < ini_val


def test_variancia_hac_sem_lags_e_a_variancia_amostral():
    d = np.random.default_rng(0).normal(size=500)
    assert np.isclose(variancia_hac(d, 0), np.var(d))


def test_dm_detecta_modelo_melhor_e_nao_inventa_diferenca():
    rng = np.random.default_rng(1)
    ref = rng.normal(1.0, 0.3, 2000) ** 2
    melhor = ref * 0.9
    assert diebold_mariano(melhor, ref, h=1)["p_valor"] < 0.01
    # Mesma distribuição de perda -> p-valores ~ uniformes: rejeição perto de 5%
    rejeicoes = 0
    for s in range(200):
        r = np.random.default_rng(100 + s)
        a, b = r.normal(size=500) ** 2, r.normal(size=500) ** 2
        rejeicoes += diebold_mariano(a, b, h=1)["p_valor"] < 0.05
    assert 0.01 < rejeicoes / 200 < 0.12


def test_dm_com_sobreposicao_e_mais_conservador():
    """Perdas com autocorrelação forte (rótulos sobrepostos) inflam o DM ingênuo."""
    rng = np.random.default_rng(2)
    ruido = rng.normal(size=3000)
    d = np.convolve(ruido, np.ones(21) / 21, mode="same") + 0.02
    ingenuo = diebold_mariano(np.zeros_like(d), d, h=1)["estatistica"]
    hac = diebold_mariano(np.zeros_like(d), d, h=21)["estatistica"]
    # A variância HAC reconhece a autocorrelação: estatística bem menor em módulo
    assert abs(hac) < abs(ingenuo) / 2


def test_binomial_efetivo_usa_subamostra_sem_sobreposicao():
    acertos = np.ones(630, dtype=bool)
    r = binomial_efetivo(acertos, h=21, p0=0.5)
    assert r["n_efetivo"] == 30
    assert np.isclose(r["p_valor"], stats.binomtest(30, 30, 0.5, alternative="greater").pvalue)


def test_metricas_direcao_casos_conhecidos():
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    perfeito = metricas_direcao(y, y * 0.8 + 0.1, p_ref=np.full(8, 0.5))
    assert perfeito["acuracia"] == 1 and perfeito["auc"] == 1 and perfeito["bss"] > 0.5
    moeda = metricas_direcao(y, np.full(8, 0.5), p_ref=np.full(8, 0.5))
    assert moeda["auc"] == 0.5 and moeda["bss"] == 0


def test_pinball_e_cobertura():
    quantis = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    rng = np.random.default_rng(3)
    y = rng.normal(size=20000)
    Q = np.tile(stats.norm.ppf(quantis), (len(y), 1))
    m = metricas_quantis(y, Q, quantis)
    assert abs(m["cobertura_80"] - 0.8) < 0.01 and abs(m["cobertura_90"] - 0.9) < 0.01
    errado = np.tile(stats.norm.ppf(quantis) * 2, (len(y), 1))
    assert pinball_pontual(y, errado, quantis).mean() > m["pinball"]


def _serie(n=600, semente=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(semente).normal(0.0005, 0.02, n))), index=idx)
    return idx, close


def test_backtest_sempre_comprado_sem_custo_igual_buy_and_hold():
    idx, close = _serie()
    caixa = pd.Series(0.0, index=idx)
    res = backtest_limiar(pd.Series(0.99, index=idx), close, caixa, limiar=0.5, custo=0.0, defasagem=0)
    # Com defasagem 0 a posição começa no 2º pregão; a partir daí os retornos coincidem
    e, b = res["curvas"]["estrategia"], res["curvas"]["buy_and_hold"]
    assert np.allclose((e / e.iloc[1]).iloc[1:], (b / b.iloc[1]).iloc[1:])


def test_backtest_sempre_fora_rende_o_caixa_e_cobra_custos():
    idx, close = _serie()
    caixa = pd.Series(0.0004, index=idx)
    fora = backtest_limiar(pd.Series(0.1, index=idx), close, caixa, limiar=0.5, custo=0.001)
    assert np.isclose(fora["estrategia"]["retorno_total"], 1.0004 ** len(idx) - 1)
    assert fora["estrategia"]["operacoes"] == 0
    # Sinal alternando todo dia -> uma troca por pregão, custo come o resultado
    alternado = pd.Series(np.where(np.arange(len(idx)) % 2, 0.9, 0.1), index=idx)
    res = backtest_limiar(alternado, close, caixa, limiar=0.5, custo=0.001)
    assert res["estrategia"]["operacoes"] >= len(idx) - 3


def test_backtest_nao_usa_o_sinal_do_mesmo_dia():
    """Um oráculo que conhece o retorno de HOJE não pode lucrar: o sinal só vale após a execução."""
    idx, close = _serie(2000, semente=5)
    r = close.pct_change().fillna(0)
    oraculo_hoje = pd.Series(np.where(r > 0, 0.9, 0.1), index=idx)
    caixa = pd.Series(0.0, index=idx)
    res = backtest_limiar(oraculo_hoje, close, caixa, limiar=0.5, custo=0.0, defasagem=1)
    assert abs(res["estrategia"]["sharpe"]) < 1.0
