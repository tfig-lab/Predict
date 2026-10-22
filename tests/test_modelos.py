import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from analisador.modelos import garch, hmm
from analisador.modelos.base import SIGMA
from analisador.modelos.baselines import GarchFHS, MomentumIngenuo, PasseioAleatorio, TaxaBase
from analisador.modelos.ensemble import pesos_online
from analisador.modelos.lgbm import LGBMDirecao, LGBMQuantis
from analisador.modelos.montecarlo import monte_carlo
from analisador.validacao.walkforward import gerar_janelas
from tests.conftest import serie_sintetica

QUANTIS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


@pytest.fixture(scope="module")
def retornos():
    return np.log(serie_sintetica(4000, semente=7)["Close"]).diff().dropna()


# ---------------------------------------------------------------- GARCH

@pytest.mark.parametrize("tipo", ["GARCH", "GJR", "EGARCH"])
def test_recursao_propria_bate_com_arch(retornos, tipo):
    r_pct = 100 * retornos.to_numpy()
    p = garch.ajustar(r_pct, tipo)
    o = 0 if tipo == "GARCH" else 1
    vol = "EGARCH" if tipo == "EGARCH" else "GARCH"
    res = arch_model(r_pct, mean="Constant", vol=vol, p=1, o=o, q=1, dist="t", rescale=False).fit(disp="off")
    s2_arch = res.conditional_volatility**2          # σ²_t (informação até t-1)
    s2_meu = garch.variancia_condicional(r_pct, p)   # s2[t] = σ²_{t+1}
    assert np.allclose(s2_meu[300:-1], s2_arch[301:], rtol=1e-3)


def test_garch_recupera_parametros_simulados(retornos):
    p = garch.ajustar(100 * retornos.to_numpy(), "GARCH")
    assert abs(p.alpha - 0.08) < 0.04
    assert abs(p.beta - 0.87) < 0.06
    assert 3 < p.nu < 15  # a simulação usa t com 6 graus de liberdade


def test_garch_online_sem_look_ahead(cfg, retornos):
    c = cfg.garch
    cheio = garch.garch_online(retornos, c)
    corte = retornos.index[1800]
    truncado = garch.garch_online(retornos[retornos.index <= corte], c)
    pd.testing.assert_frame_equal(cheio.loc[truncado.index], truncado)


def test_sigma_horizonte_estrutura_a_termo(cfg, retornos):
    online = garch.garch_online(retornos, cfg.garch).dropna()
    s1 = garch.sigma_horizonte(online, 1)
    assert np.allclose(s1, np.sqrt(online["s2_1"]) / 100)
    s21, s63 = garch.sigma_horizonte(online, 21), garch.sigma_horizonte(online, 63)
    assert (s63 > s21).all() and (s21 > s1).all()
    # Fórmula fechada == soma explícita das variâncias futuras (caso GARCH/GJR)
    linha = online[online["egarch"] == 0].iloc[0]
    phi, lp, s2 = linha["persistencia"], linha["var_lp"], linha["s2_1"]
    soma = sum(lp + phi**k * (s2 - lp) for k in range(21))
    assert np.isclose(garch.sigma_horizonte(online.loc[[linha.name]], 21).iloc[0], np.sqrt(soma) / 100)


def test_media_abs_t_padronizada():
    rng = np.random.default_rng(0)
    for nu in (4.5, 8.0):
        z = rng.standard_t(nu, 2_000_000) * np.sqrt((nu - 2) / nu)
        assert abs(garch.media_abs_t(nu) - np.abs(z).mean()) < 2e-3
    assert np.isclose(garch.media_abs_t(1e6), np.sqrt(2 / np.pi))


def test_egarch_variancia_de_longo_prazo_bate_com_simulacao(retornos):
    """Simulando o EGARCH-t por muito tempo, a variância média deve ficar perto de var_longo_prazo
    (a menos do viés de Jensen, pequeno)."""
    p = garch.ajustar(100 * retornos.to_numpy(), "EGARCH")
    sims = garch.simular(p, p.var_longo_prazo, 3000, 400, np.random.default_rng(1))
    diarios = np.diff(np.concatenate([np.zeros((400, 1)), sims * 100], axis=1), axis=1)[:, 1000:]
    assert 0.6 < diarios.var() / p.var_longo_prazo < 1.6


def test_simulacao_tem_volatilidade_coerente(retornos):
    final = garch.ajustar_final(retornos, carregar())
    sims = garch.simular(final.params, final.s2_1, 63, 20000, np.random.default_rng(0))
    online_sigma = np.sqrt(sum(
        final.params.var_longo_prazo + final.params.persistencia**k * (final.s2_1 - final.params.var_longo_prazo)
        for k in range(63))) / 100
    assert abs(sims[:, -1].std() / online_sigma - 1) < 0.1


def carregar():
    from analisador.config import carregar_config
    return carregar_config().garch


def test_monte_carlo(retornos, cfg):
    final = garch.ajustar_final(retornos, cfg.garch)
    mc = monte_carlo(final, [5, 21, 63], n_sim=2000, bloco=21, semente=1)
    assert mc.n_trajetorias == 4000
    assert len(mc.leque[0.5]) == 63
    assert (mc.leque[0.95] >= mc.leque[0.05]).all()
    assert mc.leque[0.95][-1] - mc.leque[0.05][-1] > mc.leque[0.95][4] - mc.leque[0.05][4]
    for e in mc.eventos.values():
        assert 0 <= e["p_mais_10"] <= 1 and 0 <= e["p_menos_15"] <= 1


# ---------------------------------------------------------------- HMM

@pytest.fixture(scope="module")
def close_com_regimes():
    """Série com regimes alternados: alta calma / queda volátil."""
    rng = np.random.default_rng(11)
    blocos = []
    for i in range(12):
        alta = i % 2 == 0
        blocos.append(rng.normal(0.0012 if alta else -0.0015, 0.008 if alta else 0.025, 250))
    r = np.concatenate(blocos)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-01", periods=len(r)))


def test_hmm_filtrado_nao_muda_com_futuro(cfg, close_com_regimes):
    c = cfg.hmm
    cheio = hmm.hmm_online(close_com_regimes, c, 0)
    corte = close_com_regimes.index[2000]
    trunc = hmm.hmm_online(close_com_regimes[close_com_regimes.index <= corte], c, 0)
    pd.testing.assert_frame_equal(cheio.loc[trunc.index], trunc)


def test_hmm_filtrado_bate_com_hmmlearn_na_ultima_linha(cfg, close_com_regimes):
    """Na última observação, filtrada == suavizada; antes disso, a suavizada usa o futuro."""
    obs = hmm.observacoes(close_com_regimes).to_numpy()
    aj = hmm.ajustar(obs, cfg.hmm, 0)
    filtrada = aj.filtrar(obs)
    suavizada = aj.modelo.predict_proba(aj._padronizar(obs))[:, aj.ordem]
    assert np.allclose(filtrada[-1], suavizada[-1], atol=1e-6)
    assert not np.allclose(filtrada[:-50], suavizada[:-50], atol=1e-3)


def test_hmm_identifica_regimes(cfg, close_com_regimes):
    final = hmm.ajustar_final(close_com_regimes, cfg.hmm, 0)
    assert final.nomes[0] == "Queda" and final.nomes[-1] == "Alta"
    assert final.medias[0, 0] < 0 < final.medias[-1, 0]
    assert np.allclose(final.transicao.sum(axis=1), 1)
    assert np.diag(final.transicao).min() > 0.9


# ---------------------------------------------------------------- baselines

def _X(n, sigma=0.05, mom=None):
    return pd.DataFrame({SIGMA: np.full(n, sigma), "mom_12_1": np.zeros(n) if mom is None else mom})


def test_baselines_basicos():
    rng = np.random.default_rng(0)
    y = rng.normal(0.01, 0.05, 5000)
    X = _X(5000)
    tb = TaxaBase().fit(X, y)
    assert np.isclose(tb.predict_proba(X[:1])[0], np.mean(y > 0))
    rw = PasseioAleatorio().fit(X, y)
    assert np.isclose(rw.predict_proba(X[:1])[0], norm.cdf(0.01 / 0.05), atol=0.02)
    q = rw.predict_quantiles(X[:1], QUANTIS)[0]
    assert np.allclose(q, 0.01 + 0.05 * norm.ppf(QUANTIS), atol=0.005)
    g = GarchFHS().fit(X, y)
    alto, baixo = _X(1, sigma=0.20), _X(1, sigma=0.02)
    # Com drift positivo, vol maior aproxima P(alta) de 50%
    assert g.predict_proba(alto)[0] < g.predict_proba(baixo)[0]


def test_momentum_condicional():
    rng = np.random.default_rng(1)
    mom = rng.normal(size=4000)
    y = np.where(mom > 0, rng.normal(0.02, 0.05, 4000), rng.normal(-0.02, 0.05, 4000))
    X = _X(4000, mom=mom)
    m = MomentumIngenuo().fit(X, y)
    p = m.predict_proba(_X(2, mom=np.array([1.0, -1.0])))
    assert p[0] > 0.6 > 0.4 > p[1]


# ---------------------------------------------------------------- vazamento

def _walk_forward_auc(X: pd.DataFrame, y: np.ndarray, h: int, cfg) -> float:
    prev = np.full(len(X), np.nan)
    for j in gerar_janelas(len(X), h, 600, 200, embargo=h):
        m = LGBMDirecao(cfg.lgbm, h, 0).fit(X.iloc[:j.treino_fim], y[:j.treino_fim])
        prev[j.teste_ini:j.teste_fim] = m.predict_proba(X.iloc[j.teste_ini:j.teste_fim])
    ok = ~np.isnan(prev) & ~np.isnan(y)
    return roc_auc_score(y[ok] > 0, prev[ok])


@pytest.fixture(scope="module")
def mundo_ml():
    from analisador.features import alvo_log_retorno, construir_features
    px = serie_sintetica(2400, semente=21)
    X = construir_features(px, None, {})
    y = alvo_log_retorno(px["Close"], 5).reindex(X.index).to_numpy()
    return X, y


def test_vazamento_alvo_embaralhado_cai_para_o_acaso(cfg, mundo_ml):
    X, y = mundo_ml
    y_emb = y.copy()
    ok = ~np.isnan(y_emb)
    y_emb[ok] = np.random.default_rng(0).permutation(y_emb[ok])
    auc = _walk_forward_auc(X, y_emb, 5, cfg)
    assert abs(auc - 0.5) < 0.05, f"AUC {auc:.3f} com alvo embaralhado indica vazamento"


def test_controle_positivo_vazamento_plantado_e_detectado(cfg, mundo_ml):
    """Se uma feature contivesse o futuro, o AUC dispararia: prova que o teste acima tem poder."""
    X, y = mundo_ml
    X_vaz = X.assign(vazamento=y)
    assert _walk_forward_auc(X_vaz, y, 5, cfg) > 0.9


def test_lgbm_quantis_ordenados_e_escalados(cfg):
    rng = np.random.default_rng(3)
    n = 3000
    sig = rng.uniform(0.02, 0.08, n)
    X = pd.DataFrame({SIGMA: sig, "f1": rng.normal(size=n)})
    y = sig * rng.standard_normal(n)
    m = LGBMQuantis(cfg.lgbm, QUANTIS, 0).fit(X, y)
    Q = m.predict_quantiles(X)
    assert (np.diff(Q, axis=1) >= 0).all()
    cob = np.mean((y >= Q[:, 1]) & (y <= Q[:, 5]))
    assert 0.74 < cob < 0.86


# ---------------------------------------------------------------- ensemble

def test_pesos_favorecem_o_melhor_e_so_usam_o_realizado():
    n, h = 800, 10
    perdas = np.vstack([np.full(n, 0.60), np.full(n, 0.70)])
    w = pesos_online(perdas, h=h, janela=252, minimo=63)
    assert np.allclose(w[:h + 62], 0.5), "antes do mínimo realizado, pesos iguais"
    # n_ef = 252/10 = 25,2 e diferença de perda 0,1 -> razão exp(-2,52): ~92% no melhor
    assert w[-1, 0] > 0.9
    # Mudar perdas em linhas ainda NÃO realizadas não pode alterar o peso em t
    t = 500
    alterado = perdas.copy()
    alterado[0, t - h + 1:] = 99.0
    assert np.allclose(pesos_online(alterado, h, 252, 63)[t], w[t])
