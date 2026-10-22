import numpy as np
from scipy.stats import norm

from analisador.calibracao.conformal import conformal_online
from analisador.calibracao.isotonica import calibrar_online
from analisador.calibracao.score import acerto_por_faixa, avaliar_vantagem, combinar, score_online
from analisador.validacao.metricas import brier

QUANTIS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def test_calibracao_corrige_probabilidades_exageradas():
    rng = np.random.default_rng(0)
    n = 4000
    verdadeira = rng.uniform(0.35, 0.65, n)
    y = (rng.uniform(size=n) < verdadeira).astype(float)
    exagerada = np.clip(0.5 + 3 * (verdadeira - 0.5), 0.01, 0.99)   # modelo superconfiante
    p_cal, ok = calibrar_online(exagerada, y, h=5, passo=21, aquecimento=500, minimo_isotonica=300)
    assert ok[1000:].all() and not ok[:400].any()
    assert brier(y[ok], p_cal[ok]).mean() < brier(y[ok], exagerada[ok]).mean()


def test_calibracao_nao_amplifica_ruido_de_preditor_quase_constante():
    """Preditor sem informação (base ± ruído minúsculo): calibrar só adicionaria variância."""
    rng = np.random.default_rng(7)
    n, h = 3000, 21
    y = (rng.uniform(size=n) < 0.55).astype(float)
    quase_constante = 0.55 + rng.normal(0, 0.003, n)
    p_cal, ok = calibrar_online(quase_constante, y, h, 21, 252, 300)
    assert brier(y[ok], p_cal[ok]).mean() <= brier(y[ok], quase_constante[ok]).mean() + 1e-4


def test_calibracao_nao_usa_rotulos_futuros():
    rng = np.random.default_rng(1)
    n, h, t = 2000, 21, 1500
    p = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < p).astype(float)
    a, _ = calibrar_online(p, y, h, 21, 300, 300)
    y2 = y.copy()
    y2[t - h + 1:] = 1 - y2[t - h + 1:]                            # inverte tudo que não estava realizado em t
    b, _ = calibrar_online(p, y2, h, 21, 300, 300)
    bloco = (t // 21) * 21                                          # início do bloco que contém t
    assert np.allclose(a[:bloco + 21][:t + 1], b[:bloco + 21][:t + 1])


def test_conformal_mantem_cobertura_apos_quebra_de_regime():
    """Os quantis brutos supõem vol constante; no meio da série a vol dobra.
    O ACI deve recuperar a cobertura nominal."""
    rng = np.random.default_rng(2)
    n, h = 4000, 5
    vol = np.where(np.arange(n) < n // 2, 0.02, 0.04)
    y = vol * rng.standard_normal(n)
    Q = np.tile(0.02 * norm.ppf(QUANTIS), (n, 1))
    Qa, _ = conformal_online(Q, y, QUANTIS, h=h, gamma=0.005, janela=504)
    seg = slice(n // 2 + 600, n)
    bruto_80 = np.mean((y[seg] >= Q[seg, 1]) & (y[seg] <= Q[seg, 5]))
    adj_80 = np.mean((y[seg] >= Qa[seg, 1]) & (y[seg] <= Qa[seg, 5]))
    adj_90 = np.mean((y[seg] >= Qa[seg, 0]) & (y[seg] <= Qa[seg, 6]))
    assert bruto_80 < 0.6
    assert abs(adj_80 - 0.80) < 0.04
    assert abs(adj_90 - 0.90) < 0.04
    total = np.mean((y[600:] >= Qa[600:, 1]) & (y[600:] <= Qa[600:, 5]))
    assert abs(total - 0.80) < 0.03


def test_conformal_nao_usa_o_futuro():
    rng = np.random.default_rng(3)
    n, h, t = 1500, 10, 1000
    y = rng.normal(0, 0.03, n)
    Q = np.tile(0.02 * norm.ppf(QUANTIS), (n, 1))
    a, _ = conformal_online(Q, y, QUANTIS, h, 0.005, 504)
    y2 = y.copy()
    y2[t - h + 1:] *= 10
    b, _ = conformal_online(Q, y2, QUANTIS, h, 0.005, 504)
    assert np.allclose(a[:t + 1], b[:t + 1])


def test_teto_sem_significancia(cfg):
    comps = {"desempenho": 1.0, "forca": 1.0, "concordancia": 1.0, "estabilidade": 1.0}
    assert combinar(comps, True, cfg.score) == 100
    assert combinar(comps, False, cfg.score) == cfg.score.teto_sem_significancia
    assert cfg.score.rotulo(float(combinar(comps, False, cfg.score))) == "Baixa"


def test_sem_vantagem_quando_modelo_e_ruido(cfg):
    rng = np.random.default_rng(4)
    n = 3000
    y = (rng.uniform(size=n) < 0.55).astype(float)
    ruido = np.clip(0.55 + rng.normal(0, 0.05, n), 0, 1)
    r = avaliar_vantagem(y, ruido, np.full(n, 0.55), [], h=1, cfg=cfg.score)
    assert not r["vantagem"] and r["bss"] < 0


def test_vantagem_quando_ha_sinal_real(cfg):
    rng = np.random.default_rng(5)
    n = 3000
    verdadeira = rng.uniform(0.3, 0.8, n)
    y = (rng.uniform(size=n) < verdadeira).astype(float)
    r = avaliar_vantagem(y, verdadeira, np.full(n, y.mean()), [np.full(n, 0.5)], h=1, cfg=cfg.score)
    assert r["vantagem"] and r["bss"] > 0.05


def test_score_online_e_acerto_por_faixa(cfg):
    rng = np.random.default_rng(6)
    n, h = 3000, 5
    verdadeira = rng.uniform(0.3, 0.8, n)
    y = (rng.uniform(size=n) < verdadeira).astype(float)
    base = np.full(n, 0.55)
    s = score_online(y, verdadeira, base, [], np.ones(n), np.full(n, 0.8), np.ones(n, bool), h, 21, cfg.score)
    assert np.isnan(s["bss"][:200]).all(), "sem histórico realizado, sem desempenho comprovado"
    assert s["vantagem"][-1]
    assert (s["score"][:200] <= cfg.score.teto_sem_significancia).all()
    faixas = acerto_por_faixa(s["score"], (verdadeira >= 0.5) == (y == 1), np.ones(n, bool), h, cfg.score)
    assert sum(f["n"] for f in faixas.values()) == n
