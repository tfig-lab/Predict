import numpy as np
import pytest
from scipy.stats import norm

from analisador.modelos import fundacionais
from analisador.modelos.ensemble import elegibilidade_online, pesos_online

QUANTIS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def test_conversao_de_quantis_normal():
    """Quantis de uma N(0,01; 0,05) nos níveis do Chronos -> P(alta) e quantis pedidos corretos."""
    qr = np.tile(0.01 + 0.05 * norm.ppf(fundacionais.NIVEIS), (3, 1))
    p, Q = fundacionais.quantis_para_saidas(qr, QUANTIS)
    assert np.allclose(p, 1 - norm.cdf(0, 0.01, 0.05), atol=0.01)
    assert np.allclose(Q[0], 0.01 + 0.05 * norm.ppf(QUANTIS), atol=0.003)
    assert (np.diff(Q, axis=1) >= 0).all()


def test_fundacional_so_entra_quando_vence_os_baselines():
    n, h = 1200, 5
    perdas = np.vstack([np.full(n, 0.69), np.full(n, 0.69), np.full(n, 0.69), np.full(n, 0.69)])
    perdas[3, :600] = 0.75          # candidato pior que os baselines na 1ª metade
    perdas[3, 600:] = 0.60          # e melhor na 2ª metade
    mask = elegibilidade_online(perdas, 3, [0, 1, 2], h, janela=252, minimo=63)
    w = pesos_online(perdas, h, 252, 63, mask)
    assert (w[:600, 3] == 0).all(), "não pode ter peso enquanto perde para os baselines"
    assert w[-1, 3] > 0.5
    assert np.allclose(w.sum(axis=1), 1)


@pytest.mark.skipif(not fundacionais.disponivel(), reason="extra 'fundacionais' não instalado")
def test_chronos_bolt_zero_shot():
    rng = np.random.default_rng(0)
    precos = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
    m = fundacionais.ChronosBolt()
    p, Q = m.prever(precos, np.array([600, 700]), 21, QUANTIS)
    assert p.shape == (2,) and Q.shape == (2, len(QUANTIS))
    assert ((0 < p) & (p < 1)).all()
