import numpy as np
import pandas as pd
import pytest

from analisador.features import alvo_log_retorno, construir_features
from tests.conftest import serie_sintetica


@pytest.fixture
def mundo():
    px = serie_sintetica(1500, semente=1)
    indice = serie_sintetica(1500, semente=2)
    macro = {
        "BRL=X": serie_sintetica(1500, semente=3)["Close"],
        "^VIX": serie_sintetica(1500, semente=4)["Close"],
        "SELIC": pd.Series(np.repeat(np.linspace(14, 2, 15), 100), index=px.index),
    }
    return px, indice, macro


@pytest.mark.parametrize("corte", [400, 800, 1200, 1499])
def test_features_nao_usam_dados_futuros(mundo, corte):
    """Truncar os dados em t não pode mudar nenhuma feature nas linhas <= t."""
    px, indice, macro = mundo
    t = px.index[corte]
    cheio = construir_features(px, indice, macro)
    truncado = construir_features(px[px.index <= t], indice[indice.index <= t],
                                  {k: v[v.index <= t] for k, v in macro.items()})
    cols = cheio.columns.intersection(truncado.columns)
    assert len(cols) > 30
    pd.testing.assert_frame_equal(cheio.loc[truncado.index, cols], truncado[cols], check_freq=False)


def test_futuro_envenenado_nao_altera_passado(mundo):
    """Trocar todos os preços após t por lixo não pode alterar features até t."""
    px, indice, macro = mundo
    t = px.index[1000]
    envenenado = px.copy()
    envenenado.loc[envenenado.index > t, ["Open", "High", "Low", "Close"]] *= 7.0
    a = construir_features(px, indice, macro).loc[:t]
    b = construir_features(envenenado, indice, macro).loc[:t]
    pd.testing.assert_frame_equal(a, b)


def test_features_estacionarias_sem_niveis_de_preco(mundo):
    px, indice, macro = mundo
    f = construir_features(px, indice, macro)
    for nome in ("Close", "Open", "High", "Low", "Volume"):
        assert nome not in f.columns
    # Escalonar o preço (ex.: desdobramento) não muda nenhuma feature
    g = construir_features(px.assign(**{c: px[c] * 10 for c in ("Open", "High", "Low", "Close")}), indice, macro)
    pd.testing.assert_frame_equal(f, g, rtol=1e-8)


def test_alvo_olha_para_frente_e_termina_em_nan():
    close = pd.Series(np.exp(np.arange(10) * 0.01))
    y = alvo_log_retorno(close, 3)
    assert np.allclose(y.iloc[:7], 0.03)
    assert y.iloc[7:].isna().all()
