"""Visão de aderência: alinhamento previsto x realizado e ranking."""
import numpy as np
import pytest

from analisador import aderencia as ad
from analisador.config import carregar_config
from analisador.pipeline import analisar
from tests.test_cli import cache_sintetico  # noqa: F401  (fixture)


@pytest.fixture
def resultado(cache_sintetico):  # noqa: F811
    cfg = carregar_config(rapido=True)
    return analisar("TESTE3", cfg, offline=True, horizontes=[21], pasta_cache=str(cache_sintetico / "cache"))


def test_realizado_e_o_preco_na_data_alvo(resultado):
    h = resultado.horizontes[0]
    close = resultado.graficos["preco"]["close"]
    pvr = ad.previsto_vs_realizado(h, "passeio_aleatorio")
    comuns = pvr.index.intersection(close.index)
    assert len(comuns) > 100
    assert np.allclose(pvr.loc[comuns, "realizado"], close.loc[comuns])
    # A data-alvo fica exatamente 21 pregões depois da data em que a previsão foi feita
    pos = {d: i for i, d in enumerate(h.serie_oos.index)}
    amostra = pvr.iloc[::50]
    for alvo, origem in zip(amostra.index, amostra["data_previsao"]):
        assert h.serie_oos.index[pos[origem] + 21] == alvo


def test_ranking_ordenado_com_todos_os_modelos(resultado):
    tab = ad.tabela_aderencia(resultado.horizontes[0])
    assert set(tab["chave"]) == {"taxa_base", "passeio_aleatorio", "momentum", "garch", "lgbm", "ensemble"}
    assert tab["erro_medio"].is_monotonic_increasing
    assert np.isnan(tab.set_index("chave").loc["passeio_aleatorio", "p_vs_random_walk"])
    assert tab["dentro_da_faixa"].between(0, 1).all()


def test_projecao_parte_do_preco_atual(resultado):
    proj = ad.projecao(resultado, "ensemble")
    assert proj.iloc[0]["previsto"] == resultado.preco_atual
    assert len(proj) == 2
    assert proj.iloc[1]["faixa_min"] < proj.iloc[1]["previsto"] < proj.iloc[1]["faixa_max"]
