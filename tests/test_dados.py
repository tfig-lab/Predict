from datetime import date

import numpy as np
import pandas as pd
import pytest

from analisador.dados.bcb import ipca_12m_disponivel
from analisador.dados.cache import CacheParquet, obter_com_cache
from analisador.dados.validacao import validar_ohlcv
from analisador.dados.yahoo import mercado_do_ticker, normalizar_ticker
from tests.conftest import serie_sintetica


@pytest.mark.parametrize("entrada,esperado", [
    ("petr4", "PETR4.SA"), ("BOVA11", "BOVA11.SA"), ("PETR4.SA", "PETR4.SA"),
    ("AAPL", "AAPL"), ("^BVSP", "^BVSP"), ("BRL=X", "BRL=X"),
])
def test_normalizar_ticker(entrada, esperado):
    assert normalizar_ticker(entrada) == esperado


def test_mercado_do_ticker():
    assert mercado_do_ticker("VALE3.SA") == "B3"
    assert mercado_do_ticker("AAPL") == "EUA"


class FonteFalsa:
    """Simula o Yahoo: devolve a série 'verdadeira' até a data 'hoje', a partir de `inicio`."""

    def __init__(self, serie: pd.DataFrame):
        self.serie = serie
        self.ate = serie.index[-1]
        self.chamadas: list[date | None] = []

    def __call__(self, inicio):
        self.chamadas.append(inicio)
        s = self.serie[self.serie.index <= self.ate]
        return s if inicio is None else s[s.index >= pd.Timestamp(inicio)]


def test_cache_download_inicial_e_reuso_no_mesmo_dia(tmp_path):
    fonte = FonteFalsa(serie_sintetica(300))
    cache = CacheParquet(tmp_path)
    a = obter_com_cache(cache, "X", fonte, hoje=date(2026, 1, 5))
    b = obter_com_cache(cache, "X", fonte, hoje=date(2026, 1, 5))
    assert fonte.chamadas == [None]
    pd.testing.assert_frame_equal(a, b, check_freq=False)


def test_cache_incremental_baixa_so_o_trecho_novo(tmp_path):
    completa = serie_sintetica(300)
    fonte = FonteFalsa(completa)
    fonte.ate = completa.index[250]
    cache = CacheParquet(tmp_path)
    obter_com_cache(cache, "X", fonte, hoje=date(2026, 1, 5))
    fonte.ate = completa.index[-1]
    final = obter_com_cache(cache, "X", fonte, hoje=date(2026, 1, 6))
    assert fonte.chamadas[1] is not None, "a segunda chamada deve ser incremental"
    assert len(final) == 300
    pd.testing.assert_frame_equal(final, completa, check_freq=False)


def test_cache_detecta_reajuste_e_baixa_tudo(tmp_path):
    completa = serie_sintetica(300)
    fonte = FonteFalsa(completa)
    fonte.ate = completa.index[250]
    cache = CacheParquet(tmp_path)
    obter_com_cache(cache, "X", fonte, hoje=date(2026, 1, 5))
    # O Yahoo reajusta TODO o histórico por um dividendo de 2%
    fonte.serie = completa * 0.98
    fonte.ate = completa.index[-1]
    final = obter_com_cache(cache, "X", fonte, hoje=date(2026, 1, 6))
    assert fonte.chamadas[-1] is None, "reajuste deve forçar download completo"
    assert np.allclose(final["Close"].iloc[:250], completa["Close"].iloc[:250] * 0.98)


def test_cache_usa_dados_antigos_se_download_falhar(tmp_path):
    fonte = FonteFalsa(serie_sintetica(300))
    cache = CacheParquet(tmp_path)
    obter_com_cache(cache, "X", fonte, hoje=date(2026, 1, 5))

    def falha(_):
        raise ConnectionError("sem rede")

    df = obter_com_cache(cache, "X", falha, hoje=date(2026, 1, 6))
    assert len(df) == 300


def test_validacao_encontra_e_corrige_defeitos():
    df = serie_sintetica(600)
    df.iloc[100, df.columns.get_loc("Close")] = 0.0                   # preço zerado
    df.iloc[200, df.columns.get_loc("Close")] *= 3.0                  # pico isolado (erro)
    df.iloc[300:, :4] *= 0.5                                          # queda real e persistente de 50%
    df.iloc[400:450, df.columns.get_loc("Volume")] = 0                # volume nulo
    df = df.drop(df.index[500:510])                                   # lacuna de 2 semanas
    limpo, avisos = validar_ohlcv(df, "TESTE", salto_sigma=8, lacuna_dias=7)
    texto = " | ".join(avisos)
    assert "zerado" in texto
    assert "picos isolados" in texto
    assert "mantidos" in texto, "queda persistente é evento real e deve ser mantida"
    assert "volume nulo" in texto
    assert "lacunas" in texto
    assert (limpo["Close"] > 0).all()
    assert df.index[200] not in limpo.index
    assert df.index[300] in limpo.index


def test_preco_congelado_corta_o_historico():
    """Caso real (SUZB3 no Yahoo): anos de preço idêntico e volume zero antes do ticker existir."""
    df = serie_sintetica(1500)
    df.iloc[:800, df.columns.get_loc("Close")] = 10.0
    df.iloc[:800, df.columns.get_loc("Volume")] = 0
    limpo, avisos = validar_ohlcv(df, "TESTE")
    assert limpo.index[0] == df.index[800]
    assert any("congelado" in a for a in avisos)
    # Um feriado isolado com preço repetido não corta nada
    df2 = serie_sintetica(600)
    df2.iloc[300, df2.columns.get_loc("Close")] = df2["Close"].iloc[299]
    df2.iloc[300, df2.columns.get_loc("Volume")] = 0
    assert len(validar_ohlcv(df2, "TESTE")[0]) == 600


def test_ipca_indexado_pela_data_de_divulgacao():
    meses = pd.date_range("2020-01-01", periods=24, freq="MS")
    ipca = pd.Series(0.5, index=meses)
    disp = ipca_12m_disponivel(ipca, defasagem_dias=45)
    # O acumulado até dez/2020 (referência 01/12/2020) só pode aparecer em meados de jan/2021
    assert disp.index[0] == pd.Timestamp("2020-12-01") + pd.Timedelta(days=45)
    assert abs(disp.iloc[0] - (1.005**12 - 1)) < 1e-12
