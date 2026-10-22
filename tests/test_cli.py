"""Ponta a ponta: CLI com dados em cache (sem rede)."""
from datetime import date

import pandas as pd
import pytest

from analisador.cli import main
from analisador.dados.cache import CacheParquet
from tests.conftest import serie_sintetica


@pytest.fixture
def cache_sintetico(tmp_path):
    cache = CacheParquet(tmp_path / "cache")
    hoje = date.today()
    cache.gravar("TESTE3.SA", serie_sintetica(1900, semente=31, inicio="2018-01-02"), hoje)
    cache.gravar("^BVSP", serie_sintetica(1900, semente=32, inicio="2018-01-02"), hoje)
    cache.gravar("BRL=X", serie_sintetica(1900, semente=33, inicio="2018-01-02"), hoje)
    selic = pd.DataFrame({"Close": 10.0}, index=pd.bdate_range("2017-01-02", periods=2400))
    cache.gravar("SGS_432", selic, hoje)
    return tmp_path


def test_cli_ponta_a_ponta_offline(cache_sintetico, capsys):
    saida = cache_sintetico / "relatorios"
    codigo = main(["TESTE3", "--offline", "--rapido", "--horizontes", "5,21",
                   "--cache", str(cache_sintetico / "cache"), "--saida", str(saida), "--log", "WARNING"])
    texto = capsys.readouterr().out
    assert codigo == 0
    assert "TESTE3.SA" in texto
    assert "1 semana" in texto and "1 mês" in texto
    assert "Validação walk-forward" in texto
    # Série sintética = passeio aleatório: não pode haver vantagem direcional
    assert "Sem vantagem estatística" in texto
    htmls = list(saida.glob("TESTE3.SA_*.html"))
    assert len(htmls) == 1
    html = htmls[0].read_text(encoding="utf-8")
    assert "Plotly" in html and '"ticker": "TESTE3.SA"' in html
    assert "__DADOS__" not in html


def test_cli_ticker_inexistente_retorna_erro(tmp_path, capsys):
    codigo = main(["NAOEXISTE9", "--offline", "--cache", str(tmp_path), "--log", "ERROR"])
    assert codigo == 1
