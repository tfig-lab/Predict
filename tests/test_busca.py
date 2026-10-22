from analisador.dados.busca import filtrar_resultados

# Amostra real (resumida) do que o Yahoo devolve
COTACOES = [
    {"symbol": "PBR", "quoteType": "EQUITY", "exchange": "NYQ", "longname": "Petróleo Brasileiro S.A. - Petrobras"},
    {"symbol": "PETR4.SA", "quoteType": "EQUITY", "exchange": "SAO", "longname": "Petróleo Brasileiro S.A. - Petrobras"},
    {"symbol": "PETR4F.SA", "quoteType": "EQUITY", "exchange": "SAO", "shortname": "PETROBRAS PN"},
    {"symbol": "BBAS3Q.SA", "quoteType": "EQUITY", "exchange": "SAO", "shortname": "BRASIL ON NM"},
    {"symbol": "PBR.BA", "quoteType": "EQUITY", "exchange": "BUE", "longname": "Petrobras"},
    {"symbol": "SAAPL=F", "quoteType": "FUTURE", "exchange": "CME", "longname": "Apple futures"},
    {"symbol": "MGLUY", "quoteType": "EQUITY", "exchange": "PNK", "longname": "Magazine Luiza"},
    {"symbol": "AAPL.TO", "quoteType": "EQUITY", "exchange": "TOR", "longname": "Apple CDR"},
    {"symbol": "BOVA11.SA", "quoteType": "ETF", "exchange": "SAO", "longname": "iShares Ibovespa"},
    {"symbol": "PETR4.SA", "quoteType": "EQUITY", "exchange": "SAO", "longname": "duplicado"},
]


def test_filtra_so_b3_e_eua_sem_fracionario_nem_duplicata():
    r = filtrar_resultados(COTACOES)
    assert [x["simbolo"] for x in r] == ["PBR", "PETR4.SA", "BOVA11.SA"]
    assert r[1]["bolsa"] == "B3" and r[0]["bolsa"] == "EUA"
    assert r[1]["nome"].startswith("Petróleo")
