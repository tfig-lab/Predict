from pathlib import Path

import pytest
import yaml

from analisador.config import CONFIG_PADRAO, ErroConfig, carregar_config


def test_config_padrao_carrega():
    cfg = carregar_config()
    assert cfg.horizontes == [1, 5, 10, 21, 63, 126]
    assert cfg.walkforward.embargo_para(63) == 63
    assert cfg.dados.mercados["B3"].indice == "^BVSP"
    assert cfg.dados.mercados["EUA"].indice == "^GSPC"
    assert abs(sum(cfg.score.pesos.values()) - 1) < 1e-9


def test_rotulos_do_score():
    s = carregar_config().score
    assert s.rotulo(0) == "Baixa"
    assert s.rotulo(39.9) == "Baixa"
    assert s.rotulo(40) == "Média"
    assert s.rotulo(69) == "Média"
    assert s.rotulo(70) == "Alta"


def test_rapido_usa_passo_maior():
    cfg = carregar_config(rapido=True)
    assert cfg.passo == cfg.walkforward.passo_rapido


def _config_alterado(tmp_path: Path, alterar) -> Path:
    bruto = yaml.safe_load(CONFIG_PADRAO.read_text(encoding="utf-8"))
    alterar(bruto)
    destino = tmp_path / "config.yaml"
    destino.write_text(yaml.safe_dump(bruto, allow_unicode=True), encoding="utf-8")
    return destino


@pytest.mark.parametrize("alterar", [
    lambda b: b["score"]["pesos"].update(desempenho=0.9),
    lambda b: b.update(horizontes=[21, 5]),
    lambda b: b.update(quantis=[0.1, 0.5, 0.9]),
    lambda b: b["walkforward"].update(embargo=-1),
    lambda b: b["backtest"].update(limiar=1.5),
])
def test_config_invalido_e_rejeitado(tmp_path, alterar):
    with pytest.raises(ErroConfig):
        carregar_config(_config_alterado(tmp_path, alterar))
