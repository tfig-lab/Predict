"""Carregamento e validação do config.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parent.parent
CONFIG_PADRAO = RAIZ / "config.yaml"


@dataclass(frozen=True)
class Mercado:
    indice: str
    macro: list[str]
    bcb: bool
    caixa: str


@dataclass(frozen=True)
class CfgDados:
    inicio: str | None
    anos_maximos: int
    anos_minimos: int
    pregoes_minimos: int
    pasta_cache: str
    tolerancia_reajuste: float
    salto_sigma: float
    lacuna_dias_corridos: int
    defasagem_ipca_dias: int
    mercados: dict[str, Mercado]


@dataclass(frozen=True)
class CfgWalkForward:
    treino_minimo: int
    passo: int
    passo_rapido: int
    passo_quantis: int
    embargo: int | str
    reselecao_hiper: int

    def embargo_para(self, h: int) -> int:
        return h if self.embargo == "h" else int(self.embargo)


@dataclass(frozen=True)
class CfgGarch:
    variantes: list[str]
    distribuicao: str
    reajuste: int
    reespecificacao: int
    janela_minima: int


@dataclass(frozen=True)
class CfgHmm:
    estados: list[int]
    reajuste: int
    reespecificacao: int
    iteracoes: int
    inicializacoes: int
    janela_minima: int


@dataclass(frozen=True)
class CfgLgbm:
    grade: list[dict[str, Any]]
    comum: dict[str, Any]
    early_stopping: int
    fracao_validacao: float
    quantil: dict[str, Any]


@dataclass(frozen=True)
class CfgScore:
    pesos: dict[str, float]
    escala_bss: float
    escala_forca: float
    alfa: float
    teto_sem_significancia: float
    faixas: dict[str, float]

    def rotulo(self, score: float) -> str:
        rotulo = "Baixa"
        for nome, minimo in sorted(self.faixas.items(), key=lambda kv: kv[1]):
            if score >= minimo:
                rotulo = nome
        return rotulo


@dataclass(frozen=True)
class Config:
    semente: int
    n_jobs: int
    dados: CfgDados
    horizontes: list[int]
    nomes_horizontes: dict[int, str]
    quantis: list[float]
    walkforward: CfgWalkForward
    garch: CfgGarch
    hmm: CfgHmm
    lgbm: CfgLgbm
    montecarlo: dict[str, Any]
    ensemble: dict[str, Any]
    calibracao: dict[str, Any]
    conformal: dict[str, Any]
    score: CfgScore
    backtest: dict[str, Any]
    fundacionais: dict[str, Any]
    relatorio: dict[str, Any]
    rapido: bool = field(default=False)

    @property
    def passo(self) -> int:
        return self.walkforward.passo_rapido if self.rapido else self.walkforward.passo

    def nome_horizonte(self, h: int) -> str:
        return self.nomes_horizontes.get(h, f"{h} pregões")

    def com(self, **mudancas: Any) -> "Config":
        return replace(self, **mudancas)


class ErroConfig(ValueError):
    pass


def _validar(cfg: Config) -> None:
    if not cfg.horizontes or any(h <= 0 for h in cfg.horizontes):
        raise ErroConfig("horizontes devem ser inteiros positivos")
    if sorted(set(cfg.horizontes)) != list(cfg.horizontes):
        raise ErroConfig("horizontes devem estar em ordem crescente e sem repetição")
    q = cfg.quantis
    if any(not 0 < x < 1 for x in q) or sorted(q) != q:
        raise ErroConfig("quantis devem estar em (0, 1) e em ordem crescente")
    for necessario in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        if not any(abs(x - necessario) < 1e-9 for x in q):
            raise ErroConfig(f"quantil {necessario} é obrigatório (faixas de 50/80/90%)")
    if abs(sum(cfg.score.pesos.values()) - 1) > 1e-6:
        raise ErroConfig("pesos do score devem somar 1")
    if set(cfg.score.pesos) != {"desempenho", "forca", "concordancia", "estabilidade"}:
        raise ErroConfig("score.pesos precisa de desempenho, forca, concordancia e estabilidade")
    wf = cfg.walkforward
    if wf.passo <= 0 or wf.passo_rapido <= 0 or wf.treino_minimo < 250:
        raise ErroConfig("walkforward: passo > 0 e treino_minimo >= 250")
    if wf.embargo != "h" and int(wf.embargo) < 0:
        raise ErroConfig("walkforward.embargo deve ser 'h' ou inteiro >= 0")
    if not cfg.lgbm.grade:
        raise ErroConfig("lgbm.grade não pode ser vazia")
    if not 0 < cfg.backtest["limiar"] < 1:
        raise ErroConfig("backtest.limiar deve estar em (0, 1)")


def carregar_config(caminho: str | Path | None = None, **mudancas: Any) -> Config:
    bruto = yaml.safe_load(Path(caminho or CONFIG_PADRAO).read_text(encoding="utf-8"))
    d = bruto["dados"]
    mercados = {k: Mercado(**v) for k, v in d.pop("mercados").items()}
    cfg = Config(
        semente=int(bruto["semente"]),
        n_jobs=int(bruto["n_jobs"]),
        dados=CfgDados(**d, mercados=mercados),
        horizontes=[int(h) for h in bruto["horizontes"]],
        nomes_horizontes={int(k): v for k, v in bruto["nomes_horizontes"].items()},
        quantis=[float(x) for x in bruto["quantis"]],
        walkforward=CfgWalkForward(**bruto["walkforward"]),
        garch=CfgGarch(**bruto["garch"]),
        hmm=CfgHmm(**bruto["hmm"]),
        lgbm=CfgLgbm(**bruto["lgbm"]),
        montecarlo=bruto["montecarlo"],
        ensemble=bruto["ensemble"],
        calibracao=bruto["calibracao"],
        conformal=bruto["conformal"],
        score=CfgScore(**bruto["score"]),
        backtest=bruto["backtest"],
        fundacionais=bruto["fundacionais"],
        relatorio=bruto["relatorio"],
    )
    if mudancas:
        cfg = cfg.com(**mudancas)
    _validar(cfg)
    return cfg
