"""Linha de comando: python -m analisador PETR4 [opções]."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from analisador.config import RAIZ, carregar_config
from analisador.log import configurar_logging, obter_logger

log = obter_logger(__name__)


def _horizontes(texto: str) -> list[int]:
    try:
        hs = sorted({int(x) for x in texto.split(",") if x.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError("use inteiros separados por vírgula, ex.: 5,21,63")
    if not hs or hs[0] <= 0:
        raise argparse.ArgumentTypeError("horizontes devem ser positivos")
    return hs


def criar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analisador",
        description="Analisa uma ação: tendência por horizonte com grau de confiabilidade medido fora da amostra.",
    )
    p.add_argument("tickers", nargs="+", help="códigos (ex.: PETR4 VALE3 AAPL); B3 sem sufixo vira .SA")
    p.add_argument("--horizontes", type=_horizontes, help="pregões à frente, ex.: 5,21,63,126 (padrão: config.yaml)")
    p.add_argument("--relatorio", choices=["html", "terminal"], default="html",
                   help="html = terminal + arquivo HTML (padrão); terminal = só o texto")
    p.add_argument("--inicio", help="data inicial do histórico (AAAA-MM-DD); padrão: últimos anos_maximos")
    p.add_argument("--rapido", action="store_true", help="retreino menos frequente e sem modelos fundacionais")
    p.add_argument("--fundacionais", action="store_true", help="inclui Chronos-Bolt (requer o extra 'fundacionais')")
    p.add_argument("--offline", action="store_true", help="usa só o cache local, sem baixar dados")
    p.add_argument("--cache", type=Path, help="pasta do cache Parquet (padrão: cache/)")
    p.add_argument("--config", type=Path, help="caminho de um config.yaml alternativo")
    p.add_argument("--saida", type=Path, help="pasta dos relatórios (padrão: relatorios/)")
    p.add_argument("--abrir", action="store_true", help="abre o relatório HTML no navegador")
    p.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="nível de log no terminal")
    return p


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")
    args = criar_parser().parse_args(argv)
    configurar_logging(args.log, RAIZ / "logs" / "analisador.log")

    mudancas = {"rapido": args.rapido}
    cfg = carregar_config(args.config, **mudancas)
    if args.fundacionais and not args.rapido:
        cfg = cfg.com(fundacionais={**cfg.fundacionais, "ativo": True})

    # imports pesados só depois de validar os argumentos
    from analisador.pipeline import analisar
    from analisador.relatorio.html import salvar_html
    from analisador.relatorio.terminal import relatorio_terminal

    erros = 0
    for ticker in args.tickers:
        try:
            r = analisar(ticker, cfg, inicio=args.inicio, offline=args.offline, horizontes=args.horizontes,
                         pasta_cache=args.cache)
        except Exception as e:
            log.error(f"falha ao analisar {ticker}: {e}")
            erros += 1
            continue
        print(relatorio_terminal(r))
        if args.relatorio == "html":
            arq = salvar_html(r, args.saida or RAIZ / cfg.relatorio["pasta"])
            print(f"Relatório HTML: {arq}")
            if args.abrir:
                webbrowser.open(arq.resolve().as_uri())
    return 1 if erros else 0
