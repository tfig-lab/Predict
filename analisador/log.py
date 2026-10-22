"""Logging estruturado: chave=valor no terminal, JSON por linha no arquivo."""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CAMPOS_PADRAO = set(vars(logging.makeLogRecord({})))


def _extras(registro: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(registro).items() if k not in _CAMPOS_PADRAO and k != "message"}


class FormatoTerminal(logging.Formatter):
    def format(self, registro: logging.LogRecord) -> str:
        extras = " ".join(f"{k}={v}" for k, v in _extras(registro).items())
        nivel = "" if registro.levelno == logging.INFO else f"[{registro.levelname}] "
        return f"{nivel}{registro.getMessage()}" + (f"  ({extras})" if extras else "")


class FormatoJson(logging.Formatter):
    def format(self, registro: logging.LogRecord) -> str:
        return json.dumps({
            "ts": self.formatTime(registro, "%Y-%m-%dT%H:%M:%S"),
            "nivel": registro.levelname,
            "modulo": registro.name,
            "msg": registro.getMessage(),
            **_extras(registro),
        }, ensure_ascii=False, default=str)


def configurar_logging(nivel: str = "INFO", arquivo: str | Path | None = None) -> None:
    raiz = logging.getLogger("analisador")
    raiz.handlers.clear()
    raiz.setLevel(nivel)
    raiz.propagate = False
    terminal = logging.StreamHandler(sys.stderr)
    terminal.setFormatter(FormatoTerminal())
    raiz.addHandler(terminal)
    if arquivo:
        Path(arquivo).parent.mkdir(parents=True, exist_ok=True)
        arq = logging.FileHandler(arquivo, encoding="utf-8")
        arq.setFormatter(FormatoJson())
        arq.setLevel("DEBUG")
        raiz.addHandler(arq)


def obter_logger(nome: str) -> logging.Logger:
    return logging.getLogger(nome if nome.startswith("analisador") else f"analisador.{nome}")


def fixar_sementes(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente)
