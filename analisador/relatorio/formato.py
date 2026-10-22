"""Formatação numérica no padrão brasileiro (1.234,56)."""
from __future__ import annotations

import numpy as np


def num(x: float | None, casas: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x: float | None, casas: int = 1, sinal: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    s = num(100 * x, casas) + "%"
    return ("+" + s) if sinal and x > 0 else s


def inteiro(x: int) -> str:
    return num(float(x), 0)
