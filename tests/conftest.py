import numpy as np
import pandas as pd
import pytest

from analisador.config import carregar_config


def serie_sintetica(n: int = 2000, semente: int = 0, inicio: str = "2012-01-02", vol: float = 0.02,
                    drift: float = 0.0003, garch: bool = True) -> pd.DataFrame:
    """OHLCV sintético com volatilidade em clusters (GARCH(1,1)) e caudas t."""
    rng = np.random.default_rng(semente)
    idx = pd.bdate_range(inicio, periods=n)
    omega, alpha, beta = vol**2 * 0.05, 0.08, 0.87
    s2 = vol**2
    r = np.empty(n)
    for i in range(n):
        z = rng.standard_t(6) / np.sqrt(6 / 4)
        r[i] = drift + np.sqrt(s2) * z
        s2 = omega + alpha * (r[i] - drift) ** 2 + beta * s2 if garch else vol**2
    close = 50 * np.exp(np.cumsum(r))
    abertura = close * np.exp(rng.normal(0, vol / 4, n))
    alta = np.maximum(close, abertura) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    baixa = np.minimum(close, abertura) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.lognormal(14, 0.4, n)
    return pd.DataFrame({"Open": abertura, "High": alta, "Low": baixa, "Close": close, "Volume": volume}, index=idx)


@pytest.fixture
def cfg():
    return carregar_config()


@pytest.fixture
def ohlcv():
    return serie_sintetica()
