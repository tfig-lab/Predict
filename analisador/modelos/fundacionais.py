"""Modelo fundacional de séries temporais (Chronos-Bolt, Amazon), zero-shot. OPCIONAL.

Instalação: uv sync --extra fundacionais  (PyTorch + chronos-forecasting, ~1-2 GB).
Só entra no ensemble enquanto vencer os baselines no walk-forward (ver ensemble.elegibilidade_online).

Atenção: o corpus de pré-treino do Chronos pode conter séries financeiras que se sobrepõem ao
período de teste, o que tornaria a avaliação otimista (contaminação). Por isso ele fica
desligado por padrão e é tratado apenas como comparação.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

NIVEIS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def disponivel() -> bool:
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def quantis_para_saidas(qr: np.ndarray, quantis: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """qr: (n, 9) quantis de log-retorno nos NIVEIS. Devolve (P(alta), quantis pedidos).

    Fora de [0,1; 0,9] extrapola com cauda normal ajustada pela dispersão P10-P90.
    """
    qr = np.sort(qr, axis=1)
    niveis = np.asarray(NIVEIS)
    sigma = (qr[:, -1] - qr[:, 0]) / (2 * norm.ppf(0.9))
    out = np.empty((len(qr), len(quantis)))
    for j, q in enumerate(quantis):
        if q < niveis[0]:
            out[:, j] = qr[:, 0] + sigma * (norm.ppf(q) - norm.ppf(niveis[0]))
        elif q > niveis[-1]:
            out[:, j] = qr[:, -1] + sigma * (norm.ppf(q) - norm.ppf(niveis[-1]))
        else:
            out[:, j] = [np.interp(q, niveis, linha) for linha in qr]
    # P(r > 0) = 1 - F(0), com F interpolada linearmente entre os níveis e cauda normal fora deles
    p = np.empty(len(qr))
    for i, linha in enumerate(qr):
        if 0 < linha[0]:
            f0 = norm.cdf(norm.ppf(niveis[0]) - linha[0] / max(sigma[i], 1e-9))
        elif 0 > linha[-1]:
            f0 = norm.cdf(norm.ppf(niveis[-1]) - linha[-1] / max(sigma[i], 1e-9))
        else:
            f0 = np.interp(0.0, linha, niveis)
        p[i] = 1 - f0
    return np.clip(p, 0.02, 0.98), np.sort(out, axis=1)


class ChronosBolt:
    nome = "fundacional"

    def __init__(self, modelo: str = "amazon/chronos-bolt-small", contexto: int = 512, lote: int = 128):
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as e:
            raise ImportError("modelos fundacionais exigem o extra: uv sync --extra fundacionais") from e
        self.torch = torch
        self.pipe = BaseChronosPipeline.from_pretrained(modelo, device_map="cpu", torch_dtype=torch.float32)
        self.contexto, self.lote = contexto, lote

    def prever(self, precos: np.ndarray, origens: np.ndarray, h: int, quantis: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Para cada origem o (índice em `precos`), usa só precos[..o] e prevê o log-retorno de o a o+h."""
        saidas = []
        for ini in range(0, len(origens), self.lote):
            bloco = origens[ini:ini + self.lote]
            ctx = [self.torch.tensor(precos[max(0, o - self.contexto + 1):o + 1], dtype=self.torch.float32) for o in bloco]
            q, _ = self.pipe.predict_quantiles(context=ctx, prediction_length=h, quantile_levels=NIVEIS)
            q = q[:, h - 1, :].numpy()
            saidas.append(np.log(np.maximum(q, 1e-9) / precos[bloco][:, None]))
        return quantis_para_saidas(np.vstack(saidas), quantis)
