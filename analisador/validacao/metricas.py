"""Métricas de direção, quantis e ranking."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

EPS = 1e-6


def brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def log_loss_pontual(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pinball_pontual(y: np.ndarray, Q: np.ndarray, quantis: list[float]) -> np.ndarray:
    """Perda pinball média entre os quantis, por observação. Q tem forma (n, len(quantis))."""
    q = np.asarray(quantis)[None, :]
    erro = y[:, None] - Q
    return np.mean(np.maximum(q * erro, (q - 1) * erro), axis=1)


def metricas_direcao(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray | None = None) -> dict[str, float]:
    """y binário (1 = alta). p_ref = previsão da taxa base, para o Brier skill score."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(float)
    out = {
        "n": int(len(y)),
        "acuracia": float(np.mean(pred == y)),
        "acuracia_balanceada": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and np.ptp(p) > 0 else 0.5,
        "brier": float(np.mean(brier(y, p))),
        "log_loss": float(np.mean(log_loss_pontual(y, p))),
    }
    if p_ref is not None:
        b_ref = float(np.mean(brier(y, p_ref)))
        out["bss"] = 1 - out["brier"] / b_ref if b_ref > 0 else float("nan")
    return out


def cobertura(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def metricas_quantis(y: np.ndarray, Q: np.ndarray, quantis: list[float]) -> dict[str, float]:
    qi = {round(q, 2): i for i, q in enumerate(quantis)}
    return {
        "pinball": float(np.mean(pinball_pontual(y, Q, quantis))),
        "cobertura_50": cobertura(y, Q[:, qi[0.25]], Q[:, qi[0.75]]),
        "cobertura_80": cobertura(y, Q[:, qi[0.1]], Q[:, qi[0.9]]),
        "cobertura_90": cobertura(y, Q[:, qi[0.05]], Q[:, qi[0.95]]),
        "largura_80": float(np.mean(Q[:, qi[0.9]] - Q[:, qi[0.1]])),
    }


def ic_spearman(previsao: np.ndarray, realizado: np.ndarray) -> float:
    if np.ptp(previsao) == 0:
        return 0.0
    return float(spearmanr(previsao, realizado).statistic)
