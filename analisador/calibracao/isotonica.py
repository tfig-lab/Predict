"""Calibração das probabilidades, ajustada só em previsões fora da amostra já realizadas."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

LIMITES = (0.02, 0.98)


class Calibrador:
    """Isotônica com amostra EFETIVA suficiente; Platt (logística no logit) caso contrário.

    Com rótulos de horizonte h sobrepostos, n linhas valem ~n/h observações independentes;
    a isotônica (não paramétrica) sobreajusta se esse número for pequeno.
    """

    def __init__(self, minimo_isotonica: int, h: int = 1):
        self.minimo_isotonica = minimo_isotonica
        self.h = max(h, 1)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrador":
        p = np.clip(p, 1e-4, 1 - 1e-4)
        if len(np.unique(y)) < 2:
            self.const = float(np.mean(y))
            return self
        self.const = None
        if len(p) / self.h >= self.minimo_isotonica:
            self.iso = IsotonicRegression(y_min=LIMITES[0], y_max=LIMITES[1], out_of_bounds="clip").fit(p, y)
            self.platt = None
        else:
            self.iso = None
            self.platt = LogisticRegression(C=1.0).fit(np.log(p / (1 - p))[:, None], y)
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self.const is not None:
            return np.full(len(p), self.const)
        p = np.clip(p, 1e-4, 1 - 1e-4)
        out = self.iso.predict(p) if self.iso is not None else self.platt.predict_proba(np.log(p / (1 - p))[:, None])[:, 1]
        return np.clip(out, *LIMITES)


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def calibrar_online(
    p_bruto: np.ndarray,
    y_bin: np.ndarray,
    h: int,
    passo: int,
    aquecimento: int,
    minimo_isotonica: int,
    n0_encolhimento: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Recalibra a cada `passo` linhas com os pares (p, y) cujos rótulos já se realizaram
    (linhas s <= t - h). Antes do aquecimento devolve a probabilidade bruta.

    Duas salvaguardas contra calibrar ruído (calibrar um preditor quase constante, ou com
    relação instável no tempo, só adiciona variância):
      1. encolhimento para a identidade: logit(p) + κ·(logit(cal(p)) - logit(p)),
         κ = n_ef / (n_ef + n0), n_ef = linhas realizadas / h;
      2. seleção prequencial: a proposta calibrada é calculada em todo bloco, mas só é USADA
         se, em todo o histórico já realizado, teve log loss menor que a bruta.

    Retorna (p_final, mascara_avaliavel), máscara = linhas após o aquecimento.
    """
    n = len(p_bruto)
    proposta = np.full(n, np.nan)
    p_final = p_bruto.astype(float).copy()
    avaliavel = np.zeros(n, dtype=bool)
    for t0 in range(0, n, passo):
        fim = t0 - h + 1
        if fim < aquecimento:
            continue
        avaliavel[t0:t0 + passo] = True
        cal = Calibrador(minimo_isotonica, h).fit(p_bruto[:fim], y_bin[:fim])
        bloco = p_bruto[t0:t0 + passo]
        n_ef = fim / max(h, 1)
        kappa = n_ef / (n_ef + n0_encolhimento)
        z = _logit(bloco) + kappa * (_logit(cal(bloco)) - _logit(bloco))
        proposta[t0:t0 + passo] = 1 / (1 + np.exp(-z))
        historico = np.isfinite(proposta[:fim])
        if historico.sum() >= aquecimento:
            yh = y_bin[:fim][historico]
            if _log_loss(proposta[:fim][historico], yh) < _log_loss(p_bruto[:fim][historico], yh):
                p_final[t0:t0 + passo] = proposta[t0:t0 + passo]
    return p_final, avaliavel
