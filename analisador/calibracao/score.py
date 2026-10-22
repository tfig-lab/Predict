"""Score de confiabilidade (0-100) por horizonte.

Componentes (pesos no config.yaml, score.pesos):
  (a) desempenho  - Brier skill score fora da amostra contra a taxa base (escala: score.escala_bss)
  (b) força       - |p calibrado - taxa base| (escala: score.escala_forca)
  (c) concordância- fração ponderada dos modelos do ensemble na mesma direção
  (d) estabilidade- persistência do regime HMM atual e vol atual vs. histórica

Regra de honestidade: sem vantagem estatística (Diebold-Mariano contra a taxa base, ou Brier
pior que algum baseline), o score fica limitado a `teto_sem_significancia` (faixa Baixa).
"""
from __future__ import annotations

import numpy as np

from analisador.config import CfgScore
from analisador.validacao.metricas import brier
from analisador.validacao.testes_estatisticos import diebold_mariano


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def comp_desempenho(bss: float, cfg: CfgScore) -> float:
    return float(_clip01(bss / cfg.escala_bss)) if np.isfinite(bss) else 0.0


def comp_forca(p: np.ndarray, base: np.ndarray, cfg: CfgScore) -> np.ndarray:
    return _clip01(np.abs(p - base) / cfg.escala_forca)


def comp_estabilidade(persistencia: np.ndarray, vol_razao: np.ndarray) -> np.ndarray:
    pers = _clip01((np.nan_to_num(persistencia, nan=0.5) - 0.5) / 0.5)
    vr = np.nan_to_num(vol_razao, nan=1.0)
    vol = _clip01(1 - np.abs(np.log(np.clip(vr, 1e-3, None))) / np.log(2))
    return 0.5 * pers + 0.5 * vol


def avaliar_vantagem(y: np.ndarray, p: np.ndarray, p_base: np.ndarray, outros_baselines: list[np.ndarray],
                     h: int, cfg: CfgScore) -> dict:
    """Há vantagem se o Brier do modelo for significativamente menor que o da taxa base (DM)
    e não for pior que nenhum outro baseline."""
    if len(y) < 30:
        return {"bss": float("nan"), "p_dm": 1.0, "vantagem": False}
    b_mod, b_base = brier(y, p), brier(y, p_base)
    bss = 1 - b_mod.mean() / b_base.mean() if b_base.mean() > 0 else float("nan")
    dm = diebold_mariano(b_mod, b_base, h)
    nao_pior = all(b_mod.mean() <= brier(y, ob).mean() for ob in outros_baselines)
    return {"bss": float(bss), "p_dm": dm["p_valor"], "vantagem": bool(dm["p_valor"] < cfg.alfa and bss > 0 and nao_pior)}


def combinar(componentes: dict[str, np.ndarray | float], vantagem: np.ndarray | bool, cfg: CfgScore) -> np.ndarray:
    total = sum(cfg.pesos[k] * np.asarray(v, dtype=float) for k, v in componentes.items())
    score = 100 * np.asarray(total, dtype=float)
    return np.where(vantagem, score, np.minimum(score, cfg.teto_sem_significancia))


def score_online(
    y_bin: np.ndarray,
    p_cal: np.ndarray,
    p_base: np.ndarray,
    outros_baselines: list[np.ndarray],
    concordancia: np.ndarray,
    estabilidade: np.ndarray,
    avaliavel: np.ndarray,
    h: int,
    passo: int,
    cfg: CfgScore,
    minimo_realizado: int = 252,
) -> dict[str, np.ndarray]:
    """Score como teria sido calculado em cada pregão, usando só o desempenho já realizado.
    Permite medir o acerto histórico por faixa de confiança."""
    n = len(p_cal)
    bss_t = np.full(n, np.nan)
    vant_t = np.zeros(n, dtype=bool)
    pdm_t = np.ones(n)
    for t0 in range(0, n, passo):
        fim = t0 - h + 1
        sel = np.zeros(n, dtype=bool)
        if fim > 0:
            sel[:fim] = avaliavel[:fim]
        if sel.sum() < minimo_realizado:
            continue
        r = avaliar_vantagem(y_bin[sel], p_cal[sel], p_base[sel], [b[sel] for b in outros_baselines], h, cfg)
        bss_t[t0:t0 + passo], pdm_t[t0:t0 + passo], vant_t[t0:t0 + passo] = r["bss"], r["p_dm"], r["vantagem"]
    comps = {
        "desempenho": np.array([comp_desempenho(b, cfg) for b in bss_t]),
        "forca": comp_forca(p_cal, p_base, cfg),
        "concordancia": np.asarray(concordancia, dtype=float),
        "estabilidade": np.asarray(estabilidade, dtype=float),
    }
    return {"score": combinar(comps, vant_t, cfg), "vantagem": vant_t, "bss": bss_t, "p_dm": pdm_t, **comps}


def acerto_por_faixa(score: np.ndarray, acerto: np.ndarray, mascara: np.ndarray, h: int, cfg: CfgScore) -> dict[str, dict]:
    """Taxa de acerto direcional histórica em cada faixa de confiança (Baixa/Média/Alta)."""
    rotulos = np.array([cfg.rotulo(s) for s in score])
    out = {}
    for faixa in cfg.faixas:
        sel = mascara & (rotulos == faixa)
        n = int(sel.sum())
        out[faixa] = {"n": n, "n_efetivo": n // max(h, 1), "acerto": float(acerto[sel].mean()) if n else float("nan")}
    return out
