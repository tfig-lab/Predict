"""Walk-forward completo de UM horizonte: modelos, ensemble, calibração, conformal,
métricas contra baselines, testes de significância, score histórico, backtest e SHAP."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from analisador.calibracao.conformal import conformal_online
from analisador.calibracao.isotonica import calibrar_online
from analisador.calibracao.score import (acerto_por_faixa, avaliar_vantagem, combinar, comp_desempenho,
                                         comp_estabilidade, comp_forca, score_online)
from analisador.config import Config
from analisador.features import alvo_log_retorno, alvo_relativo
from analisador.log import obter_logger
from analisador.modelos.base import SIGMA
from analisador.modelos.baselines import GarchFHS, MomentumIngenuo, PasseioAleatorio, TaxaBase
from analisador.modelos.garch import sigma_horizonte
from analisador.modelos.ensemble import (combinar_probabilidades, combinar_quantis, concordancia,
                                         elegibilidade_online, pesos_online)
from analisador.modelos.lgbm import LGBMDirecao, LGBMQuantis, selecionar_hiper
from analisador.tipos import SEM_VANTAGEM, PrevisaoHoje, ResultadoHorizonte
from analisador.validacao.backtest import backtest_limiar
from analisador.validacao.metricas import (ic_spearman, log_loss_pontual, metricas_direcao, metricas_quantis,
                                           pinball_pontual)
from analisador.validacao.testes_estatisticos import binomial_efetivo, diebold_mariano
from analisador.validacao.walkforward import gerar_janelas

log = obter_logger(__name__)
NOMES_MODELOS = {
    "taxa_base": "Taxa base", "passeio_aleatorio": "Random walk c/ drift", "momentum": "Momentum 12-1",
    "garch": "GARCH-FHS", "lgbm": "LightGBM", "fundacional": "Chronos-Bolt",
    "ensemble_bruto": "Ensemble (bruto)", "ensemble": "Ensemble calibrado", "ensemble_conformal": "Ensemble + conformal",
}
def _trilha_probabilidades(P: np.ndarray, y: np.ndarray, h: int, passo: int, cfg: Config, candidato: int | None = None):
    """Ensemble online + calibração online para uma matriz de probabilidades (M, n).
    `candidato`: índice de um modelo que só entra enquanto vencer os baselines (0, 1, 2)."""
    realizado = np.isfinite(y)
    y_bin = np.where(realizado, (y > 0).astype(float), np.nan)
    perdas = np.where(realizado[None, :], log_loss_pontual(np.nan_to_num(y_bin)[None, :], P), 0.0)
    ens = cfg.ensemble
    elegivel = None if candidato is None else elegibilidade_online(
        perdas, candidato, [0, 1, 2], h, ens["janela_perda"], ens["minimo_realizado"])
    pesos = pesos_online(perdas, h, ens["janela_perda"], ens["minimo_realizado"], elegivel)
    p_bruto = combinar_probabilidades(P, pesos)
    cal = cfg.calibracao
    p_cal, calibrado = calibrar_online(p_bruto, np.nan_to_num(y_bin), h, passo, cal["aquecimento"],
                                       cal["minimo_isotonica"], cal["n0_encolhimento"])
    return p_bruto, p_cal, calibrado, pesos, y_bin


def _curva_calibracao(p: np.ndarray, y: np.ndarray, minimo: int) -> list[dict[str, float]]:
    bins = np.clip((p * 20).astype(int), 0, 19)
    out = []
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() >= minimo:
            out.append({"previsto": float(p[sel].mean()), "realizado": float(y[sel].mean()), "n": int(sel.sum())})
    return out


def avaliar_horizonte(
    h: int,
    X: pd.DataFrame,
    close: pd.Series,
    indice_close: pd.Series | None,
    caixa: pd.Series,
    cfg: Config,
) -> ResultadoHorizonte:
    warnings.filterwarnings("ignore")
    t_ini = time.time()
    q = cfg.quantis
    passo = cfg.passo
    wf = cfg.walkforward
    semente = cfg.semente

    # ------------------------------------------------------------ dados do horizonte
    close_x = close.reindex(X.index)
    r_diario = np.log(close).diff()
    online = X[["_s2_1", "_persistencia", "_var_lp", "_egarch"]].rename(columns=lambda c: c[1:])
    # Antes do primeiro ajuste GARCH, a vol realizada de 21 dias faz o papel de σ
    sig = sigma_horizonte(online, h).fillna(X["vol_21"] / np.sqrt(252) * np.sqrt(h))
    Xh = X.drop(columns=[c for c in X.columns if c.startswith("_")]).copy()
    Xh["garch_vol_h"] = sig * np.sqrt(252 / h)
    Xh[SIGMA] = sig
    y = alvo_log_retorno(close_x, h).to_numpy()
    y_rel_s = alvo_relativo(close_x, indice_close, h) if indice_close is not None else None
    y_rel = None if y_rel_s is None else y_rel_s.to_numpy()
    n = len(Xh)
    n_rot = n - h

    janelas = gerar_janelas(n, h, wf.treino_minimo, passo, wf.embargo_para(h))
    if not janelas:
        raise ValueError(f"histórico insuficiente para o horizonte de {h} pregões")
    oos = janelas[0].teste_ini

    nomes_dir = ["taxa_base", "passeio_aleatorio", "momentum", "garch", "lgbm"]
    nomes_q = ["taxa_base", "passeio_aleatorio", "momentum", "garch", "lgbm"]
    P = np.full((len(nomes_dir), n), np.nan)
    Q = np.full((len(nomes_q), n, len(q)), np.nan)
    P_rel = np.full((2, n), np.nan)

    hiper, ult_hiper = None, -10**9
    hiper_rel, ult_hiper_rel = None, -10**9
    mod_q, ult_q = None, -10**9
    mod_rel = None

    def treinar_baselines(Xtr, ytr):
        return [TaxaBase().fit(Xtr, ytr), PasseioAleatorio().fit(Xtr, ytr), MomentumIngenuo().fit(Xtr, ytr),
                GarchFHS().fit(Xtr, ytr)]

    for j in janelas:
        tr, te = slice(0, j.treino_fim), slice(j.teste_ini, j.teste_fim)
        Xtr, ytr, Xte = Xh.iloc[tr], y[tr], Xh.iloc[te]
        base = treinar_baselines(Xtr, ytr)
        for i, m in enumerate(base):
            P[i, te] = m.predict_proba(Xte)
            Q[i, te] = m.predict_quantiles(Xte, q)
        if hiper is None or j.teste_ini - ult_hiper >= wf.reselecao_hiper:
            hiper, ult_hiper = selecionar_hiper(Xtr, (ytr > 0).astype(int), h, cfg.lgbm, semente), j.teste_ini
        P[4, te] = LGBMDirecao(cfg.lgbm, h, semente, hiper).fit(Xtr, ytr).predict_proba(Xte)
        if mod_q is None or j.teste_ini - ult_q >= wf.passo_quantis:
            mod_q = LGBMQuantis(cfg.lgbm, q, semente).fit(Xtr, ytr)
            if y_rel is not None:
                yr = y_rel[tr]
                okr = np.isfinite(yr)
                if hiper_rel is None or j.teste_ini - ult_hiper_rel >= wf.reselecao_hiper:
                    hiper_rel = selecionar_hiper(Xtr[okr], (yr[okr] > 0).astype(int), h, cfg.lgbm, semente)
                    ult_hiper_rel = j.teste_ini
                mod_rel = (TaxaBase().fit(Xtr[okr], yr[okr]),
                           LGBMDirecao(cfg.lgbm, h, semente, hiper_rel).fit(Xtr[okr], yr[okr]))
            ult_q = j.teste_ini
        Q[4, te] = mod_q.predict_quantiles(Xte)
        if mod_rel is not None:
            P_rel[0, te] = mod_rel[0].predict_proba(Xte)
            P_rel[1, te] = mod_rel[1].predict_proba(Xte)

    # ------------------------------------------------------------ modelos finais -> previsão de hoje
    tr_final = slice(0, n_rot)
    Xf, yf, X_hoje = Xh.iloc[tr_final], y[tr_final], Xh.iloc[[n - 1]]
    for i, m in enumerate(treinar_baselines(Xf, yf)):
        P[i, n - 1] = m.predict_proba(X_hoje)[0]
        Q[i, n - 1] = m.predict_quantiles(X_hoje, q)[0]
    lgbm_final = LGBMDirecao(cfg.lgbm, h, semente, hiper).fit(Xf, yf)
    P[4, n - 1] = lgbm_final.predict_proba(X_hoje)[0]
    Q[4, n - 1] = LGBMQuantis(cfg.lgbm, q, semente).fit(Xf, yf).predict_quantiles(X_hoje)[0]
    if y_rel is not None:
        yr = y_rel[tr_final]
        okr = np.isfinite(yr)
        P_rel[0, n - 1] = TaxaBase().fit(Xf[okr], yr[okr]).predict_proba(X_hoje)[0]
        P_rel[1, n - 1] = LGBMDirecao(cfg.lgbm, h, semente, hiper_rel).fit(Xf[okr], yr[okr]).predict_proba(X_hoje)[0]

    # ------------------------------------------------------------ fora da amostra: ensemble, calibração, conformal
    sl = slice(oos, n)
    idx = Xh.index[sl]
    Po, Qo, yo = P[:, sl], Q[:, sl, :], y[sl]
    candidato = None
    if cfg.fundacionais.get("ativo") and not cfg.rapido:
        fund = _fundacional(close, Xh.index[sl], h, q, cfg)
        if fund is not None:
            Po, Qo = np.vstack([Po, fund[0][None]]), np.concatenate([Qo, fund[1][None]])
            nomes_dir, nomes_q = nomes_dir + ["fundacional"], nomes_q + ["fundacional"]
            candidato = len(nomes_dir) - 1
    p_bruto, p_cal, calibrado, pesos, y_bin = _trilha_probabilidades(Po, yo, h, passo, cfg, candidato)
    realizado = np.isfinite(yo)
    perdas_q = np.where(realizado[None, :], np.stack([pinball_pontual(np.nan_to_num(yo), Qo[m], q) for m in range(len(nomes_q))]), 0.0)
    ens = cfg.ensemble
    eleg_q = None if candidato is None else elegibilidade_online(
        perdas_q, candidato, [0, 1, 2], h, ens["janela_perda"], ens["minimo_realizado"])
    pesos_q = pesos_online(perdas_q, h, ens["janela_perda"], ens["minimo_realizado"], eleg_q)
    Q_ens = combinar_quantis(Qo, pesos_q)
    janela_conf = max(cfg.conformal["janela"], cfg.conformal["janela_efetiva"] * h)
    Q_conf, _ = conformal_online(Q_ens, yo, q, h, cfg.conformal["gamma_aci"], janela_conf)
    conc = concordancia(Po, pesos, p_bruto)
    estab = comp_estabilidade(X["hmm_persistencia"].to_numpy()[sl], X["garch_vol_razao"].to_numpy()[sl])
    av = calibrado & realizado

    # ------------------------------------------------------------ métricas (mesmas linhas para todos)
    yb, ya = y_bin[av], yo[av]
    p_ref = Po[0, av]
    met_dir = {nm: metricas_direcao(yb, Po[i, av], p_ref) for i, nm in enumerate(nomes_dir)}
    met_dir["ensemble_bruto"] = metricas_direcao(yb, p_bruto[av], p_ref)
    met_dir["ensemble"] = metricas_direcao(yb, p_cal[av], p_ref)
    for nm, pp in [*zip(nomes_dir, Po), ("ensemble", p_cal)]:
        met_dir[nm]["ic"] = ic_spearman(pp[av], ya)
    met_q = {nm: metricas_quantis(ya, Qo[i, av], q) for i, nm in enumerate(nomes_q)}
    met_q["ensemble_bruto"] = metricas_quantis(ya, Q_ens[av], q)
    met_q["ensemble_conformal"] = metricas_quantis(ya, Q_conf[av], q)

    br = lambda p: (p - yb) ** 2  # noqa: E731
    acertos = (p_cal[av] >= 0.5) == (yb == 1)
    acc_base = float(np.mean((p_ref >= 0.5) == (yb == 1)))
    pin = lambda Qm: pinball_pontual(ya, Qm, q)  # noqa: E731
    testes = {
        "dm_brier_vs_taxa_base": diebold_mariano(br(p_cal[av]), br(p_ref), h),
        "dm_brier_vs_random_walk": diebold_mariano(br(p_cal[av]), br(Po[1, av]), h),
        "dm_brier_vs_momentum": diebold_mariano(br(p_cal[av]), br(Po[2, av]), h),
        "binomial_vs_taxa_base": binomial_efetivo(acertos, h, acc_base),
        "dm_pinball_vs_random_walk": diebold_mariano(pin(Q_conf[av]), pin(Qo[1, av]), h),
        "dm_pinball_vs_taxa_base": diebold_mariano(pin(Q_conf[av]), pin(Qo[0, av]), h),
    }

    # Volatilidade: σ_h do GARCH vs. vol histórica de 63 dias, contra a variância realizada
    rv = (r_diario**2).rolling(h).sum().shift(-h).reindex(idx).to_numpy()
    prev_garch = Xh[SIGMA].to_numpy()[sl] ** 2
    prev_hist = (X["vol_63"].to_numpy()[sl] ** 2) * h / 252
    okv = av & np.isfinite(rv) & (rv > 0) & np.isfinite(prev_hist) & (prev_hist > 0) & (prev_garch > 0)
    qlike = lambda f: rv[okv] / f[okv] - np.log(rv[okv] / f[okv]) - 1  # noqa: E731
    met_vol = {
        "qlike_garch": float(qlike(prev_garch).mean()),
        "qlike_historica": float(qlike(prev_hist).mean()),
        "correl_garch": ic_spearman(prev_garch[okv], rv[okv]),
        "correl_historica": ic_spearman(prev_hist[okv], rv[okv]),
        "dm_garch_vs_historica": diebold_mariano(qlike(prev_garch), qlike(prev_hist), h),
    }

    met_rel = None
    if y_rel is not None and np.isfinite(P_rel[:, sl]).all():
        yro = y_rel[sl]
        pr_bruto, pr_cal, cal_r, _, yr_bin = _trilha_probabilidades(P_rel[:, sl], yro, h, passo, cfg)
        avr = cal_r & np.isfinite(yro)
        met_rel = metricas_direcao(yr_bin[avr], pr_cal[avr], P_rel[0, sl][avr])
        met_rel["dm_p"] = diebold_mariano((pr_cal[avr] - yr_bin[avr]) ** 2, (P_rel[0, sl][avr] - yr_bin[avr]) ** 2, h)["p_valor"]
        met_rel["p_hoje"] = float(pr_cal[-1])
        met_rel["taxa_base"] = float(P_rel[0, n - 1])

    # ------------------------------------------------------------ score histórico e de hoje
    outros = [Po[1], Po[2]]
    s_on = score_online(np.nan_to_num(y_bin), p_cal, Po[0], outros, conc, estab, av, h, passo, cfg.score)
    faixas = acerto_por_faixa(s_on["score"], (p_cal >= 0.5) == (np.nan_to_num(y_bin) == 1), av, h, cfg.score)
    vant = avaliar_vantagem(yb, p_cal[av], p_ref, [Po[1, av], Po[2, av]], h, cfg.score)
    comps = {
        "desempenho": comp_desempenho(vant["bss"], cfg.score),
        "forca": float(comp_forca(p_cal[-1:], Po[0, -1:], cfg.score)[0]),
        "concordancia": float(conc[-1]),
        "estabilidade": float(estab[-1]),
    }
    score = float(combinar(comps, vant["vantagem"], cfg.score))
    rotulo = cfg.score.rotulo(score)

    # ------------------------------------------------------------ previsão de hoje
    preco = float(close.iloc[-1])
    quantis_hoje = {qq: float(Q_conf[-1, i]) for i, qq in enumerate(q)}
    w_hoje = pesos[-1]  # pesos da última linha = todas as previsões já realizadas até hoje
    if vant["vantagem"]:
        direcao = "alta" if p_cal[-1] >= 0.5 else "queda"
        mensagem = f"Tendência de {direcao}: P(alta) = {p_cal[-1]:.0%}"
    else:
        mensagem = f"{SEM_VANTAGEM} (frequência histórica de alta: {Po[0, -1]:.0%})"
    hoje = PrevisaoHoje(
        p_alta=float(p_cal[-1]), p_alta_bruta=float(p_bruto[-1]), p_taxa_base=float(Po[0, -1]),
        p_supera_indice=met_rel["p_hoje"] if met_rel else None,
        quantis=quantis_hoje, precos={qq: preco * float(np.exp(v)) for qq, v in quantis_hoje.items()},
        retorno_mediano=float(np.expm1(quantis_hoje[0.5])),
        vol_periodo=float(Xh[SIGMA].iloc[-1]), vol_anualizada=float(Xh["garch_vol_h"].iloc[-1]),
        score=score, rotulo=rotulo, vantagem=vant["vantagem"], componentes=comps,
        pesos=dict(zip(nomes_dir, map(float, w_hoje))), mensagem=mensagem,
    )

    # ------------------------------------------------------------ backtest, SHAP, séries
    bt = backtest_limiar(pd.Series(p_cal, index=idx)[calibrado], close, caixa, cfg.backtest["limiar"],
                         cfg.backtest["custo_operacao"], cfg.backtest["execucao_defasagem"])
    shap_global, shap_local = _shap(lgbm_final, Xh, n)

    posicoes = np.arange(oos, n) + h
    data_alvo = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
    dentro = posicoes < n
    data_alvo[dentro] = Xh.index[posicoes[dentro]]
    serie = pd.DataFrame({"p_cal": p_cal, "p_bruto": p_bruto, "y": yo, "score": s_on["score"],
                          "q10": Q_conf[:, q.index(0.10)], "q50": Q_conf[:, q.index(0.50)], "q90": Q_conf[:, q.index(0.90)],
                          "calibrado": calibrado, "preco": close_x.to_numpy()[sl], "data_alvo": data_alvo}, index=idx)
    iq = {"q10": q.index(0.10), "q50": q.index(0.50), "q90": q.index(0.90)}
    previsoes = {nm: pd.DataFrame({**{k: Qo[i][:, j] for k, j in iq.items()}, "p_alta": Po[i]}, index=idx)
                 for i, nm in enumerate(nomes_q)}
    previsoes["ensemble"] = pd.DataFrame({**{k: Q_conf[:, j] for k, j in iq.items()}, "p_alta": p_cal}, index=idx)
    res = ResultadoHorizonte(
        h=h, nome=cfg.nome_horizonte(h), hoje=hoje,
        periodo_oos=(idx[av][0].strftime("%d/%m/%Y"), idx[av][-1].strftime("%d/%m/%Y")) if av.any() else ("", ""),
        n_oos=int(av.sum()), metricas_direcao=met_dir, metricas_quantis=met_q, metricas_vol=met_vol,
        metricas_relativo=met_rel, testes=testes, acerto_faixas=faixas, vantagem_historica=vant,
        calibracao=_curva_calibracao(p_cal[av], yb, max(30, 3 * h)), backtest=bt,
        shap_global=shap_global, shap_local=shap_local, serie_oos=serie, tempo_s=time.time() - t_ini,
        previsoes=previsoes,
    )
    return res


def _fundacional(close: pd.Series, datas: pd.DatetimeIndex, h: int, q: list[float], cfg: Config):
    """Previsões zero-shot do Chronos-Bolt para cada linha fora da amostra (só passado como contexto)."""
    from analisador.modelos import fundacionais
    if not fundacionais.disponivel():
        log.warning("modelos fundacionais ativados, mas o extra não está instalado (uv sync --extra fundacionais)")
        return None
    try:
        modelo = fundacionais.ChronosBolt(cfg.fundacionais["modelo"], cfg.fundacionais["contexto"])
        origens = close.index.get_indexer(datas)
        return modelo.prever(close.to_numpy(), origens, h, q)
    except Exception as e:
        log.warning("Chronos-Bolt falhou; seguindo sem ele", extra={"erro": str(e)})
        return None


def _shap(modelo: LGBMDirecao, X: pd.DataFrame, n: int) -> tuple[list, list]:
    if modelo.constante is not None:
        return [], []
    try:
        import shap
        cols = modelo.cols
        amostra = X[cols].iloc[max(0, n - 504):n]
        valores = shap.TreeExplainer(modelo.modelo).shap_values(amostra)
        if isinstance(valores, list):
            valores = valores[1]
        valores = np.asarray(valores)
        if valores.ndim == 3:
            valores = valores[:, :, 1]
        imp = np.abs(valores).mean(axis=0)
        ordem = np.argsort(imp)[::-1][:15]
        glob = [(cols[i], float(imp[i])) for i in ordem]
        ultima = valores[-1]
        ordem_l = np.argsort(np.abs(ultima))[::-1][:8]
        loc = [(cols[i], float(ultima[i]), float(amostra.iloc[-1, i])) for i in ordem_l]
        return glob, loc
    except Exception as e:  # SHAP é explicativo; não pode derrubar a análise
        log.warning("SHAP indisponível", extra={"erro": str(e)})
        return [], []
