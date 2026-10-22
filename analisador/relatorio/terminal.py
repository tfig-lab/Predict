"""Saída em texto para o terminal."""
from __future__ import annotations

from analisador import __version__
from analisador.relatorio import formato as fmt
from analisador.tipos import ResultadoAnalise


def _pct(x: float | None, sinal: bool = False, casas: int = 1) -> str:
    return fmt.pct(x, casas, sinal)


def _sinal(x: float) -> str:
    return ("+" if x > 0 else "") + fmt.num(x, 3)


def _tabela(cab: list[str], linhas: list[list[str]], alinhar_dir: set[int]) -> str:
    larg = [max(len(c), *(len(l[i]) for l in linhas)) for i, c in enumerate(cab)]
    fmt = lambda cel: "  ".join(c.rjust(larg[i]) if i in alinhar_dir else c.ljust(larg[i]) for i, c in enumerate(cel))  # noqa: E731
    sep = "  ".join("─" * w for w in larg)
    return "\n".join([fmt(cab), sep, *map(fmt, linhas)])


def relatorio_terminal(r: ResultadoAnalise) -> str:
    moeda = "R$" if r.info.get("moeda") == "BRL" else r.info.get("moeda", "")
    t = r.tecnico
    out = [
        "",
        f"{r.info.get('nome', r.ticker)} ({r.ticker}) · {moeda} {fmt.num(r.preco_atual)} em {r.data_ultima:%d/%m/%Y}",
        f"Retornos: 1m {_pct(t['ret_1m'], True)} · 3m {_pct(t['ret_3m'], True)} · 12m {_pct(t['ret_12m'], True)} · "
        f"RSI {t['rsi']:.0f} · vol 21d {_pct(t['vol_21'], casas=0)} · regime HMM: {r.hmm['atual']} "
        f"({_pct(max(r.hmm['probs']), casas=0)})",
        "",
    ]
    cab = ["Horizonte", "P(alta)", "Ret. mediano", f"Faixa P10–P90 ({moeda})", "Vol prevista", "Confiança", "Acerto histórico na faixa"]
    linhas = []
    for h in r.horizontes:
        p = h.hoje
        fx = h.acerto_faixas.get(p.rotulo, {})
        acerto = f"{_pct(fx.get('acerto'), casas=0)} (n={fx.get('n', 0)}, n ef.≈{fx.get('n_efetivo', 0)})" if fx.get("n") else "sem histórico"
        linhas.append([
            h.nome,
            _pct(p.p_alta, casas=0) if p.vantagem else f"{_pct(p.p_alta, casas=0)}*",
            _pct(p.retorno_mediano, True),
            f"{fmt.num(p.precos[0.10])} – {fmt.num(p.precos[0.90])}",
            f"{_pct(p.vol_periodo)} ({_pct(p.vol_anualizada, casas=0)} a.a.)",
            f"{p.rotulo} ({p.score:.0f})",
            acerto,
        ])
    out.append(_tabela(cab, linhas, {1, 2, 4}))
    sem = [h.nome for h in r.horizontes if not h.hoje.vantagem]
    if sem:
        out.append(f"\n* Sem vantagem estatística sobre o acaso em: {', '.join(sem)}. Nesses prazos a probabilidade")
        out.append("  não deve ser lida como previsão de direção; a faixa de preço e a volatilidade continuam validadas.")
    for h in r.horizontes:
        if h.hoje.vantagem:
            out.append(f"  {h.nome}: {h.hoje.mensagem}")

    out += ["", "Validação walk-forward (fora da amostra) contra os baselines", ""]
    cab = ["Horizonte", "n", "AUC ens.", "BSS ens.", "melhor baseline (BSS)", "DM p (vs taxa base)", "binomial p",
           "Cobertura 80%", "QLIKE GARCH / hist.", "Backtest Sharpe (estr. / B&H)"]
    linhas = []
    for h in r.horizontes:
        md = h.metricas_direcao
        melhor = max(("passeio_aleatorio", "momentum"), key=lambda k: md[k]["bss"])
        bt = h.backtest
        linhas.append([
            h.nome, fmt.inteiro(h.n_oos), fmt.num(md["ensemble"]["auc"], 3), _sinal(md["ensemble"]["bss"]),
            f"{'RW' if melhor == 'passeio_aleatorio' else 'Momentum'} {_sinal(md[melhor]['bss'])}",
            fmt.num(h.testes["dm_brier_vs_taxa_base"]["p_valor"]), fmt.num(h.testes["binomial_vs_taxa_base"]["p_valor"]),
            _pct(h.metricas_quantis["ensemble_conformal"]["cobertura_80"]),
            f"{fmt.num(h.metricas_vol['qlike_garch'], 3)} / {fmt.num(h.metricas_vol['qlike_historica'], 3)}",
            f"{fmt.num(bt['estrategia']['sharpe'])} / {fmt.num(bt['buy_and_hold']['sharpe'])}",
        ])
    out.append(_tabela(cab, linhas, set(range(1, 10))))
    out += [
        "",
        "BSS = Brier skill score contra a taxa base (>0 = melhor). DM p < 0,05 = vantagem estatística.",
        "QLIKE menor = previsão de volatilidade melhor. n ef. = observações independentes (n / horizonte).",
    ]
    if r.avisos:
        out += ["", "Avisos dos dados:"] + [f"  - {a}" for a in r.avisos]
    out += ["", f"Predict v{__version__} · tempo de análise: {r.tempo_total_s:.0f} s.", ""]
    return "\n".join(out)
