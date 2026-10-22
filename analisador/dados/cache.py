"""Cache local em Parquet com atualização incremental."""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from analisador.log import obter_logger

log = obter_logger(__name__)
Baixador = Callable[[date | None], pd.DataFrame]


class CacheParquet:
    def __init__(self, pasta: str | Path):
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self._meta_arq = self.pasta / "_atualizacoes.json"

    def _arquivo(self, chave: str) -> Path:
        return self.pasta / (re.sub(r"[^A-Za-z0-9_.-]", "_", chave) + ".parquet")

    def ler(self, chave: str) -> pd.DataFrame | None:
        arq = self._arquivo(chave)
        return pd.read_parquet(arq) if arq.exists() else None

    def gravar(self, chave: str, df: pd.DataFrame, hoje: date) -> None:
        df.to_parquet(self._arquivo(chave))
        meta = self._meta()
        meta[chave] = hoje.isoformat()
        self._meta_arq.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def atualizado_em(self, chave: str) -> date | None:
        v = self._meta().get(chave)
        return date.fromisoformat(v) if v else None

    def _meta(self) -> dict:
        if self._meta_arq.exists():
            return json.loads(self._meta_arq.read_text(encoding="utf-8"))
        return {}


def obter_com_cache(
    cache: CacheParquet,
    chave: str,
    baixar: Baixador,
    tolerancia_reajuste: float = 0.001,
    sobreposicao_dias: int = 10,
    hoje: date | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """Devolve a série completa, baixando só o trecho novo.

    Se o trecho sobreposto mudou mais que `tolerancia_reajuste` (ex.: o Yahoo reajustou o
    histórico por um dividendo), baixa tudo de novo para manter a série consistente.
    """
    hoje = hoje or date.today()
    antigo = cache.ler(chave)

    if antigo is not None and (offline or cache.atualizado_em(chave) == hoje):
        return antigo

    try:
        if antigo is None:
            novo = baixar(None)
            log.info("download completo", extra={"serie": chave, "linhas": len(novo)})
        else:
            inicio = (antigo.index[-1] - timedelta(days=sobreposicao_dias)).date()
            recente = baixar(inicio)
            comum = antigo.index.intersection(recente.index)
            ref = "Close"
            desvio = 0.0
            if len(comum):
                a, r = antigo.loc[comum, ref], recente.loc[comum, ref]
                desvio = float(((r - a).abs() / a.abs().where(a != 0)).max(skipna=True) or 0.0)
            if not len(comum) or desvio > tolerancia_reajuste:
                log.info("histórico reajustado pela fonte: novo download completo",
                         extra={"serie": chave, "desvio": round(desvio, 5)})
                novo = baixar(None)
            else:
                novo = pd.concat([antigo[antigo.index < recente.index[0]], recente])
                novo = novo[~novo.index.duplicated(keep="last")].sort_index()
                log.debug("atualização incremental", extra={"serie": chave, "novas": len(novo) - len(antigo)})
    except Exception as e:
        if antigo is None:
            raise
        log.warning("falha no download; usando cache", extra={"serie": chave, "erro": str(e)})
        return antigo

    cache.gravar(chave, novo, hoje)
    return novo
