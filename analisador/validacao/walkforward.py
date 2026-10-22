"""Janelas walk-forward expansivas com purga e embargo (López de Prado)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Janela:
    treino_fim: int   # treino = linhas [0, treino_fim)
    teste_ini: int    # teste = linhas [teste_ini, teste_fim)
    teste_fim: int


def gerar_janelas(n: int, h: int, treino_minimo: int, passo: int, embargo: int, treino_minimo_efetivo: int = 250) -> list[Janela]:
    """O rótulo da linha i usa o preço de i+h. Para prever a partir de k:

    - purga: só entram no treino linhas com i + h < k (rótulo realizado antes do teste);
    - embargo: descarta mais `embargo` linhas antes disso, como margem contra
      dependência serial entre o fim do treino e o início do teste.
    """
    janelas = []
    k = treino_minimo
    while k < n:
        fim = min(k + passo, n)
        treino_fim = k - h - embargo
        if treino_fim >= treino_minimo_efetivo:
            janelas.append(Janela(treino_fim, k, fim))
        k = fim
    return janelas


def divisao_interna(treino_fim: int, h: int, fracao_validacao: float) -> tuple[int, int]:
    """Divisão temporal dentro do treino (early stopping / CV aninhada), também purgada.

    Devolve (fim_do_subtreino, inicio_da_validacao); a validação vai até treino_fim.
    """
    ini_val = int(treino_fim * (1 - fracao_validacao))
    return max(ini_val - h, 1), ini_val
