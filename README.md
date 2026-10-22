# Analisador Preditivo de Ações

Analisa uma ação da B3 ou dos EUA e, para cada horizonte (1 dia, 1 semana, 2 semanas, 1 mês, 3 meses e 6 meses), entrega:

- **P(alta)** e **P(superar o índice)**;
- a **distribuição do retorno** (P5 a P95), convertida em faixa de preço;
- a **volatilidade prevista**;
- um **grau de confiabilidade de 0 a 100**, medido fora da amostra.

O princípio que guia tudo: **o retorno de ações é quase imprevisível; volatilidade e regime são bem mais previsíveis.** Por isso a ferramenta mede com rigor quanto o modelo realmente acerta. Quando não há vantagem sobre os baselines, ela diz *"Sem vantagem estatística sobre o acaso neste horizonte"* em vez de inventar confiança.

> Ferramenta educacional. **Não é recomendação de investimento.**

---

## 1. Instalação

Requer [uv](https://docs.astral.sh/uv/). O projeto usa **Python 3.12** porque o `hmmlearn` não tem binário para 3.14.

```bash
uv sync                          # dependências principais (versões fixadas no pyproject.toml / uv.lock)
uv sync --extra dashboard        # + Streamlit
uv sync --extra fundacionais     # + PyTorch e Chronos-Bolt (~1-2 GB, opcional)
```

Neste computador, `.venv` é um *junction* para `%USERPROFILE%\.venvs\analisador_acoes`. Assim o ambiente virtual fica fora do OneDrive e não é sincronizado.

## 2. Uso

### Tela simples: aderência dos modelos (recomendado)

Dê dois cliques em **`ABRIR_ANALISADOR.bat`**, ou rode `uv run streamlit run app.py`. Depois:

1. Busque a ação pelo nome ou código (qualquer ação ou ETF da B3 ou dos EUA) e clique em **Rodar modelos**. Leva cerca de 1 minuto na primeira vez.
2. Escolha o prazo: 1, 3 ou 6 meses. A tela informa **qual modelo teve maior aderência ao realizado** (menor erro médio do preço previsto) e se essa vantagem sobre o random walk é estatisticamente significativa.
3. Escolha um modelo para ver três coisas:
   - o gráfico **previsto × realizado**: o preço que ele previa *h* meses antes contra o que aconteceu;
   - a **projeção para os próximos meses**;
   - o ranking de todos os modelos.

O dashboard técnico completo continua disponível em `uv run streamlit run app_completo.py`.

### Publicar online (Streamlit Community Cloud, gratuito)

1. Acesse https://share.streamlit.io e entre com a conta do GitHub.
2. Clique em **Create app → Deploy a public app from GitHub** e preencha:
   - Repository: `tfig-lab/Predict`
   - Branch: `main`
   - Main file path: `app.py`
3. Em **Advanced settings**, escolha **Python 3.12**, porque o `hmmlearn` não tem binário para 3.13 ou mais novo. Opcional: em *Secrets*, `ANALISADOR_N_JOBS = "2"` limita os processos paralelos se faltar memória.
4. Clique em **Deploy**. A primeira instalação leva alguns minutos; as dependências vêm do `requirements.txt`.

**Limitações do plano gratuito:**
- a CPU é modesta: cada análise leva de 1 a 3 minutos no modo rápido;
- o app "dorme" depois de um tempo sem acessos e acorda no clique seguinte;
- o Yahoo Finance às vezes limita requisições vindas de servidores em nuvem;
- a API do Banco Central pode não responder fora do Brasil; nesse caso a análise segue sem Selic e IPCA, com aviso.

### Linha de comando

```bash
uv run python -m analisador PETR4                        # B3: adiciona .SA automaticamente
uv run python -m analisador AAPL --horizontes 5,21,63,126
uv run python -m analisador VALE3 --relatorio html --inicio 2010-01-01
uv run python -m analisador ITUB4 --rapido --abrir       # retreino menos frequente; abre o HTML
uv run streamlit run app.py                              # dashboard opcional
```

| Opção | Efeito |
|---|---|
| `--horizontes 5,21,63` | pregões à frente (padrão: `config.yaml`) |
| `--relatorio html\|terminal` | `html` (padrão) gera também o arquivo em `relatorios/` |
| `--inicio AAAA-MM-DD` | início do histórico (padrão: últimos 15 anos) |
| `--rapido` | retreino a cada 63 pregões em vez de 21 e sem modelos fundacionais |
| `--fundacionais` | inclui o Chronos-Bolt (requer o extra) |
| `--offline` / `--cache PASTA` | usa só o cache Parquet local |
| `--config ARQ` | outro `config.yaml` |

Tempo medido neste notebook (Ryzen 5 4600H, 6 núcleos): **cerca de 50 s por ação no modo completo**, com 15 anos de dados, 6 horizontes e retreino a cada 21 pregões.

## 3. Como ler a saída

### P(alta)
É a probabilidade calibrada de o preço estar acima do atual daqui a *h* pregões.

**Leia junto com a confiança.** Um asterisco (\*) no terminal, ou o selo "sem vantagem" no HTML, indica que o modelo não superou o acaso naquele prazo. Nesse caso o número não é sinal de direção; o relatório mostra a **frequência histórica de alta** como referência.

### Faixa de preço P10 a P90
Em 80% dos casos históricos, intervalos construídos assim contiveram o preço realizado. Essa cobertura é medida fora da amostra e aparece no relatório.

### Grau de confiabilidade (0 a 100)

| Componente | Peso | O que mede |
|---|---|---|
| (a) Desempenho comprovado | 40% | Brier skill score fora da amostra contra a taxa base. 5% de BSS dá nota máxima. |
| (b) Força do sinal | 25% | Distância entre P(alta) calibrada e a taxa base. 15 p.p. dá nota máxima. |
| (c) Concordância | 15% | Fração ponderada dos modelos do ensemble que apontam a mesma direção. |
| (d) Estabilidade | 20% | Persistência do regime HMM atual e volatilidade atual vs. a de longo prazo. |

Faixas: **0 a 39 Baixa, 40 a 69 Média, 70 a 100 Alta.** Os pesos e as escalas estão no `config.yaml`.

**Regras de honestidade:**
- Se o ensemble não for **significativamente melhor que a taxa base** (Diebold-Mariano, p < 0,05), ou se for pior que algum outro baseline, o score fica **limitado a 39 (Baixa)** e a direção não é exibida.
- Ao lado do score aparece sempre o **acerto histórico naquela faixa**: "nas vezes em que a confiança foi Baixa, acertou X% (n = …, n efetivo ≈ n/h)".
  - O score é recalculado ao longo de todo o walk-forward, usando só o desempenho já realizado em cada data.
  - Com confiança Baixa, esse acerto fica próximo da frequência histórica de alta, e isso é esperado. Em período de forte alta (ex.: PETR4 nos últimos anos), ele pode parecer alto **sem** refletir habilidade do modelo.

## 4. Metodologia

### Dados
- **Preços:** `yfinance`, ajustados por desdobramentos e proventos. Usa histórico máximo, limitado aos últimos 15 anos (`anos_maximos`).
- **Contexto:** índice de referência (`^BVSP` ou `^GSPC`), `BRL=X`, `^VIX`, `^TNX`, mais Selic (SGS 432) e IPCA (SGS 433) do Banco Central.
  - O IPCA é **indexado pela data de divulgação** (+45 dias), não pela de referência, para evitar look-ahead.
- **Cache Parquet incremental:** se a fonte reajustar o histórico (ex.: um dividendo), o trecho sobreposto muda e o histórico é baixado de novo por inteiro.
- **Validação dos dados:** preços zerados, máxima/mínima incoerentes, lacunas, saltos acima de 8σ robustos (picos isolados com reversão são removidos), volume nulo e **trechos de preço congelado**.
  - Esse último caso é real: o Yahoo devolve SUZB3, GGBR4 e WEGE3 com **anos de preço idêntico e volume zero** antes de o ticker atual existir.
  - Na primeira execução isso produziu um AUC falso de 0,80. Hoje o histórico passa a começar depois desse trecho, com aviso.
- **Fundamentos** (P/L, P/VP, DY, ROE, margem): só como contexto. São valores atuais, então não podem entrar no backtest.

### Alvo
Log-retorno acumulado em *h* pregões, nunca o preço. O alvo relativo é o retorno da ação menos o retorno do índice.

### Features
Todas são estacionárias e calculadas com dados até *t*:
- retornos em 1, 5, 21, 63, 126 e 252 dias, e momentum 12-1;
- volatilidade realizada (5, 21 e 63 dias), Parkinson e Garman-Klass;
- RSI, MACD, distância às médias de 20, 50 e 200 dias, Bandas de Bollinger, drawdown de 52 semanas e z-score do volume;
- beta, correlação e força relativa contra o índice;
- variações do dólar, VIX, juros, Selic e IPCA;
- calendário (mês, dia da semana);
- **vol GARCH e regime HMM, recalculados "online"**: os parâmetros são reestimados só com o passado.

### Modelos (interface comum: `fit` / `predict_proba` / `predict_quantiles`)

| Modelo | Papel |
|---|---|
| Taxa base | Fração histórica de janelas positivas. Baseline obrigatório. |
| Random walk com drift | Retorno médio e volatilidade históricos. Baseline obrigatório. |
| Momentum 12-1 | Frequência de alta condicionada ao sinal do momentum. Baseline obrigatório. |
| GARCH-FHS | GARCH / GJR / EGARCH com t de Student (escolha por BIC); σ do horizonte pela estrutura a termo, com resíduos empíricos. |
| HMM | 2 ou 3 regimes (BIC) sobre retorno e volatilidade. **Probabilidades filtradas** (o `predict_proba` do hmmlearn é suavizado e usa o futuro). |
| LightGBM | Classificador (direção) e 7 regressões quantílicas sobre o retorno padronizado pela vol GARCH. Regularização forte; hiperparâmetros por CV aninhada purgada só no treino. |
| Monte Carlo | 10.000 trajetórias GARCH-t + 10.000 por bootstrap em blocos dos resíduos, para o fan chart e eventos como P(> +10% em 3 meses). |
| Chronos-Bolt | Opcional, zero-shot. Só entra no ensemble enquanto vencer os baselines. |

**Ensemble:** média ponderada com pesos ∝ exp(−n_ef · Δperda), usando log loss para probabilidades e pinball para quantis. Os pesos são recalculados a cada dia **só com previsões cujo resultado já se realizou**, e n_ef = linhas/h impede que pouca evidência produza pesos extremos.

### Validação
- **Walk-forward expansivo**, com retreino a cada 21 pregões.
- **Purga** dos rótulos que cruzam o teste, mais **embargo de h pregões**.
- Scalers, calibradores, pesos e hiperparâmetros são ajustados só no passado de cada janela.

**Testes automatizados (`tests/`, 81 testes):**
- teste de *shift*: truncar os dados em *t* não muda nenhuma feature ≤ *t*;
- "futuro envenenado": trocar preços futuros por lixo também não muda nada até *t*;
- purga e embargo respeitados em todas as janelas;
- **alvo embaralhado → AUC ≈ 0,5**, mais um controle positivo: um vazamento plantado de propósito é detectado (AUC > 0,9);
- GARCH e HMM online sem look-ahead;
- cobertura do conformal em dados sintéticos com quebra de regime;
- CLI de ponta a ponta com cache, sem rede.

**Métricas, sempre ao lado dos baselines e nas mesmas linhas:**
- direção: acurácia, acurácia balanceada, AUC, Brier, log loss, BSS e IC de Spearman;
- quantis: pinball e **cobertura real** de 50/80/90%;
- volatilidade: QLIKE contra a vol histórica de 63 dias.

**Testes de significância:** Diebold-Mariano com HAC de h−1 lags e correção de Harvey-Leybourne-Newbold, e binomial numa subamostra **sem sobreposição** (1 a cada h pregões).

**Backtest:** comprado se P(alta) > 0,55, com execução no fechamento do dia seguinte ao sinal e custo de 0,08% por operação. O caixa rende Selic (B3) ou T-bill (EUA). A comparação com o buy-and-hold usa CAGR, Sharpe, Sortino, drawdown máximo e número de operações.

### Calibração e intervalos
- **Calibração:** isotônica quando há ≥ 300 observações *efetivas* (n/h); Platt nos demais casos. Duas salvaguardas, porque calibrar um preditor quase constante só adiciona ruído:
  - (1) encolhimento para a identidade, com κ = n_ef/(n_ef + 50);
  - (2) seleção prequencial: a calibração só é usada se, em todo o histórico já realizado, reduziu a log loss.
- **Intervalos:** CQR split-conformal com **ACI** (α adaptativo, atualizado com atraso de h). A janela de pontuações é max(504, 24·h) linhas, ou seja, ≥ 24 observações independentes.

## 5. Avaliação final: PETR4, VALE3, ITUB4 e AAPL

Rodada **única**, com o `config.yaml` congelado (sha256 `ce47b2559298…`), em 21/10/2026.

- Período fora da amostra: ~10 anos por ação, de 2016 a 2026, com cerca de 2.200 a 2.500 previsões por horizonte.
- Arquivos completos (terminal, HTML e JSON) em [`relatorios/avaliacao_final/`](relatorios/avaliacao_final/).

### Direção: BSS do ensemble calibrado vs. taxa base (p-valor de Diebold-Mariano)

| Horizonte | PETR4 | VALE3 | ITUB4 | AAPL |
|---|---|---|---|---|
| 1 dia | −0,02% (0,74) | −0,01% (0,53) | −0,02% (0,60) | −0,05% (0,93) |
| 1 semana | −0,33% (0,77) | +0,18% (0,34) | −0,45% (0,88) | −0,03% (0,53) |
| 2 semanas | +0,68% (0,05) | +0,71% (0,10) | −0,22% (0,78) | −0,34% (0,82) |
| 1 mês | −0,43% (0,93) | +1,32% (0,17) | +0,53% (0,23) | −0,97% (0,87) |
| 3 meses | −0,14% (0,55) | −0,80% (0,63) | +0,64% (0,25) | −1,81% (0,93) |
| 6 meses | −1,30% (0,72) | −1,14% (0,75) | −3,33% (0,89) | −2,47% (0,77) |

### Volatilidade: QLIKE GARCH vs. vol histórica de 63 dias (p-valor de DM; menor QLIKE = melhor)

| Horizonte | PETR4 | VALE3 | ITUB4 | AAPL |
|---|---|---|---|---|
| 1 dia | 1,57 vs 1,71 (0,005) | 1,55 vs 1,59 (0,013) | 1,43 vs 1,49 (0,002) | 1,59 vs 1,71 (<0,001) |
| 1 semana | 0,53 vs 0,68 (0,011) | 0,46 vs 0,50 (0,012) | 0,40 vs 0,47 (0,002) | 0,46 vs 0,57 (<0,001) |
| 2 semanas | 0,42 vs 0,59 (0,029) | 0,32 vs 0,36 (0,008) | 0,28 vs 0,36 (0,004) | 0,34 vs 0,44 (<0,001) |
| 1 mês | 0,40 vs 0,61 (0,050) | 0,26 vs 0,31 (0,019) | 0,21 vs 0,31 (0,010) | 0,28 vs 0,39 (0,002) |
| 3 meses | 0,36 vs 0,63 (0,065) | 0,25 vs 0,33 (0,025) | 0,18 vs 0,31 (0,031) | 0,29 vs 0,47 (0,014) |
| 6 meses | 0,30 vs 0,60 (0,069) | 0,20 vs 0,31 (0,034) | 0,16 vs 0,29 (0,054) | 0,26 vs 0,43 (0,003) |

### Intervalos: cobertura real do intervalo de 80% (conformal) e pinball vs. random walk

| Horizonte | PETR4 | VALE3 | ITUB4 | AAPL | Pinball vs. RW |
|---|---|---|---|---|---|
| 1 dia | 79,5% | 80,0% | 79,5% | 79,9% | melhor nos 4 (p < 0,001) |
| 1 semana | 79,5% | 79,8% | 79,2% | 80,0% | melhor nos 4 (p ≤ 0,017) |
| 2 semanas | 79,1% | 79,6% | 79,1% | 80,0% | melhor em 2 (PETR4, VALE3); empate nos outros |
| 1 mês | 79,1% | 80,1% | 79,2% | 79,9% | empate em 3; **pior** em VALE3 (p ≈ 0,01) |
| 3 meses | 79,7% | 80,2% | 81,8% | 81,3% | maior que o RW nos 4; **pior** de forma significativa em ITUB4 e AAPL |
| 6 meses | 83,5% | 84,7% | 82,8% | 78,5% | **pior** que o RW nos 4 (p ≤ 0,03) |

### Veredito: onde há vantagem real e onde não há

- **Direção (P(alta)): sem vantagem em nenhum dos 24 casos** (4 ações × 6 horizontes).
  - Nenhum BSS foi significativo. O mais próximo, PETR4 em 2 semanas (+0,68%, p = 0,05), não passou do limiar; e com 24 testes ao nível de 5% espera-se cerca de 1 falso positivo só por acaso.
  - O LightGBM sozinho foi **pior** que a taxa base em quase todos os casos, especialmente nos prazos longos, onde aprende regimes que se invertem.
  - P(superar o índice) também ficou sem vantagem em todos os casos.
  - Por isso **toda confiança direcional saiu "Baixa"**. É o resultado esperado, e a ferramenta o reporta em vez de escondê-lo.
- **Volatilidade: vantagem real e consistente.**
  - O GARCH teve QLIKE menor que a vol histórica nos **24 de 24** casos, com significância estatística em **20 de 24**. As exceções são PETR4 em 1, 3 e 6 meses (p de 0,05 a 0,07) e ITUB4 em 6 meses (p = 0,054).
- **Intervalos de preço: bem calibrados.**
  - Cobertura de 80% entre **78,5% e 84,7%**, e entre **79,1% e 80,1% até 1 mês**.
  - Até 1 semana, eles também são **mais precisos que o random walk** (pinball significativamente menor nos 4 ativos).
  - Em 3 e 6 meses a cobertura continua correta, mas o pinball é **pior** que o do random walk normal simples, ou seja, os intervalos longos são calibrados porém menos precisos. É um limite conhecido e não foi "consertado" depois de ver o resultado.
- **Backtest:** a estratégia baseada em P(alta) não supera o buy-and-hold de forma consistente. Sem vantagem direcional, ela não deve ser usada.

### Transparência sobre o processo
- **Contato prévio com o conjunto de teste:** antes do sistema atual, um protótipo descartado foi avaliado em PETR4, VALE3 e ITUB4 (AUC de 4 variantes). AAPL não foi vista antes.
- **Desenvolvimento só em tickers próprios:** todo o desenvolvimento usou SUZB3, RENT3, GGBR4, MSFT, JNJ e dados sintéticos. As mudanças feitas nessa fase foram metodológicas ou de tempo de execução, todas antes da rodada final:
  - detecção de preço congelado;
  - janela de 15 anos;
  - calibração com encolhimento e seleção prequencial;
  - janela do conformal em observações efetivas;
  - correção do nível de longo prazo do EGARCH com inovações t;
  - cadência das regressões quantílicas.
- **Nada ajustado depois:** nenhum parâmetro foi alterado depois de ver os resultados da avaliação final.

## 6. Limitações

- O modelo só vê preço, volume e séries de mercado. Notícias, balanços, eventos corporativos e mudanças estruturais ficam de fora.
- Rótulos sobrepostos reduzem muito a amostra efetiva: em 6 meses, 10 anos de teste equivalem a ~17 observações independentes. As métricas de prazos longos são ruidosas, e os testes já levam isso em conta.
- A frequência histórica de alta reflete o período analisado. Uma década de alta forte, como em PETR4, deixa a taxa base otimista.
- O Yahoo Finance tem falhas de dados. A validação detecta as mais comuns, mas não todas.
- O pré-treino do Chronos-Bolt pode conter séries que se sobrepõem ao período de teste, o que tornaria sua avaliação otimista. Por isso ele fica desligado por padrão.
- A vol GARCH de longo prazo do EGARCH desconsidera o viés de Jensen (pequeno).
- O backtest é deliberadamente simples: sem impostos, sem lote mínimo e sem venda a descoberto.

## 7. Estrutura

```
analisador/
  dados/        yahoo.py, bcb.py, cache.py, validacao.py
  features/     tecnicas.py, mercado.py, construir.py
  modelos/      base.py, baselines.py, garch.py, hmm.py, lgbm.py, montecarlo.py, fundacionais.py, ensemble.py
  validacao/    walkforward.py, metricas.py, testes_estatisticos.py, backtest.py
  calibracao/   isotonica.py, conformal.py, score.py
  relatorio/    terminal.py, html.py, template.html, formato.py
  horizonte.py  walk-forward completo de um horizonte
  pipeline.py   orquestração (horizontes em paralelo)
  cli.py
config.yaml     todos os parâmetros (horizontes, janelas, custos, limiares, pesos do score)
tests/          uv run pytest
app.py          dashboard Streamlit
```
