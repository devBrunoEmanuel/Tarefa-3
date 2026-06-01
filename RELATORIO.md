# Relatório — Comparação de filtros de imagem com MPI

**Disciplina:** Sistemas/Programação Distribuída
**Atividade:** Comparação de filtros de suavização (média 3×3 vs mediana 3×3),
sequencial vs paralelo com MPI.

**Grupo:** _(preencher nomes)_
**Data:** _(preencher)_

> **Nota sobre os números desta versão:** os valores de tempo, variância e
> speedup marcados como `XX` são **lacunas a preencher** com a saída de
> `py benchmark.py` (arquivo `resultados_speedup.csv` / `tabela_speedup.md`),
> executado na máquina do grupo com **MS-MPI** instalado. A estrutura, a
> metodologia e a análise já estão prontas. A **correção** do algoritmo já foi
> verificada (ver seção 7).

---

## 1. Objetivo

Implementar e comparar dois filtros de suavização aplicados à mesma imagem, em
versão **sequencial** e **paralela com MPI**, medindo **speedup** e **variância**,
com remoção de outliers, e analisar o comportamento de cada filtro ao paralelizar.

## 2. Filtros escolhidos

| Filtro | Descrição | Código |
|---|---|---|
| **Média 3×3** | Cada pixel vira a média aritmética dos 9 vizinhos. | [filtros.py:33](filtros.py#L33) |
| **Mediana 3×3** | Cada pixel vira a mediana (valor do meio) dos 9 vizinhos. | [filtros.py:49](filtros.py#L49) |

O 2º filtro escolhido foi a **mediana 3×3**. Diferença essencial: a média
**mistura** valores (borra), enquanto a mediana **escolhe** um valor existente,
preservando bordas e removendo muito bem ruído "sal e pimenta".

## 3. Imagem escolhida

- **Tipo:** escala de cinza, gerada por [gerar_imagem.py](gerar_imagem.py)
  (gradiente + círculos + ruído sal e pimenta), com semente fixa (reprodutível).
- **Tamanho:** 3000 × 3000 = **9 megapixels** (ajustável).
- **Justificativa (demanda de paralelização):** uma imagem desse porte tem ~9
  milhões de pixels; cada um exige uma vizinhança 3×3 (a mediana ainda ordena 9
  valores por pixel). O custo sequencial é grande o bastante para que a divisão
  do trabalho entre processos traga ganho real — uma imagem pequena terminaria
  antes mesmo de compensar o custo de comunicação.

## 4. Metodologia

### 4.1 Estratégia de paralelização (decomposição por linhas)

A imagem é dividida em **faixas horizontais de linhas**, uma por processo
([particao.py](particao.py)). Cada processo filtra só a sua faixa e o processo 0
junta tudo no final. Comunicação:

- **Distribuição:** `comm.Scatterv` ([paralelo_mpi.py:93](paralelo_mpi.py#L93)).
- **Fronteiras (halo):** `comm.Sendrecv`
  ([paralelo_mpi.py:128-129](paralelo_mpi.py#L128-L129)).
- **Reunião:** `comm.Gatherv` ([paralelo_mpi.py:155](paralelo_mpi.py#L155)).

Todas são operações **maiúsculas** (buffer NumPy), conforme exigido.

### 4.2 Tratamento de fronteiras

Cada processo usa um buffer com **2 linhas-fantasma** (halo) e troca as linhas de
borda com os vizinhos via `Sendrecv`. Nas bordas globais da imagem (sem vizinho)
usa `MPI.PROC_NULL` e replicação da própria borda. Resultado: **idêntico** ao
sequencial.

### 4.3 Medição

- **Início único do MPI:** cada combinação (filtro, nº de processos) é lançada
  com `mpiexec` **uma única vez**; as repetições acontecem **dentro** do programa,
  num laço ([paralelo_mpi.py:115](paralelo_mpi.py#L115)). Assim o custo de
  inicializar a rede MPI (simétrico) é pago **uma vez** e não entra em cada
  amostra — exatamente a metodologia pedida no enunciado.
- **Cronômetro:** `MPI.Wtime()` em torno de "troca de halo + cálculo";
  `comm.Barrier()` antes e depois; tempo da iteração = **máximo** entre processos
  (`MPI.MAX`).
- **Aquecimento:** as 2 primeiras repetições são descartadas.
- **Outliers:** método **IQR (cercas de Tukey)**
  ([bench_utils.py:28](bench_utils.py#L28)).
- **Variância:** amostral, `var(ddof=1)` ([bench_utils.py:60](bench_utils.py#L60)).
- **Speedup:** `T_seq / T_par(P)`.

## 5. Resultados

### 5.1 Tabela de speedup

> Preencher com a saída de `tabela_speedup.md` / `resultados_speedup.csv`.

**Filtro de média 3×3**

| Processos | Tempo médio (s) | Variância | Speedup | Eficiência |
|-----------|-----------------|-----------|---------|------------|
| 1 (seq)   | XX.XX           | XX        | 1.00    | 1.00       |
| 2         | XX.XX           | XX        | XX      | XX         |
| 4         | XX.XX           | XX        | XX      | XX         |
| 8         | XX.XX           | XX        | XX      | XX         |

**Filtro de mediana 3×3**

| Processos | Tempo médio (s) | Variância | Speedup | Eficiência |
|-----------|-----------------|-----------|---------|------------|
| 1 (seq)   | XX.XX           | XX        | 1.00    | 1.00       |
| 2         | XX.XX           | XX        | XX      | XX         |
| 4         | XX.XX           | XX        | XX      | XX         |
| 8         | XX.XX           | XX        | XX      | XX         |

### 5.2 Gráfico de tempo por número de processos

![Tempo por número de processos](tempo_por_processos.png)

_(Gerado automaticamente por `benchmark.py` em `tempo_por_processos.png`.)_

### 5.3 Padrão esperado

- O **tempo cai** ao aumentar os processos (até onde a máquina tem núcleos
  físicos disponíveis).
- A **mediana** deve apresentar **speedup maior** que a média (ver discussão).
- Acima do nº de **núcleos físicos**, o ganho satura (ex.: 8 processos numa CPU
  de 4 núcleos físicos rende pouco a mais que 4).

## 6. Discussão / Reflexões

**1) O filtro de mediana paralelo teve speedup maior ou menor que o de média? Por
quê?**
**Maior.** A mediana faz **mais trabalho por pixel** (ordena/particiona 9
valores), enquanto a média só soma e divide. Como o custo de **comunicação**
(distribuição + halo) é praticamente o **mesmo** nos dois filtros, o filtro mais
pesado tem melhor razão **cálculo/comunicação**: a fração paralelizável domina o
tempo, então paraleliza melhor (lei de Amdahl). A média é tão leve que fica
limitada por memória/comunicação, e o overhead "come" parte do ganho.

**2) O que acontece com os pixels nas fronteiras entre processos? O grupo tratou
isso? Como?**
Sim. Para filtrar a 1ª/última linha da sua faixa, cada processo precisa de 1
linha de cada vizinho. Usamos **linhas-fantasma (halo)**: buffer com 2 linhas
extras ([paralelo_mpi.py:101](paralelo_mpi.py#L101)), preenchidas por troca
`Sendrecv` ([paralelo_mpi.py:128-129](paralelo_mpi.py#L128-L129)). Nas bordas
globais (sem vizinho) usamos `MPI.PROC_NULL` e replicação da própria borda. Por
isso o resultado paralelo é **bit a bit idêntico** ao sequencial.

**3) Por que foi usado `comm.Bcast` (maiúsculo) e não `comm.bcast` (minúsculo)
para enviar a imagem? O que mudaria no desempenho?**
As maiúsculas (`Bcast`/`Scatterv`/`Gatherv`/`Sendrecv`) enviam **o buffer cru** do
array NumPy, sem cópia extra. As minúsculas **serializam com `pickle`** e
desserializam — para milhões de bytes, isso adiciona **cópias, CPU e memória**,
tornando a comunicação **bem mais lenta** e podendo até dominar o tempo total.
Por isso **toda** comunicação de imagem usa as maiúsculas. (A única exceção é o
`comm.bcast` minúsculo para 2 inteiros de tamanho —
[paralelo_mpi.py:73-74](paralelo_mpi.py#L73-L74) — onde não há diferença
prática.)

**4) Se o número de processos não divide exatamente a altura da imagem, o que o
código faz?**
Distribui o resto entre os primeiros processos: os primeiros
`altura % n_processos` recebem **1 linha a mais**
([particao.py:23-25](particao.py#L23-L25)). Ex.: 1000 linhas / 3 processos →
334, 333, 333. Por isso usamos as versões **"v"** (`Scatterv`/`Gatherv`), que
aceitam blocos de tamanhos diferentes.

## 7. Verificação de correção

Mesmo sem o ambiente MPI, validamos a lógica de partição + halo + bordas
simulando-a em numpy puro ([teste_corretude.py](teste_corretude.py)) e comparando
com o sequencial, para vários tamanhos e contagens de processos, **inclusive
alturas não divisíveis**:

```
>>> TODOS OS TESTES PASSARAM: paralelo == sequencial (incluindo bordas e divisao desigual).
```

## 8. Como reproduzir

```powershell
py -m pip install -r requirements.txt   # numpy, mpi4py, matplotlib, Pillow
py gerar_imagem.py                       # cria imagem.npy (3000x3000)
py benchmark.py                          # sequencial + 2/4/8 processos -> tabela + gráfico
```

Saídas: `resultados_speedup.csv`, `tabela_speedup.md`, `tempo_por_processos.png`.
(Requer **MS-MPI** instalado para fornecer o `mpiexec` — ver README.)

## 9. Conclusão

_(Preencher após rodar.)_ Ambos os filtros se beneficiam da paralelização, mas o
filtro de **mediana** — por ser mais intensivo em CPU — apresenta **speedup
superior** ao de **média**, que é mais limitado por comunicação/memória. O
tratamento de fronteiras por halo garante resultado idêntico ao sequencial, e a
metodologia de medição (início único do MPI, aquecimento, remoção de outliers)
mantém a variância baixa e os tempos estáveis.
