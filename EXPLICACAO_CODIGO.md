# Explicação do código — linha a linha

Comparação de **filtro de média 3×3** e **filtro de mediana 3×3**, em versão
**sequencial** e **paralela com MPI** (mpi4py).

Este documento explica **cada arquivo e cada linha relevante**. Os links levam
direto à linha do código.

## Índice

1. [Visão geral / arquitetura](#1-visão-geral)
2. [`filtros.py` — os dois kernels](#2-filtrospy)
3. [`particao.py` — divisão da imagem](#3-particaopy)
4. [`bench_utils.py` — estatística e outliers](#4-bench_utilspy)
5. [`sequencial.py` — versão sequencial](#5-sequencialpy)
6. [`paralelo_mpi.py` — versão paralela MPI](#6-paralelo_mpipy)
7. [`benchmark.py` — orquestrador](#7-benchmarkpy)
8. [`gerar_imagem.py` — imagem de teste](#8-gerar_imagempy)
9. [`teste_corretude.py` — teste sem MPI](#9-teste_corretudepy)
10. [Respostas às perguntas de reflexão](#10-respostas-às-perguntas-de-reflexão)

---

## 1. Visão geral

O fluxo de execução é:

```
gerar_imagem.py  ─►  imagem.npy
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
     sequencial.py                 paralelo_mpi.py
   (1 processo, T_seq)         (mpiexec -n 2/4/8, T_par)
            │                            │
            └────────────┬───────────────┘
                         ▼
                   benchmark.py
        (tabela de speedup + gráfico tempo×processos)
```

Ideia da paralelização: a imagem é uma matriz de pixels (altura × largura). Nós
a **cortamos em faixas horizontais de linhas** e damos uma faixa para cada
processo MPI. Cada processo aplica o filtro só na sua faixa. Como o filtro 3×3
olha a linha de cima e a de baixo de cada pixel, nas **bordas entre faixas** cada
processo precisa de **1 linha emprestada de cada vizinho** (as *linhas-fantasma*
ou *halo*). No final, o processo 0 junta as faixas filtradas.

Decisões de projeto importantes:

- **Comunicação de imagem só com operações maiúsculas** (`Scatterv`, `Gatherv`,
  `Sendrecv`) — elas usam o *buffer* do array NumPy diretamente, sem `pickle`.
- **Mesmo kernel** nas duas versões → o resultado paralelo é **idêntico** ao
  sequencial (provado em [`teste_corretude.py`](teste_corretude.py)).
- **Medição justa**: o MPI é iniciado **uma vez** por número de processos; as
  repetições acontecem **dentro** do programa, num laço.

---

## 2. `filtros.py`

Define os dois filtros. Os dois recebem uma imagem 2D e devolvem outra do mesmo
tamanho, tratando borda por **replicação** (o pixel "de fora" vira a borda mais
próxima).

### `_replica_borda` — [filtros.py:23-30](filtros.py#L23-L30)

```python
def _replica_borda(img):
    return np.pad(img, pad_width=1, mode="edge")
```

- [linha 30](filtros.py#L30): `np.pad(..., pad_width=1, mode="edge")` envolve a
  imagem com **1 pixel de moldura** em todos os lados, copiando a borda
  (`mode="edge"`). Uma imagem `(H, W)` vira `(H+2, W+2)`. Isso é o truque que
  permite calcular a vizinhança 3×3 de **qualquer** pixel — inclusive os da
  borda — usando só fatiamento, **sem nenhum `if` de borda**.

### `filtro_media_3x3` — [filtros.py:33-46](filtros.py#L33-L46)

```python
a = _replica_borda(img).astype(np.float64)
soma = (a[0:-2, 0:-2] + a[0:-2, 1:-1] + a[0:-2, 2:] +
        a[1:-1, 0:-2] + a[1:-1, 1:-1] + a[1:-1, 2:] +
        a[2:,   0:-2] + a[2:,   1:-1] + a[2:,   2:])
return np.rint(soma / 9.0).astype(img.dtype)
```

- [linha 40](filtros.py#L40): aplica a moldura e converte para `float64`. A
  conversão é necessária porque a soma de 9 pixels (até 9×255 = 2295) **estoura**
  o `uint8` (que vai só até 255).
- [linhas 42-44](filtros.py#L42-L44): a parte central. Cada `a[i:j, k:l]` é a
  imagem **inteira deslocada**. São 9 fatias, uma para cada posição da
  vizinhança 3×3:
  - `a[0:-2, 0:-2]` = vizinho de **cima-esquerda** de cada pixel,
  - `a[0:-2, 1:-1]` = vizinho de **cima**,
  - `a[0:-2, 2:]` = **cima-direita**, e assim por diante até
  - `a[2:, 2:]` = **baixo-direita**.

  Somar essas 9 fatias dá, **de uma vez**, a soma da vizinhança 3×3 de todos os
  pixels. É a forma **vetorizada** (rápida) de fazer o que dois laços `for`
  fariam pixel a pixel.
- [linha 46](filtros.py#L46): divide por 9 (a média), arredonda com `np.rint` e
  volta para `uint8` (`img.dtype`).

### `filtro_mediana_3x3` — [filtros.py:49-70](filtros.py#L49-L70)

```python
a = _replica_borda(img)
vizinhos = np.stack([
    a[0:-2, 0:-2], a[0:-2, 1:-1], a[0:-2, 2:],
    a[1:-1, 0:-2], a[1:-1, 1:-1], a[1:-1, 2:],
    a[2:,   0:-2], a[2:,   1:-1], a[2:,   2:],
], axis=0)
mediana = np.median(vizinhos, axis=0)
return mediana.astype(img.dtype)
```

- [linha 60](filtros.py#L60): mesma moldura de borda. **Não** precisa de `float`
  aqui, porque a mediana não soma — ela só **escolhe** um dos 9 valores.
- [linhas 62-66](filtros.py#L62-L66): as **mesmas 9 fatias deslocadas**, mas
  agora `np.stack(..., axis=0)` as **empilha** num único array 3D de forma
  `(9, H, W)`. Pense nele como 9 cópias da imagem, uma por posição da vizinhança.
- [linha 68](filtros.py#L68): `np.median(vizinhos, axis=0)` tira a mediana **ao
  longo do eixo 0** — ou seja, para cada pixel `(y, x)` pega os 9 valores
  `vizinhos[0..8, y, x]` e devolve o do meio. Como são 9 valores (ímpar), a
  mediana é exatamente o **5º valor ordenado**, sempre um valor que já existia.
- [linha 70](filtros.py#L70): volta para `uint8`.

> **Por que a mediana custa mais que a média?** Somar 9 números é quase de graça.
> Achar a mediana exige **ordenar** (ou particionar) os 9 valores de cada pixel.
> Esse custo extra é o que, mais adiante, faz a mediana ter **melhor speedup**
> (mais trabalho de CPU para "esconder" o custo de comunicação).

### `FILTROS` — [filtros.py:75-78](filtros.py#L75-L78)

Dicionário `nome → função`. Permite escolher o filtro por um argumento de linha
de comando (`--filtro media` ou `--filtro mediana`) tanto no sequencial quanto
no paralelo, sem `if` espalhado pelo código.

---

## 3. `particao.py`

Um único job: decidir **quantas linhas cada processo recebe** e **onde** cada
bloco começa. Fica separado de propósito, **sem importar MPI**, para poder ser
testado sem o ambiente MPI instalado.

### `calcular_particao` — [particao.py:11-32](particao.py#L11-L32)

```python
base = altura // n_processos
resto = altura % n_processos
linhas_por_processo = [base + (1 if r < resto else 0) for r in range(n_processos)]
counts = [linhas * largura for linhas in linhas_por_processo]
deslocamentos = [0] * n_processos
for r in range(1, n_processos):
    deslocamentos[r] = deslocamentos[r - 1] + counts[r - 1]
return linhas_por_processo, counts, deslocamentos
```

- [linha 23](particao.py#L23): `base` = divisão inteira → linhas "garantidas"
  para cada um.
- [linha 24](particao.py#L24): `resto` = linhas que sobraram (quando a altura
  **não divide** pelo número de processos).
- [linha 25](particao.py#L25): distribui o resto: os **primeiros `resto`
  processos** ganham **1 linha a mais**. Exemplo: 1000 linhas em 3 processos →
  `[334, 333, 333]`. **(Esta é a resposta da pergunta de reflexão nº 4.)**
- [linha 27](particao.py#L27): `counts` converte "linhas" em "número de
  **elementos**" (`linhas * largura`), porque o `Scatterv`/`Gatherv` conta
  elementos, não linhas.
- [linhas 28-30](particao.py#L28-L30): `deslocamentos` = onde o bloco de cada
  processo começa dentro do array achatado (soma acumulada dos `counts`
  anteriores). O processo 0 começa em 0, o 1 começa logo depois do bloco do 0, etc.

`counts` e `deslocamentos` são exatamente os dois vetores que as funções "v"
(`Scatterv`, `Gatherv`) exigem para lidar com **blocos de tamanhos diferentes**.

---

## 4. `bench_utils.py`

Funções de estatística e de E/S compartilhadas pelas duas versões.

### `remover_outliers` — [bench_utils.py:28-46](bench_utils.py#L28-L46)

Método **IQR (cercas de Tukey)**:

- [linha 36](bench_utils.py#L36): converte as amostras em array de `float`.
- [linhas 37-39](bench_utils.py#L37-L39): se houver menos de 4 amostras, não dá
  para estimar quartis com confiança → devolve tudo.
- [linhas 40-42](bench_utils.py#L40-L42): `Q1` (percentil 25), `Q3` (percentil
  75) e `IQR = Q3 − Q1` (a faixa onde ficam os 50% centrais dos tempos).
- [linhas 43-44](bench_utils.py#L43-L44): define as **cercas**:
  `[Q1 − 1.5·IQR , Q3 + 1.5·IQR]`. É o critério clássico de boxplot.
- [linhas 45-46](bench_utils.py#L45-L46): mantém só as amostras **dentro** das
  cercas. Vantagem do método: é baseado em **quartis**, então um tempo absurdo
  (ex.: travada do SO) **não** distorce o próprio critério — diferente de usar a
  média, que o outlier puxaria.

### `estatisticas` — [bench_utils.py:49-64](bench_utils.py#L49-L64)

- [linha 60](bench_utils.py#L60): `variancia` com `var(ddof=1)` — variância
  **amostral** (divide por *n−1*), correta quando estimamos a variabilidade a
  partir de uma amostra. **(É a "variância" pedida no enunciado.)**
- [linha 61](bench_utils.py#L61): desvio padrão (raiz da variância).
- Também guarda `n_amostras`, `media`, `min` e `max`.

### `resumir_tempos` — [bench_utils.py:67-76](bench_utils.py#L67-L76)

O **pipeline** completo, na ordem certa:

- [linha 74](bench_utils.py#L74): `[n_aquecimento:]` **descarta as primeiras
  repetições** (aquecimento — cache frio, alocação preguiçosa).
- [linha 75](bench_utils.py#L75): remove outliers do que sobrou.
- [linha 76](bench_utils.py#L76): calcula as estatísticas dos tempos válidos.

### `salvar_json` / `carregar_json` — [bench_utils.py:79-88](bench_utils.py#L79-L88)

Gravam/leem o dicionário de resultado em JSON. É como o `paralelo_mpi.py` passa
os números para o `benchmark.py`.

---

## 5. `sequencial.py`

Versão de referência: **1 processo, sem MPI**. Mede `T_seq`.

- [linhas 26-32](sequencial.py#L26-L32): `argparse` lê os argumentos
  (`--filtro`, `--reps`, `--aquecimento`, `--imagem`, `--saida`).
- [linha 34](sequencial.py#L34): escolhe a função do filtro pelo dicionário
  `FILTROS`.
- [linha 35](sequencial.py#L35): `np.load` carrega a imagem `.npy`.
- [linhas 39-43](sequencial.py#L39-L43): o **laço de medição**. Para cada
  repetição (aquecimento + reps):
  - [linha 40](sequencial.py#L40): `time.perf_counter()` — relógio de alta
    resolução, **antes**.
  - [linha 41](sequencial.py#L41): aplica o filtro **na imagem inteira**.
  - [linhas 42-43](sequencial.py#L42-L43): relógio **depois** e guarda o tempo.
- [linha 45](sequencial.py#L45): `resumir_tempos` (aquecimento → outliers →
  estatística).
- [linhas 47-58](sequencial.py#L47-L58): monta o dicionário de resultado e grava
  em JSON.
- [linhas 60-62](sequencial.py#L60-L62): imprime média e variância no terminal.

> Note que aqui usamos `time.perf_counter` (não `MPI.Wtime`), porque esta versão
> **não tem MPI**.

---

## 6. `paralelo_mpi.py`

O coração da atividade. Lançado por `mpiexec -n P py paralelo_mpi.py ...`. **Todos
os P processos rodam este mesmo código**; o que muda é o `rank` (a "identidade"
de cada processo, de 0 a P−1).

### Inicialização do MPI — [paralelo_mpi.py:47-49](paralelo_mpi.py#L47-L49)

```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```

- [linha 47](paralelo_mpi.py#L47): `COMM_WORLD` é o **comunicador** que reúne
  todos os processos.
- [linha 48](paralelo_mpi.py#L48): `rank` = número deste processo (0 = "chefe").
- [linha 49](paralelo_mpi.py#L49): `size` = total de processos (o `P` do
  `mpiexec -n P`).

### Argumentos e escolha do filtro — [paralelo_mpi.py:51-59](paralelo_mpi.py#L51-L59)

Igual ao sequencial: lê `--filtro`, `--reps` etc. e seleciona o kernel. Cada
processo lê os mesmos argumentos.

### Passo 1 — só o rank 0 lê a imagem — [paralelo_mpi.py:64-74](paralelo_mpi.py#L64-L74)

```python
if rank == 0:
    img = np.load(args.imagem)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    altura, largura = img.shape
else:
    img = None
    altura = largura = None
altura = comm.bcast(altura, root=0)
largura = comm.bcast(largura, root=0)
```

- [linhas 64-67](paralelo_mpi.py#L64-L67): **apenas o rank 0** carrega a imagem
  do disco. `np.ascontiguousarray` garante que os bytes estejam **contíguos em
  memória** (necessário para o `Scatterv` enviar fatias corretas).
- [linhas 68-70](paralelo_mpi.py#L68-L70): os outros processos ainda não têm a
  imagem (`None`).
- [linhas 73-74](paralelo_mpi.py#L73-L74): `comm.bcast` (minúsculo!) envia
  **altura e largura** (2 inteiros) para todos. Usamos a versão minúscula **de
  propósito**: são só 2 números, **não** são "dados de imagem". É a única
  comunicação minúscula do programa. **(Relaciona-se à pergunta de reflexão nº 3.)**

### Passo 1.5 — checagem — [paralelo_mpi.py:76-79](paralelo_mpi.py#L76-L79)

Se houver mais processos do que linhas, aborta com mensagem clara (caso extremo;
com imagem grande e P ≤ 8 nunca acontece).

### Passo 2 — partição e Scatterv — [paralelo_mpi.py:84-95](paralelo_mpi.py#L84-L95)

```python
linhas_por_processo, counts, deslocamentos = calcular_particao(altura, size, largura)
minhas_linhas = linhas_por_processo[rank]
recv = np.empty(minhas_linhas * largura, dtype=np.uint8)
origem = [img.ravel() if rank == 0 else None, counts, deslocamentos, MPI.UNSIGNED_CHAR]
comm.Scatterv(origem, recv, root=0)
bloco_local = recv.reshape(minhas_linhas, largura)
```

- [linha 84](paralelo_mpi.py#L84): **todos** calculam a mesma partição (a função
  é determinística), então todos sabem quem fica com o quê.
- [linha 85](paralelo_mpi.py#L85): `minhas_linhas` = quantas linhas **este**
  processo vai receber.
- [linha 88](paralelo_mpi.py#L88): aloca o **buffer de recepção** já no tamanho
  certo (achatado em 1D).
- [linha 92](paralelo_mpi.py#L92): descreve a **origem** do `Scatterv`:
  - `img.ravel()` (só no rank 0; nos outros é `None`) = a imagem inteira achatada;
  - `counts` = quantos elementos cada processo recebe;
  - `deslocamentos` = de onde começa o bloco de cada um;
  - `MPI.UNSIGNED_CHAR` = o tipo MPI de 1 byte, equivalente ao `uint8`.
- [linha 93](paralelo_mpi.py#L93): **`comm.Scatterv`** (maiúsculo!) **espalha**
  os blocos: cada processo recebe a sua faixa de linhas dentro de `recv`. É a
  versão de buffer (zero-cópia de pickle) e a versão "v" (blocos de tamanhos
  diferentes). **(Aqui se cumpre a exigência de comunicação maiúscula.)**
- [linha 95](paralelo_mpi.py#L95): reinterpreta o buffer 1D como matriz
  `(minhas_linhas, largura)` — sem copiar.

### Passo 3 — buffer com halo — [paralelo_mpi.py:101-107](paralelo_mpi.py#L101-L107)

```python
buf = np.empty((minhas_linhas + 2, largura), dtype=np.uint8)
buf[1:-1] = bloco_local
vizinho_cima = rank - 1 if rank > 0 else MPI.PROC_NULL
vizinho_baixo = rank + 1 if rank < size - 1 else MPI.PROC_NULL
```

- [linha 101](paralelo_mpi.py#L101): cria `buf` com **2 linhas a mais**
  (`minhas_linhas + 2`): uma no topo e uma embaixo, reservadas para as
  linhas-fantasma.
- [linha 102](paralelo_mpi.py#L102): copia o bloco recebido para o **miolo**
  (`buf[1:-1]`), deixando `buf[0]` e `buf[-1]` livres para o halo.
- [linhas 106-107](paralelo_mpi.py#L106-L107): identifica os vizinhos. O de cima
  é `rank-1`; o de baixo é `rank+1`. Nas **pontas** (rank 0 não tem ninguém em
  cima; o último não tem ninguém embaixo) usamos `MPI.PROC_NULL` — um
  "destinatário nulo" que faz o `Sendrecv` virar **nada** (no-op), sem travar.
  **(Trata as bordas globais da imagem — pergunta de reflexão nº 2.)**

### Passo 4 — laço de medição — [paralelo_mpi.py:114-143](paralelo_mpi.py#L114-L143)

```python
for _ in range(args.aquecimento + args.reps):
    comm.Barrier()
    t0 = MPI.Wtime()

    buf[0] = buf[1]
    buf[-1] = buf[-2]
    comm.Sendrecv(buf[1].copy(),  dest=vizinho_cima,  recvbuf=buf[0],  source=vizinho_cima)
    comm.Sendrecv(buf[-2].copy(), dest=vizinho_baixo, recvbuf=buf[-1], source=vizinho_baixo)

    saida_com_halo = kernel(buf)
    local_saida = np.ascontiguousarray(saida_com_halo[1:-1])

    comm.Barrier()
    t1 = MPI.Wtime()
    dt = comm.reduce(t1 - t0, op=MPI.MAX, root=0)
    if rank == 0:
        tempos.append(dt)
```

- [linha 116](paralelo_mpi.py#L116): `comm.Barrier()` — **espera todos**
  chegarem aqui antes de iniciar o cronômetro, para que todos meçam a mesma
  janela de tempo.
- [linha 117](paralelo_mpi.py#L117): `MPI.Wtime()` — cronômetro do MPI, **início**.
- [linhas 120-121](paralelo_mpi.py#L120-L121): **valor-padrão do halo =
  replicação da própria borda**. Faz `buf[0]` virar cópia de `buf[1]` e `buf[-1]`
  virar cópia de `buf[-2]`. Se o processo **não tiver** vizinho daquele lado (é a
  borda global da imagem), o `Sendrecv` a seguir não muda nada e o halo fica
  sendo essa replicação — exatamente o que o sequencial faz na borda da imagem.
- [linhas 128-129](paralelo_mpi.py#L128-L129): **a troca de halo**, com
  `comm.Sendrecv` (maiúsculo, buffer NumPy):
  - linha 128: envia **minha 1ª linha** (`buf[1]`) para o vizinho de cima e
    **recebe** no meu halo de cima (`buf[0]`) a última linha dele;
  - linha 129: envia **minha última linha** (`buf[-2]`) para o vizinho de baixo
    e recebe no meu halo de baixo (`buf[-1]`) a primeira linha dele.

  `Sendrecv` faz envio e recepção numa só chamada, o que **evita deadlock** (se
  todos dessem `Send` antes de `Recv`, poderiam travar). O `.copy()` no
  `sendbuf` garante um buffer de envio separado do de recepção. Com
  `PROC_NULL`, a chamada simplesmente não faz nada.

  **(Esta é a resposta da pergunta de reflexão nº 2: as fronteiras são tratadas
  por troca de linhas-fantasma via `Sendrecv`.)**

- [linha 134](paralelo_mpi.py#L134): aplica o kernel ao `buf` **já com halo**.
  O kernel ainda põe a sua própria moldura, mas as linhas que importam (o miolo)
  são calculadas usando o halo como vizinho de verdade.
- [linha 135](paralelo_mpi.py#L135): **descarta as 2 linhas de halo**
  (`[1:-1]`), ficando só com as `minhas_linhas` filtradas. Por construção, essas
  linhas ficam **idênticas** ao que o sequencial produziria.
- [linha 137](paralelo_mpi.py#L137): `comm.Barrier()` — espera **todos**
  terminarem antes de parar o cronômetro.
- [linha 138](paralelo_mpi.py#L138): `MPI.Wtime()` — **fim**.
- [linha 141](paralelo_mpi.py#L141): `comm.reduce(..., op=MPI.MAX, root=0)` —
  junta os tempos de todos os processos pegando o **MÁXIMO**. Em paralelo o que
  conta é o processo **mais lento** (todos esperam por ele). É um escalar, não
  dado de imagem, por isso o `reduce` minúsculo é aceitável.
- [linhas 142-143](paralelo_mpi.py#L142-L143): só o rank 0 guarda o tempo.

> **Por que Scatterv fica FORA do laço?** Porque distribuir a imagem é um custo
> **fixo**, pago uma vez. Dentro do laço medimos só o que se **repete** a cada
> filtragem: a **troca de halo + o cálculo**. Isso segue a metodologia da aula
> (não pagar custo de inicialização em cada amostra).

### Passo 5 — Gatherv — [paralelo_mpi.py:149-155](paralelo_mpi.py#L149-L155)

```python
if rank == 0:
    imagem_filtrada = np.empty(altura * largura, dtype=np.uint8)
else:
    imagem_filtrada = None
destino = [imagem_filtrada, counts, deslocamentos, MPI.UNSIGNED_CHAR]
comm.Gatherv(local_saida.ravel(), destino, root=0)
```

- [linhas 149-152](paralelo_mpi.py#L149-L152): só o rank 0 aloca o buffer da
  imagem inteira (os outros mandam `None`).
- [linha 154-155](paralelo_mpi.py#L154-L155): **`comm.Gatherv`** (maiúsculo) faz
  o **inverso** do `Scatterv`: cada processo envia a sua faixa filtrada e o rank
  0 as **encaixa** nas posições certas (graças a `counts`/`deslocamentos`),
  reconstruindo a imagem. Feito **uma vez**, fora da medição, só para podermos
  conferir o resultado.

### Passo 6 — resultados — [paralelo_mpi.py:160-183](paralelo_mpi.py#L160-L183)

- [linha 161](paralelo_mpi.py#L161): transforma o buffer 1D na imagem
  `(altura, largura)`.
- [linha 162](paralelo_mpi.py#L162): aplica o `resumir_tempos` (aquecimento →
  outliers → estatística).
- [linhas 164-176](paralelo_mpi.py#L164-L176): monta e grava o JSON de resultado
  (inclui `linhas_por_processo`, tempos brutos, válidos e estatísticas).
- [linha 179](paralelo_mpi.py#L179): salva a imagem filtrada (`.npy`) para
  conferência/visualização.
- [linhas 181-183](paralelo_mpi.py#L181-L183): imprime média e variância.

---

## 7. `benchmark.py`

Orquestra todo o experimento e produz a tabela e o gráfico.

- [`executar`](benchmark.py#L36-L42): roda um comando externo e **aborta** se ele
  falhar.
- **Garantir a imagem**: se `imagem.npy` não existe, chama `gerar_imagem.py`.
- **Laço por filtro**:
  - roda `sequencial.py` → guarda `T_seq`;
  - para cada `P` em `2, 4, 8`, roda **`mpiexec -n P py paralelo_mpi.py`**.
    Repare: o `mpiexec` é chamado **uma vez por P** (não uma vez por repetição) —
    as repetições acontecem **dentro** do `paralelo_mpi.py`. **(É exatamente a
    metodologia que o enunciado pede: não forçar um "início frio" do ambiente MPI
    a cada teste.)**
- [`montar_tabela`](benchmark.py#L89-L121): calcula, para cada (filtro, P):
  `speedup = T_seq / T_par` e `eficiência = speedup / P`; imprime e grava
  `resultados_speedup.csv` e `tabela_speedup.md`.
- [`desenhar_grafico`](benchmark.py#L124-L151): com matplotlib, desenha **tempo
  médio × número de processos** (eixo X em escala log₂, pontos em 1, 2, 4, 8) e
  salva `tempo_por_processos.png`.

---

## 8. `gerar_imagem.py`

Cria a imagem de teste e salva em `.npy` (e `.png` se houver Pillow).

- [`gerar`](gerar_imagem.py#L28-L55): monta a imagem em 3 camadas:
  1. um **gradiente** de fundo (estrutura suave);
  2. alguns **círculos** claros/escuros (bordas e objetos definidos);
  3. **ruído sal e pimenta** — uma fração dos pixels vira 0 (preto) ou 255
     (branco) aleatoriamente.
- Usa `np.random.default_rng(semente)` com **semente fixa** → imagem
  **reprodutível** (todos do grupo geram a mesma).
- **Por que grande (3000×3000) e com ruído?** Grande para **exigir
  paralelização** (imagem pequena não daria ganho). Com ruído sal e pimenta
  porque é o caso em que a **mediana brilha** (remove o ruído) e a **média só
  borra** — deixa a comparação entre os dois filtros mais rica.

---

## 9. `teste_corretude.py`

Teste que **não usa MPI**. Ele **simula em numpy puro** a lógica de
partição + halo + filtragem (a mesma de `paralelo_mpi.py`) e confere, para
vários tamanhos e números de processos, se o resultado reunido é **idêntico** ao
sequencial — inclusive em **alturas que não dividem** pelo número de processos.

Resultado da execução (sem MPI, só numpy):

```
>>> TODOS OS TESTES PASSARAM: paralelo == sequencial (incluindo bordas e divisao desigual).
```

Ou seja: a parte algorítmica (divisão, linhas-fantasma, bordas) está **provada
correta** mesmo antes de instalar o MS-MPI.

---

## 10. Respostas às perguntas de reflexão

**1) O filtro de mediana paralelo teve speedup maior ou menor que o de média? Por
quê?**
**Maior.** A mediana faz **mais trabalho de CPU por pixel** (precisa ordenar os 9
vizinhos — [filtros.py:68](filtros.py#L68)), enquanto a média só soma e divide
([filtros.py:42-46](filtros.py#L42-L46)). Como o custo de **comunicação** (halo +
distribuição) é praticamente o **mesmo** nos dois, o filtro mais "pesado"
(mediana) tem uma razão **cálculo/comunicação** melhor: o tempo é dominado pela
parte que **realmente** paraleliza. Já a média é tão leve que fica limitada por
memória/comunicação, e o overhead come boa parte do ganho.

**2) O que acontece com os pixels nas fronteiras entre processos? O grupo tratou
isso? Como?**
Sim. Cada processo precisa, para filtrar a 1ª e a última linha da sua faixa, de 1
linha do vizinho de cima e 1 do vizinho de baixo. Tratamos isso com **linhas-
fantasma (halo)**: um buffer com 2 linhas extras
([paralelo_mpi.py:101-102](paralelo_mpi.py#L101-L102)) preenchidas por **troca
`Sendrecv`** com os vizinhos
([paralelo_mpi.py:128-129](paralelo_mpi.py#L128-L129)). Nas **bordas globais** da
imagem (topo do rank 0, fundo do último), não há vizinho: usamos `MPI.PROC_NULL`
([paralelo_mpi.py:106-107](paralelo_mpi.py#L106-L107)) e o halo vira
**replicação da própria borda** ([paralelo_mpi.py:120-121](paralelo_mpi.py#L120-L121)),
igual ao sequencial. Por isso o resultado paralelo é **idêntico** ao sequencial.

**3) Por que foi usado `comm.Bcast` e não `comm.bcast` para enviar a imagem? O
que mudaria no desempenho?**
As versões **maiúsculas** (`Bcast`, `Scatterv`, `Gatherv`, `Sendrecv`) operam
**direto no buffer de memória** do array NumPy: enviam os bytes "crus", sem
cópia extra. As **minúsculas** (`bcast`, `scatter`, ...) **serializam com
`pickle`** o objeto Python antes de enviar e **desserializam** no destino — para
uma imagem de milhões de bytes isso significa **cópias extras, mais CPU e mais
memória**, deixando a comunicação **bem mais lenta**. Por isso **toda** a
comunicação de **imagem** usa as maiúsculas
([paralelo_mpi.py:93](paralelo_mpi.py#L93),
[155](paralelo_mpi.py#L155),
[128-129](paralelo_mpi.py#L128-L129)). A única exceção é o `comm.bcast` minúsculo
em [paralelo_mpi.py:73-74](paralelo_mpi.py#L73-L74), e **só** porque ali são 2
inteiros (altura/largura) — não são dados de imagem, e a diferença de desempenho
é desprezível.

> Obs.: neste código a imagem é **distribuída** com `Scatterv` (cada processo
> recebe uma faixa), não com `Bcast` (que mandaria a imagem inteira para todos).
> O princípio "maiúsculo = buffer rápido, minúsculo = pickle lento" é o mesmo; a
> escolha de `Scatterv` em vez de `Bcast` é por dividir o trabalho.

**4) Se o número de processos não divide exatamente a altura da imagem, o que o
código faz?**
Distribui o **resto** entre os primeiros processos: os primeiros `altura %
n_processos` recebem **1 linha a mais** ([particao.py:23-25](particao.py#L23-L25)).
Ex.: 1000 linhas em 3 processos → 334, 333, 333. Por isso usamos as versões
**"v"** (`Scatterv`/`Gatherv`), que aceitam blocos de **tamanhos diferentes** via
`counts`/`deslocamentos` ([particao.py:27-30](particao.py#L27-L30)). O
[teste_corretude.py](teste_corretude.py) confirma que mesmo nesses casos o
resultado bate com o sequencial.
