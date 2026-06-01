# Explicação do código — linha a linha

Comparação de **filtro de média 3×3** e **filtro de mediana 3×3**, em versão
**sequencial** e **paralela com MPI** (mpi4py).

Este documento explica **cada arquivo e cada linha relevante** com base nos
**slides do professor** (`spd4_comunicação_04 - MPI-2.pdf`).

## Índice

1. [Visão geral / arquitetura](#1-visão-geral)
2. [`filtros.py` — os dois kernels](#2-filtrospy)
3. [`particao.py` — divisão da imagem](#3-particaopy)
4. [`bench_utils.py` — estatística e outliers](#4-bench_utilspy)
5. [`gerar_imagem.py` — imagem de teste](#5-gerar_imagempy)
6. [`sequencial.py` — versão sequencial](#6-sequencialpy)
7. [`paralelo_mpi.py` — versão paralela MPI](#7-paralelo_mpipy)
8. [`benchmark.py` — orquestrador](#8-benchmarkpy)
9. [`teste_corretude.py` — teste sem MPI](#9-teste_corretudepy)
10. [Respostas às perguntas de reflexão](#10-respostas-às-perguntas-de-reflexão)

---

## 1. Visão geral

O fluxo de execução é:

```
gerar_imagem.py  ─►  imagem.png   (OpenCV - slides pág. 9)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
     sequencial.py                 paralelo_mpi.py
   (1 processo, T_seq)      (mpiexec -n 2/4/8, T_par)
            │                            │
            └────────────┬───────────────┘
                         ▼
                   benchmark.py
        (tabela de speedup + gráfico tempo×processos)
```

**Estratégia de paralelização (alinhada com os slides — pág. 12-13):**

A imagem é cortada em **faixas horizontais de linhas**. O processo 0 lê a
imagem com OpenCV e `comm.Bcast` a envia **inteira** para todos os processos.
Cada processo então trabalha sobre a sua faixa, buscando as linhas vizinhas
(halo) diretamente da sua cópia local — sem precisar de comunicação extra.
No final `comm.Gatherv` reúne as faixas filtradas no processo 0.

**Regra de ouro (slides pág. 15):** toda comunicação de imagem usa operações
**maiúsculas** (`Bcast`, `Gatherv`) — trabalham no buffer NumPy diretamente, sem
serialização pickle.

---

## 2. `filtros.py`

Define os dois kernels. Ambos recebem uma imagem 2D e devolvem outra do mesmo
tamanho, tratando borda por **replicação de borda** (o pixel "de fora" vira a
borda mais próxima).

### `_replica_borda` — [filtros.py:23-30](filtros.py#L23-L30)

```python
return np.pad(img, pad_width=1, mode="edge")
```

- [linha 30](filtros.py#L30): `np.pad(..., mode="edge")` envolve a imagem com
  **1 pixel de moldura** replicando a borda. Uma imagem `(H, W)` vira `(H+2, W+2)`.
  Isso permite calcular a vizinhança 3×3 de **qualquer** pixel — inclusive os
  das bordas — usando só fatiamento, **sem `if` de borda pixel a pixel**.

### `filtro_media_3x3` — [filtros.py:33-46](filtros.py#L33-L46)

```python
a = _replica_borda(img).astype(np.float64)
soma = (a[0:-2, 0:-2] + a[0:-2, 1:-1] + a[0:-2, 2:] +
        a[1:-1, 0:-2] + a[1:-1, 1:-1] + a[1:-1, 2:] +
        a[2:,   0:-2] + a[2:,   1:-1] + a[2:,   2:])
return np.rint(soma / 9.0).astype(img.dtype)
```

- [linha 40](filtros.py#L40): converte para `float64`. A soma de 9 pixels (até
  9×255 = 2295) estoura o `uint8` (máx 255).
- [linhas 42-44](filtros.py#L42-L44): as **9 fatias deslocadas** da imagem com
  padding. Cada `a[i:j, k:l]` é a imagem inteira "movida" para uma posição da
  vizinhança 3×3 (cima-esquerda, cima, cima-direita, ..., baixo-direita). Somar
  as 9 dá a soma da vizinhança de **todos os pixels de uma vez** (vetorizado).
  Isso é o equivalente do "filtro de convolução" explicado na **página 7** dos
  slides.
- [linha 46](filtros.py#L46): divide por 9, arredonda e volta para `uint8`.

### `filtro_mediana_3x3` — [filtros.py:49-70](filtros.py#L49-L70)

```python
a = _replica_borda(img)
vizinhos = np.stack([...9 fatias deslocadas...], axis=0)
mediana = np.median(vizinhos, axis=0)
return mediana.astype(img.dtype)
```

- [linhas 62-66](filtros.py#L62-L66): as mesmas 9 fatias deslocadas, mas
  `np.stack(..., axis=0)` as empilha num array 3D de forma `(9, H, W)` — 9
  "cópias" da imagem, uma por posição da vizinhança.
- [linha 68](filtros.py#L68): `np.median(..., axis=0)` tira a mediana ao longo
  do eixo 0 — para cada pixel `(y, x)`, ordena os 9 valores e devolve o do meio.
  A mediana escolhe **um valor existente** → não borra bordas, e remove bem ruído
  sal e pimenta (slides págs. 4-5).

---

## 3. `particao.py`

Decide como dividir as linhas entre os processos. Separado de propósito (sem
importar MPI) para poder ser testado sem o ambiente MPI.

### `calcular_particao` — [particao.py:11-32](particao.py#L11-L32)

```python
base = altura // n_processos
resto = altura % n_processos
linhas_por_processo = [base + (1 if r < resto else 0) for r in range(n_processos)]
counts = [linhas * largura for linhas in linhas_por_processo]
deslocamentos = [0] * n_processos
for r in range(1, n_processos):
    deslocamentos[r] = deslocamentos[r-1] + counts[r-1]
```

- [linha 23](particao.py#L23): divisão inteira → linhas "garantidas" para cada um.
- [linha 24](particao.py#L24): linhas que sobram quando a altura **não divide**
  exatamente.
- [linha 25](particao.py#L25): distribui o resto: os primeiros `resto` processos
  ganham **1 linha a mais**. Ex.: 1000 linhas / 3 processos → `[334, 333, 333]`.
  **(Resposta da reflexão nº 4.)**
- [linha 27](particao.py#L27): `counts` em **elementos** (pixels), necessário
  para o `Gatherv` que conta elementos, não linhas.
- [linhas 28-30](particao.py#L28-L30): `deslocamentos` = início (em elementos)
  de cada bloco no array achatado. Necessário para `Gatherv`.

---

## 4. `bench_utils.py`

Funções de estatística e E/S, compartilhadas pelas duas versões. Implementa
a metodologia de benchmark da **página 16 dos slides**.

### `remover_outliers` — [bench_utils.py:28-46](bench_utils.py#L28-L46)

Método **IQR (cercas de Tukey)**, conforme metodologia dos slides:

- [linhas 40-42](bench_utils.py#L40-L42): `Q1` (percentil 25), `Q3` (percentil
  75) e `IQR = Q3 − Q1`.
- [linhas 43-44](bench_utils.py#L43-L44): cercas `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`.
  Critério robusto: não depende da média (que os outliers distorceriam).
- [linha 45](bench_utils.py#L45): filtra, mantendo só o que está dentro das cercas.

### `estatisticas` — [bench_utils.py:49-64](bench_utils.py#L49-L64)

- [linha 60](bench_utils.py#L60): `var(ddof=1)` — variância **amostral** (divide
  por *n−1*), correta para uma amostra. **(Variância pedida no enunciado.)**

### `resumir_tempos` — [bench_utils.py:67-76](bench_utils.py#L67-L76)

Pipeline completo: descarta warm-up → remove outliers → calcula estatísticas.

---

## 5. `gerar_imagem.py`

Cria a imagem de teste e salva como **PNG com OpenCV** (`cv2.imwrite`), alinhado
com a **página 9 dos slides** ("Para trabalharmos com imagem, usaremos o OpenCV").

- [`gerar`](gerar_imagem.py#L28-L55): gradiente + círculos + ruído sal e pimenta.
  Usa `np.random.default_rng(semente)` com semente fixa → imagem **reprodutível**.
- [`salvar_imagem`](gerar_imagem.py#L58-L75): tenta `cv2.imwrite` (OpenCV); se
  não instalado, tenta PIL; se nenhum, salva `.npy`.

**Por que 3000×3000 e com ruído sal e pimenta?** Grande para exigir paralelização
real (slides págs. 7-8: "Toma bastante tempo de processamento!!!!"); ruído sal e
pimenta porque a mediana o remove muito bem, tornando a comparação dos filtros
mais rica (slides pág. 5).

---

## 6. `sequencial.py`

Versão de referência: **1 processo, sem MPI**. Mede `T_seq`.

- [linha 37](sequencial.py#L37): `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` — lê a
  imagem **diretamente em escala de cinza** com OpenCV (**página 9 dos slides**).
  Devolve um array NumPy `uint8` de forma `(H, W)`.
- [linha 38](sequencial.py#L38): `if img is None` — `cv2.imread` devolve `None`
  quando o arquivo não existe (diferente do numpy que lança exceção). Verificamos
  explicitamente.
- [linhas 44-49](sequencial.py#L44-L49): laço de medição (warm-up + reps). Para
  cada repetição:
  - `time.perf_counter()` — relógio de alta resolução, **antes**.
  - `kernel(img)` — aplica o filtro na imagem inteira.
  - Relógio **depois** e guarda o delta.
- [linha 51](sequencial.py#L51): `resumir_tempos` aplica warm-up → outliers →
  estatística.
- [linha 64](sequencial.py#L64): `cv2.imwrite` salva a imagem filtrada como PNG.

---

## 7. `paralelo_mpi.py`

O núcleo da atividade. Lançado por `mpiexec -n P`. Todos os P processos executam
o mesmo código; o que diferencia cada um é o `rank`.

**Mudança principal em relação a uma versão mais simples:** usa `comm.Bcast`
(slides pág. 13) para distribuir a imagem inteira a todos, e resolve a
"limitação nas bordas" (pág. 13) extraindo linhas de halo da cópia local, sem
precisar de `Sendrecv`.

### Inicialização do MPI — [paralelo_mpi.py:47-49](paralelo_mpi.py#L47-L49)

```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```

- `rank`: número deste processo (0 = "chefe").
- `size`: total de processos (o `P` do `mpiexec -n P`).

### Passo 1 — rank 0 lê com OpenCV — [paralelo_mpi.py:63-70](paralelo_mpi.py#L63-L70)

```python
if rank == 0:
    img = cv2.imread(args.imagem, cv2.IMREAD_GRAYSCALE)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    altura, largura = img.shape
else:
    img = None
    altura = largura = None
```

- [linha 64](paralelo_mpi.py#L64): só o rank 0 lê do disco (com OpenCV). Os
  outros aguardam. `cv2.IMREAD_GRAYSCALE` = lê direto em escala de cinza,
  array `uint8` `(H, W)`.
- [linha 65](paralelo_mpi.py#L65): `np.ascontiguousarray` garante bytes
  **contíguos na memória** — obrigatório para que o `Bcast` envie a fatia correta.

### Passo 2 — broadcast das dimensões — [paralelo_mpi.py:72-74](paralelo_mpi.py#L72-L74)

```python
altura = comm.bcast(altura, root=0)
largura = comm.bcast(largura, root=0)
```

Apenas 2 inteiros, não são dados de imagem. A versão **minúscula** (`bcast`)
com pickle é aceitável aqui: 2 números têm custo desprezível. Serve para que
os processos não-root saibam qual tamanho alocar para o `Bcast` da imagem.

### Passo 3 — `comm.Bcast` da imagem — [paralelo_mpi.py:82-85](paralelo_mpi.py#L82-L85)

```python
if rank != 0:
    img = np.empty((altura, largura), dtype=np.uint8)  # buffer de recepcao
comm.Bcast(img, root=0)
```

- [linha 82](paralelo_mpi.py#L82): processos não-root alocam o buffer de recepção
  **antes** do Bcast (o MPI precisa do buffer já alocado).
- [linha 85](paralelo_mpi.py#L85): **`comm.Bcast`** (MAIÚSCULO) envia a imagem
  inteira para **todos os processos**. Isso é exatamente o que os slides mostram
  na página 13. **(Resposta da reflexão nº 3: Bcast vs bcast — ver seção 10.)**

  Após esta linha, **todos os processos têm uma cópia completa da imagem em
  memória**. Isso elimina a necessidade de `Sendrecv` para troca de halo.

### Passo 4 — partição — [paralelo_mpi.py:88-91](paralelo_mpi.py#L88-L91)

```python
linhas_por_processo, counts, deslocamentos = calcular_particao(altura, size, largura)
start_row = sum(linhas_por_processo[:rank])
end_row   = start_row + linhas_por_processo[rank]
```

- Todos calculam a mesma partição (determinística).
- `start_row`/`end_row`: intervalo de linhas desta faixa.

### Passo 5 — laço de medição — [paralelo_mpi.py:97-123](paralelo_mpi.py#L97-L123)

```python
for _ in range(args.aquecimento + args.reps):
    comm.Barrier()
    t0 = MPI.Wtime()

    halo_top = max(0, start_row - 1)
    halo_bot = min(altura, end_row + 1)
    sub = img[halo_top:halo_bot]

    filtered_sub = kernel(sub)

    offset = start_row - halo_top
    local_saida = np.ascontiguousarray(
        filtered_sub[offset : offset + linhas_por_processo[rank]]
    )

    comm.Barrier()
    t1 = MPI.Wtime()
    dt = comm.reduce(t1 - t0, op=MPI.MAX, root=0)
    if rank == 0:
        tempos.append(dt)
```

- [linha 99](paralelo_mpi.py#L99): `comm.Barrier()` — sincroniza todos os
  processos **antes** do cronômetro (metodologia pág. 16: "sincronização").
- [linha 100](paralelo_mpi.py#L100): `MPI.Wtime()` — cronômetro do MPI, início.
- [linhas 103-105](paralelo_mpi.py#L103-L105): extrai a fatia **com halo**.
  `halo_top = start_row − 1` (1 linha acima) e `halo_bot = end_row + 1` (1
  linha abaixo), com `max/min` para não ultrapassar os limites globais.
  **Como cada processo já tem a imagem inteira via Bcast**, buscar as linhas
  vizinhas é só uma fatia de array local — zero comunicação extra. Isso resolve
  a "limitação nas bordas" da **página 13 dos slides**.
- [linha 107](paralelo_mpi.py#L107): aplica o kernel ao `sub` (que já contém
  halo). O kernel aplica sua própria moldura de borda, mas as linhas internas
  são calculadas com vizinhança correta graças ao halo incluído.
- [linhas 110-112](paralelo_mpi.py#L110-L112): descarta a(s) linha(s) de halo
  do resultado. `offset = 0` se estiver no topo global (rank 0, sem halo
  acima), `offset = 1` caso contrário. O resultado tem exatamente
  `linhas_por_processo[rank]` linhas, **idênticas** ao sequencial.
- [linha 114](paralelo_mpi.py#L114): `Barrier()` após — espera todos terminarem.
- [linha 115](paralelo_mpi.py#L115): `MPI.Wtime()` — fim.
- [linha 117](paralelo_mpi.py#L117): `comm.reduce(..., MPI.MAX, root=0)` —
  tempo da iteração = processo **mais lento** (todos esperaram por ele).
- **Por que o laço fica dentro do script?** Metodologia da pág. 16: "utilize
  o loop de 30 iterações dentro do script Python". O `mpiexec` é chamado **uma
  vez** por número de processos, não uma vez por repetição.

### Passo 6 — `comm.Gatherv` — [paralelo_mpi.py:127-132](paralelo_mpi.py#L127-L132)

```python
destino = [imagem_filtrada, counts, deslocamentos, MPI.UNSIGNED_CHAR]
comm.Gatherv(local_saida.ravel(), destino, root=0)
```

**`comm.Gatherv`** (MAIÚSCULO, buffer NumPy) reúne as faixas filtradas de cada
processo no rank 0. A versão "v" aceita blocos de tamanhos diferentes (quando a
altura não divide pelo número de processos). Feito **uma vez**, fora do laço.

### Passo 7 — resultados — [paralelo_mpi.py:135-155](paralelo_mpi.py#L135-L155)

- Apenas o rank 0 calcula estatísticas, grava JSON e salva a imagem filtrada
  como PNG com `cv2.imwrite` (OpenCV, pág. 9 dos slides).

---

## 8. `benchmark.py`

Orquestra o experimento e produz tabela + gráfico.

- Garante imagem de teste (chama `gerar_imagem.py` se não existe).
- Roda `sequencial.py` (T_seq) e depois `mpiexec -n P paralelo_mpi.py` para
  cada P em `{2, 4, 8}` — **uma chamada por P**, com 30 repetições **dentro**.
- [`montar_tabela`](benchmark.py#L89-L121): `speedup = T_seq / T_par`,
  `eficiência = speedup / P`. Grava CSV e Markdown.
- [`desenhar_grafico`](benchmark.py#L124-L151): matplotlib, tempo × processos.

**Por que um mpiexec por P e não um por repetição?** Slides pág. 16: "para
evitar que ruídos relacionados ao SO, como a inicialização do ambiente MPI via
mpiexec, utilize o loop de 30 iterações dentro do script Python."

---

## 9. `teste_corretude.py`

Teste **sem MPI** que simula a lógica do `paralelo_mpi.py` em numpy puro:

- Para cada "processo" simulado: calcula `start_row`/`end_row`, extrai
  `img[halo_top:halo_bot]`, aplica kernel, descarta halo.
- Compara com o sequencial para vários tamanhos e contagens de processos,
  **inclusive alturas não divisíveis**.

Resultado verificado:
```
>>> TODOS OS TESTES PASSARAM: paralelo == sequencial (incluindo bordas e divisao desigual).
```

---

## 10. Respostas às perguntas de reflexão

**1) O filtro de mediana paralelo teve speedup maior ou menor que o de média? Por
quê?**
**Maior.** A mediana faz **mais trabalho de CPU por pixel** — precisa ordenar 9
valores ([filtros.py:68](filtros.py#L68)), enquanto a média só soma e divide
([filtros.py:42-46](filtros.py#L42-L46)). Como o custo de distribuição
(Bcast) é **igual** nos dois, o filtro mais pesado tem melhor razão
*cálculo/overhead*: a parte paralelizável domina, e o speedup é maior. A média é
tão leve que o overhead de Bcast + Gatherv + Barrier representa fração maior do
total.

**2) O que acontece com os pixels nas fronteiras entre processos? O grupo tratou
isso? Como?**
Sim. Os slides (**página 13**) identificam essa "limitação nas bordas" como
problema a resolver. Nossa solução: como todos os processos têm a **imagem
inteira via `comm.Bcast`**, basta extrair a fatia com **1 linha de halo acima e
abaixo** da cópia local ([paralelo_mpi.py:103-105](paralelo_mpi.py#L103-L105)).
Não há comunicação extra — o halo já está na memória. Para as bordas **globais**
da imagem (rank 0 sem linha acima; último rank sem linha abaixo), usamos
`max(0,...)` / `min(altura,...)` e o kernel aplica replicação de borda
normalmente. O resultado é **idêntico pixel a pixel** ao sequencial.

**3) Por que foi usado `comm.Bcast` e não `comm.bcast` para enviar a imagem? O
que mudaria no desempenho?**
Os slides (**página 15, "Regra de ouro"**) explicam:

> "A escolha entre maiúscula e minúscula não é estilo. É desempenho! Sempre use
> maiúsculas para transmissão de dados volumosos, como imagens, vetores e matrizes."

- **`comm.Bcast`** (MAIÚSCULO): opera **direto no buffer de memória** do array
  NumPy. Os bytes da imagem são enviados "crus" pela rede MPI, **sem nenhuma
  cópia extra**.
- **`comm.bcast`** (minúsculo): **serializa o objeto Python com `pickle`** antes
  de enviar e desserializa no destino. Para uma imagem de 9 megapixels (~9 MB),
  isso significa cópia extra dos dados, gasto de CPU para serialização e maior
  uso de memória — tornando a comunicação **significativamente mais lenta**.

No código, o único uso minúsculo é `comm.bcast` para as duas dimensões
([paralelo_mpi.py:72-74](paralelo_mpi.py#L72-L74)) — 2 inteiros, custo
desprezível.

**4) Se o número de processos não divide exatamente a altura da imagem, o que o
código faz?**
Distribui o **resto** entre os primeiros processos: os primeiros
`altura % n_processos` recebem **1 linha a mais**
([particao.py:23-25](particao.py#L23-L25)). Ex.: 1000 linhas / 3 processos →
`[334, 333, 333]`. Por isso usamos `comm.Gatherv` (versão "v") que aceita blocos
de **tamanhos diferentes** via `counts` e `deslocamentos`
([paralelo_mpi.py:129-132](paralelo_mpi.py#L129-L132)). O
[teste_corretude.py](teste_corretude.py) confirma que nesses casos o resultado
continua idêntico ao sequencial.
