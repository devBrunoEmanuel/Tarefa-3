# Explicação das Tecnologias e Decisões do Projeto

## Bibliotecas e ferramentas usadas

### `mpi4py` — MPI para Python

**O quê:** Interface Python para o padrão MPI (Message Passing Interface), que permite que múltiplos processos se comuniquem trocando mensagens.

**Por quê:** É o requisito central da atividade — o enunciado pede paralelização com MPI. O `mpi4py` é o binding oficial de MPI para Python e é o que o professor usa nos slides.

---

### `comm.Bcast` (maiúsculo) vs `comm.bcast` (minúsculo)

**O quê:** Dois modos de broadcast no `mpi4py`.

**Por quê a diferença importa:**
- `bcast` (minúsculo) → serializa o objeto com pickle. Lento para arrays grandes.
- `Bcast` (maiúsculo) → opera diretamente no buffer de memória do NumPy, sem pickle. Muito mais rápido para imagens (a "Regra de ouro" da página 15 dos slides).

Para as dimensões (2 inteiros), `bcast` está ok. Para a imagem inteira (9 milhões de pixels), `Bcast` é obrigatório.

---

### `comm.Gatherv` — Coleta com tamanhos variáveis

**O quê:** Reúne blocos de diferentes tamanhos de cada processo no processo 0.

**Por quê:** Quando a altura não é divisível pelo número de processos, cada processo tem um número diferente de linhas. O `Gatherv` suporta isso via os arrays `counts` e `deslocamentos` calculados em `particao.py`.

---

### `comm.reduce(MPI.MAX)` — Tempo do processo mais lento

**O quê:** Reduz os tempos de todos os processos pegando o máximo.

**Por quê:** O tempo real de uma iteração paralela é ditado pelo processo mais lento — os outros ficam esperando no `Barrier()`. Pegar a média seria enganoso.

---

### `comm.Barrier()` — Sincronização

**O quê:** Faz todos os processos esperarem até que o último chegue naquele ponto.

**Por quê:** Para medir o tempo com precisão: o cronômetro começa e termina com todos os processos sincronizados, garantindo que a medição captura o trabalho real e não diferenças de escalonamento do SO.

---

### `numpy` — Operações vetorizadas

**O quê:** Biblioteca de arrays numéricos com operações vetorizadas (sem laços Python explícitos).

**Por quê:** Os filtros em `filtros.py` precisam processar milhões de pixels. Laços Python seriam proibitivamente lentos. Com NumPy, o filtro de média soma 9 fatias do array de uma vez; o de mediana empilha as 9 vizinhanças em um array 3D e chama `np.median` uma vez — tudo em C por baixo.

---

### `np.pad(mode="edge")` — Replicação de borda

**O quê:** Adiciona uma borda ao redor da imagem replicando os pixels da borda.

**Por quê:** Filtros 3×3 precisam de vizinhos para todo pixel, inclusive os das bordas. Com padding, qualquer pixel tem sempre 9 vizinhos válidos, eliminando tratamento especial de borda.

---

### `cv2` (OpenCV) — Leitura e escrita de imagens

**O quê:** Biblioteca de visão computacional usada aqui apenas para `imread`/`imwrite`.

**Por quê:** O professor indicou nos slides (página 9: "pip install opencv-python"). Antes o projeto usava `np.save/.npy` (formato binário NumPy), mas foi migrado para PNG/OpenCV para seguir o material da aula.

---

### Halo de linhas (fatiamento local)

**O quê:** Cada processo pega 1 linha extra acima e abaixo da sua fatia antes de filtrar (`img[halo_top:halo_bot]`).

**Por quê:** Filtros 3×3 nas bordas da fatia precisam de pixels dos processos vizinhos. Como todos já têm a imagem completa via `Bcast`, buscar essa linha vizinha é apenas um fatiamento local — não precisa de `Sendrecv` (troca ponto-a-ponto). Isso resolve a "limitação nas bordas" mencionada na página 13 dos slides.

---

### IQR para remoção de outliers (`bench_utils.py`)

**O quê:** Método estatístico que define como outlier qualquer amostra fora de `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`.

**Por quê:** Picos ocasionais de tempo (GC, preempção do SO, cache frio) distorcem a média. O IQR é robusto porque não usa a média para decidir o que é outlier — ele usa os quartis.

---

### Warm-up (aquecimento)

**O quê:** As 2 primeiras repetições são descartadas antes de ligar o cronômetro.

**Por quê:** A primeira execução sofre com cache de CPU frio e alocações preguiçosas de memória, produzindo tempos artificialmente altos. Descartar essas medições dá uma visão justa do desempenho em regime permanente (metodologia da página 16 dos slides).

---

### `particao.py` — Módulo separado para divisão de linhas

**O quê:** Calcula quantas linhas cada processo recebe, tratando o caso em que a altura não é divisível pelo número de processos.

**Por quê:** Ficar em módulo separado (sem import de MPI) permite testar a lógica de partição em `teste_corretude.py` sem precisar do ambiente MPI instalado, além de manter o código de negócio separado da infraestrutura de comunicação.
