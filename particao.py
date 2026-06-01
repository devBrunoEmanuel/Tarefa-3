"""
particao.py
===========

Calcula como a imagem e dividida entre os processos. Fica em um modulo separado
(sem dependencia de MPI) para poder ser testado isoladamente, sem precisar do
ambiente MPI instalado (ver teste_corretude.py).
"""


def calcular_particao(altura, n_processos, largura):
    """Decide quantas linhas cada processo recebe e onde cada bloco comeca.

    Devolve (linhas_por_processo, counts, deslocamentos):
      - linhas_por_processo[r] : numero de linhas do processo r
      - counts[r]              : numero de ELEMENTOS (linhas*largura) do processo r
      - deslocamentos[r]       : posicao inicial (em elementos) do bloco de r

    Se a altura nao for divisivel pelo numero de processos, os primeiros
    (altura % n_processos) processos recebem 1 linha a mais. Ex.: 1000 linhas em
    3 processos -> 334, 333, 333.
    """
    base = altura // n_processos
    resto = altura % n_processos
    linhas_por_processo = [base + (1 if r < resto else 0) for r in range(n_processos)]

    counts = [linhas * largura for linhas in linhas_por_processo]
    deslocamentos = [0] * n_processos
    for r in range(1, n_processos):
        deslocamentos[r] = deslocamentos[r - 1] + counts[r - 1]

    return linhas_por_processo, counts, deslocamentos
