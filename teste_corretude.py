"""
teste_corretude.py
==================

Teste que NAO usa MPI. Ele simula em numpy puro exatamente a logica de particao
+ halo + filtragem da versao paralela (paralelo_mpi.py) e confere se o resultado
reunido fica IDENTICO ao da versao sequencial.

Isso valida a parte algoritmica (divisao por linhas, linhas-fantasma, bordas)
mesmo sem ter o ambiente MPI instalado. Rode com:

    py teste_corretude.py
"""

import numpy as np

from filtros import FILTROS
from particao import calcular_particao


def parperalelo_simulado(img, kernel, n_processos):
    """Reproduz, sem MPI, o que paralelo_mpi.py faz com n_processos."""
    altura, largura = img.shape
    linhas_por_processo, counts, deslocamentos = calcular_particao(altura, n_processos, largura)

    blocos_saida = []
    inicio = 0
    for r in range(n_processos):
        minhas_linhas = linhas_por_processo[r]
        if minhas_linhas == 0:
            continue
        bloco_local = img[inicio:inicio + minhas_linhas]

        # Buffer com halo (2 linhas extras), como em paralelo_mpi.py
        buf = np.empty((minhas_linhas + 2, largura), dtype=np.uint8)
        buf[1:-1] = bloco_local

        # Halo de cima: linha do vizinho de cima OU replicacao (borda global)
        buf[0] = img[inicio - 1] if r > 0 else bloco_local[0]
        # Halo de baixo: linha do vizinho de baixo OU replicacao (borda global)
        fim = inicio + minhas_linhas
        buf[-1] = img[fim] if r < n_processos - 1 else bloco_local[-1]

        saida_com_halo = kernel(buf)
        blocos_saida.append(saida_com_halo[1:-1])
        inicio += minhas_linhas

    return np.vstack(blocos_saida)


def main():
    rng = np.random.default_rng(0)
    falhas = 0

    # Varios tamanhos, incluindo alturas que NAO dividem pelo numero de processos.
    for (altura, largura) in [(10, 7), (101, 33), (256, 100), (333, 64)]:
        img = rng.integers(0, 256, size=(altura, largura), dtype=np.uint8)
        for nome, kernel in FILTROS.items():
            esperado = kernel(img)  # sequencial
            for n_processos in [1, 2, 3, 4, 8]:
                if n_processos > altura:
                    continue
                obtido = parperalelo_simulado(img, kernel, n_processos)
                if np.array_equal(obtido, esperado):
                    status = "OK  "
                else:
                    status = "FALHA"
                    falhas += 1
                print(f"{status} filtro={nome:8s} img={altura}x{largura} P={n_processos}")

    print()
    if falhas == 0:
        print(">>> TODOS OS TESTES PASSARAM: paralelo == sequencial (incluindo bordas e divisao desigual).")
    else:
        print(f">>> {falhas} TESTE(S) FALHARAM.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
