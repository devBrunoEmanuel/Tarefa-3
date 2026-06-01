"""
teste_corretude.py
==================

Teste que NAO usa MPI. Ele simula em numpy puro exatamente a logica que
paralelo_mpi.py executa com o MPI e confere se o resultado reunido fica
IDENTICO ao da versao sequencial.

Logica simulada (alinhada com a versao Bcast do paralelo_mpi.py):
    1. Todos os processos tem a imagem inteira (simulando o comm.Bcast).
    2. Cada "processo" calcula start_row/end_row via calcular_particao.
    3. Extrai a fatia img[halo_top:halo_bot] (com 1 linha de halo em cada borda).
    4. Aplica o kernel e descarta a(s) linha(s) de halo do resultado.

Isso valida que:
    - A divisao por linhas (incluindo alturas nao divisiveis) esta correta.
    - A logica de halo resolve a "limitacao nas bordas" da pagina 13 dos slides.
    - O resultado paralelo e identico ao sequencial.

Rode com:
    py teste_corretude.py
"""

import numpy as np

from filtros import FILTROS
from particao import calcular_particao


def paralelo_simulado(img, kernel, n_processos):
    """Simula o que paralelo_mpi.py faz com n_processos, sem MPI."""
    altura, largura = img.shape
    linhas_por_processo, counts, deslocamentos = calcular_particao(altura, n_processos, largura)

    blocos_saida = []
    for r in range(n_processos):
        start_row = sum(linhas_por_processo[:r])
        end_row   = start_row + linhas_por_processo[r]

        # Extrai fatia COM halo (1 linha acima e abaixo quando existem),
        # exatamente como paralelo_mpi.py faz apos receber a imagem via Bcast.
        halo_top = max(0, start_row - 1)
        halo_bot = min(altura, end_row + 1)
        sub = img[halo_top:halo_bot]

        filtered_sub = kernel(sub)

        # Descarta linha(s) de halo; guarda apenas as linhas desta particao.
        offset = start_row - halo_top  # 0 se borda global do topo; 1 caso contrario
        blocos_saida.append(filtered_sub[offset : offset + linhas_por_processo[r]])

    return np.vstack(blocos_saida)


def main():
    rng = np.random.default_rng(0)
    falhas = 0

    # Varios tamanhos, incluindo alturas que NAO dividem pelo numero de processos.
    for (altura, largura) in [(10, 7), (101, 33), (256, 100), (333, 64)]:
        img = rng.integers(0, 256, size=(altura, largura), dtype=np.uint8)
        for nome, kernel in FILTROS.items():
            esperado = kernel(img)  # referencia sequencial
            for n_processos in [1, 2, 3, 4, 8]:
                if n_processos > altura:
                    continue
                obtido = paralelo_simulado(img, kernel, n_processos)
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
