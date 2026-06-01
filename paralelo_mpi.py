"""
paralelo_mpi.py
===============

Versao PARALELA com MPI (mpi4py). A imagem e dividida em FAIXAS DE LINHAS, uma
para cada processo. Cada processo filtra apenas a sua faixa e, ao final, os
pedacos sao reunidos no processo 0.

Pontos exigidos pelo enunciado e como sao atendidos aqui:

  * "uso de operacoes com maiuscula (buffer numpy) para toda comunicacao de
    dados de imagem": usamos Scatterv / Gatherv / Sendrecv (TODAS com inicial
    maiuscula). Essas versoes trabalham diretamente sobre o buffer de memoria do
    array NumPy, sem serializar com pickle (que e o que as versoes minusculas,
    como ``scatter``/``gather``/``bcast``, fariam). So usamos a versao minuscula
    (``bcast``) para 2 numeros inteiros (altura e largura), que nao sao "dados de
    imagem".

  * Fronteiras entre processos (linhas-fantasma / halo): para filtrar a primeira
    e a ultima linha da sua faixa, cada processo precisa de 1 linha do vizinho de
    cima e 1 do vizinho de baixo. Essas linhas sao trocadas com Sendrecv.

  * Numero de processos que nao divide a altura: distribuimos o resto entre os
    primeiros processos (alguns recebem 1 linha a mais). Por isso usamos as
    versoes "v" (Scatterv/Gatherv), que aceitam blocos de tamanhos diferentes.

  * Metodologia de medicao: a inicializacao do MPI e a distribuicao da imagem
    (Scatterv) sao feitas UMA vez, ANTES do laco de medicao. Dentro do laco
    medimos apenas o custo que se repete a cada aplicacao do filtro: troca de
    halo + calculo. Assim nao pagamos o custo fixo de inicializacao em cada
    amostra (ele e simetrico e seria so ruido).

Uso (lancado UMA vez por contagem de processos):
    mpiexec -n 4 py paralelo_mpi.py --filtro media --reps 15 --imagem imagem.npy --saida par_media_4.json
"""

import argparse
import numpy as np
from mpi4py import MPI

from filtros import FILTROS
from bench_utils import resumir_tempos, salvar_json
from particao import calcular_particao


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    p = argparse.ArgumentParser(description="Versao paralela (MPI) dos filtros.")
    p.add_argument("--filtro", choices=list(FILTROS.keys()), required=True)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--aquecimento", type=int, default=2)
    p.add_argument("--imagem", type=str, default="imagem.npy")
    p.add_argument("--saida", type=str, default=None)
    args = p.parse_args()

    kernel = FILTROS[args.filtro]

    # ------------------------------------------------------------------
    # 1) Apenas o processo 0 le a imagem do disco. Os demais ainda nao a tem.
    # ------------------------------------------------------------------
    if rank == 0:
        img = np.load(args.imagem)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        altura, largura = img.shape
    else:
        img = None
        altura = largura = None

    # Apenas 2 inteiros (NAO sao dados de imagem) -> pode usar bcast minusculo.
    altura = comm.bcast(altura, root=0)
    largura = comm.bcast(largura, root=0)

    if size > altura:
        if rank == 0:
            print("ERRO: ha mais processos do que linhas na imagem.")
        return

    # ------------------------------------------------------------------
    # 2) Particao por linhas e distribuicao da imagem com Scatterv (UMA vez).
    # ------------------------------------------------------------------
    linhas_por_processo, counts, deslocamentos = calcular_particao(altura, size, largura)
    minhas_linhas = linhas_por_processo[rank]

    # Buffer de recepcao do meu bloco (achatado em 1D).
    recv = np.empty(minhas_linhas * largura, dtype=np.uint8)

    # Scatterv: envia para cada processo o seu bloco de linhas. As versoes "v"
    # permitem blocos de tamanhos diferentes (counts/deslocamentos).
    origem = [img.ravel() if rank == 0 else None, counts, deslocamentos, MPI.UNSIGNED_CHAR]
    comm.Scatterv(origem, recv, root=0)

    bloco_local = recv.reshape(minhas_linhas, largura)

    # ------------------------------------------------------------------
    # 3) Buffer com halo: 2 linhas a mais (uma em cima, uma embaixo) para
    #    guardar as linhas-fantasma vindas dos vizinhos.
    # ------------------------------------------------------------------
    buf = np.empty((minhas_linhas + 2, largura), dtype=np.uint8)
    buf[1:-1] = bloco_local

    # Quem sao meus vizinhos. PROC_NULL = "nao ha vizinho" (bordas globais da
    # imagem). Sendrecv com PROC_NULL nao faz nada (vira no-op).
    vizinho_cima = rank - 1 if rank > 0 else MPI.PROC_NULL
    vizinho_baixo = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    local_saida = None

    # ------------------------------------------------------------------
    # 4) Laco de medicao. Cada iteracao = troca de halo + aplicacao do filtro.
    # ------------------------------------------------------------------
    tempos = []
    for _ in range(args.aquecimento + args.reps):
        comm.Barrier()                 # alinha todos os processos antes de medir
        t0 = MPI.Wtime()

        # Por padrao, replica a propria borda (vale para quem nao tem vizinho).
        buf[0] = buf[1]
        buf[-1] = buf[-2]

        # Troca de halo com Sendrecv (operacao MAIUSCULA, buffer numpy):
        #  - envio minha 1a linha para cima e recebo no halo de cima a ultima
        #    linha do vizinho de cima;
        #  - envio minha ultima linha para baixo e recebo no halo de baixo a 1a
        #    linha do vizinho de baixo.
        comm.Sendrecv(buf[1].copy(), dest=vizinho_cima, recvbuf=buf[0], source=vizinho_cima)
        comm.Sendrecv(buf[-2].copy(), dest=vizinho_baixo, recvbuf=buf[-1], source=vizinho_baixo)

        # Aplica o filtro ao bloco JA com halo e descarta as 2 linhas de halo.
        # As linhas internas que sobram foram calculadas com vizinhanca correta,
        # ficando IDENTICAS ao resultado sequencial.
        saida_com_halo = kernel(buf)
        local_saida = np.ascontiguousarray(saida_com_halo[1:-1])

        comm.Barrier()                 # garante que todos terminaram
        t1 = MPI.Wtime()

        # Tempo da iteracao = tempo do processo MAIS LENTO (o que importa em paralelo).
        dt = comm.reduce(t1 - t0, op=MPI.MAX, root=0)
        if rank == 0:
            tempos.append(dt)

    # ------------------------------------------------------------------
    # 5) Reune os blocos no processo 0 com Gatherv (UMA vez, fora da medicao)
    #    para permitir conferir a corretude do resultado.
    # ------------------------------------------------------------------
    if rank == 0:
        imagem_filtrada = np.empty(altura * largura, dtype=np.uint8)
    else:
        imagem_filtrada = None

    destino = [imagem_filtrada, counts, deslocamentos, MPI.UNSIGNED_CHAR]
    comm.Gatherv(local_saida.ravel(), destino, root=0)

    # ------------------------------------------------------------------
    # 6) Apenas o processo 0 calcula estatisticas e grava o resultado.
    # ------------------------------------------------------------------
    if rank == 0:
        imagem_filtrada = imagem_filtrada.reshape(altura, largura)
        estat, validos = resumir_tempos(tempos, args.aquecimento)

        resultado = {
            "modo": "paralelo",
            "filtro": args.filtro,
            "n_processos": size,
            "imagem": {"altura": int(altura), "largura": int(largura)},
            "linhas_por_processo": linhas_por_processo,
            "tempos_brutos": tempos,
            "tempos_validos": validos.tolist(),
            "estatisticas": estat,
        }

        caminho = args.saida or f"par_{args.filtro}_{size}.json"
        salvar_json(caminho, resultado)

        # Salva tambem a imagem filtrada (para conferencia/visualizacao).
        np.save(f"saida_{args.filtro}_{size}p.npy", imagem_filtrada)

        print(f"[PAR {args.filtro} n={size}] media = {estat['media']:.4f}s  "
              f"variancia = {estat['variancia']:.2e}  "
              f"(amostras={estat['n_amostras']})  -> {caminho}")


if __name__ == "__main__":
    main()
