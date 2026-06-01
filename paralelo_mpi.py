"""
paralelo_mpi.py
===============

Versao PARALELA com MPI (mpi4py).

Estrategia de distribuicao (alinhada com os slides do professor - pagina 13):

    1. O processo 0 le a imagem com cv2 (OpenCV).
    2. comm.Bcast envia a IMAGEM INTEIRA para TODOS os processos.
       - Uppercase = operacao de buffer: usa o buffer de memoria do array NumPy
         diretamente, SEM serializar com pickle. Para imagens grandes isso e
         MUITO mais rapido que a versao minuscula (ver "Regra de ouro" - pag.15).
    3. Cada processo calcula QUAIS linhas sao suas (via calcular_particao).
    4. Para filtrar, cada processo extrai do seu proprio exemplar da imagem as
       linhas que precisa, INCLUINDO 1 linha halo acima e abaixo (vizinhos de
       borda). Isso resolve a "limitacao nas bordas" mencionada na pagina 13:
       como todos ja tem a imagem completa, buscar as linhas vizinhas e trivial,
       sem precisar de Sendrecv.
    5. Gatherv reune os pedacos filtrados no processo 0.

Metodologia de medicao (pagina 16 dos slides):
    - warm-up : repeticoes iniciais descartadas antes de ligar o cronometro.
    - sincronizacao: Barrier() antes e depois de cada iteracao.
    - isolamento de ruido: as 30 repeticoes ficam DENTRO do script Python.
      O mpiexec e executado UMA SO VEZ por numero de processos, evitando o
      overhead de "inicio frio" do ambiente MPI em cada amostra.
    - outliers: removidos pelo metodo IQR (bench_utils.py).

Uso:
    mpiexec -n 4 py paralelo_mpi.py --filtro media   --imagem imagem.png --saida par_media_4.json
    mpiexec -n 4 py paralelo_mpi.py --filtro mediana --imagem imagem.png --saida par_mediana_4.json
"""

import argparse
import numpy as np
import cv2
from mpi4py import MPI

from filtros import FILTROS
from bench_utils import resumir_tempos, salvar_json
from particao import calcular_particao


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()   # identidade deste processo (0 = "chefe")
    size = comm.Get_size()   # total de processos (o P do mpiexec -n P)

    p = argparse.ArgumentParser(description="Versao paralela (MPI) dos filtros.")
    p.add_argument("--filtro", choices=list(FILTROS.keys()), required=True)
    p.add_argument("--reps", type=int, default=30,
                   help="repeticoes medidas dentro do loop (pagina 16 dos slides)")
    p.add_argument("--aquecimento", type=int, default=2,
                   help="repeticoes de warm-up descartadas (pagina 16 dos slides)")
    p.add_argument("--imagem", type=str, default="imagem.png")
    p.add_argument("--saida", type=str, default=None)
    args = p.parse_args()

    kernel = FILTROS[args.filtro]

    # ------------------------------------------------------------------
    # 1) Apenas o processo 0 le a imagem do disco com OpenCV (pagina 9).
    # ------------------------------------------------------------------
    if rank == 0:
        img = cv2.imread(args.imagem, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Imagem '{args.imagem}' nao encontrada ou formato nao suportado.")
        img = np.ascontiguousarray(img, dtype=np.uint8)
        altura, largura = img.shape
    else:
        img = None
        altura = largura = None

    # 2) Broadcast das DIMENSOES - apenas 2 inteiros, nao sao dados de imagem.
    #    A versao minuscula (bcast) e aceitavel aqui: e rapido e sem pickle
    #    relevante para apenas 2 numeros.
    altura = comm.bcast(altura, root=0)
    largura = comm.bcast(largura, root=0)

    if size > altura:
        if rank == 0:
            print("ERRO: ha mais processos do que linhas na imagem.")
        return

    # ------------------------------------------------------------------
    # 3) comm.Bcast envia a IMAGEM INTEIRA para todos os processos.
    #    MAIUSCULO = operacao de buffer NumPy (sem pickle) -> muito mais rapido.
    #    Cada processo vai trabalhar com a sua copia local da imagem.
    #    (Resposta da pergunta de reflexao 3: Bcast vs bcast)
    # ------------------------------------------------------------------
    if rank != 0:
        img = np.empty((altura, largura), dtype=np.uint8)  # buffer de recepcao

    comm.Bcast(img, root=0)  # <<< MAIUSCULO: buffer numpy, sem pickle

    # ------------------------------------------------------------------
    # 4) Calcula a particao: quais linhas sao responsabilidade deste processo.
    # ------------------------------------------------------------------
    linhas_por_processo, counts, deslocamentos = calcular_particao(altura, size, largura)
    start_row = sum(linhas_por_processo[:rank])           # primeira linha minha
    end_row   = start_row + linhas_por_processo[rank]     # limite superior (excl.)

    # ------------------------------------------------------------------
    # 5) Laco de medicao.
    #    Cada iteracao = extrair fatia com halo + aplicar filtro.
    #    Nao ha Sendrecv: cada processo ja tem a imagem inteira via Bcast e
    #    busca as linhas vizinhas diretamente.
    #    (Trata a "limitacao nas bordas" da pagina 13: como todos tem a imagem
    #    completa, pegar a linha do vizinho e apenas uma fatia de array local.)
    # ------------------------------------------------------------------
    local_saida = None
    tempos = []

    for _ in range(args.aquecimento + args.reps):
        comm.Barrier()   # sincronizacao: todos chegam aqui antes do cronometro
        t0 = MPI.Wtime()

        # Fatia COM halo: inclui 1 linha acima e 1 abaixo (quando existem),
        # para que o kernel calcule as bordas da faixa com vizinhanca correta.
        halo_top = max(0, start_row - 1)
        halo_bot = min(altura, end_row + 1)
        sub = img[halo_top:halo_bot]   # shape: (my_lines + 0/1/2, largura)

        filtered_sub = kernel(sub)

        # Descarta a(s) linha(s) de halo do resultado.
        # offset=0 se start_row==0 (global top), offset=1 caso contrario.
        offset = start_row - halo_top
        local_saida = np.ascontiguousarray(
            filtered_sub[offset : offset + linhas_por_processo[rank]]
        )

        comm.Barrier()   # sincronizacao: todos terminaram
        t1 = MPI.Wtime()

        # Tempo = processo mais LENTO (todos esperam pelo gargalo).
        dt = comm.reduce(t1 - t0, op=MPI.MAX, root=0)
        if rank == 0:
            tempos.append(dt)

    # ------------------------------------------------------------------
    # 6) Gatherv reune os pedacos filtrados no processo 0.
    #    Uppercase + MPI.UNSIGNED_CHAR = buffer numpy, sem pickle.
    # ------------------------------------------------------------------
    if rank == 0:
        imagem_filtrada = np.empty(altura * largura, dtype=np.uint8)
    else:
        imagem_filtrada = None

    destino = [imagem_filtrada, counts, deslocamentos, MPI.UNSIGNED_CHAR]
    comm.Gatherv(local_saida.ravel(), destino, root=0)

    # ------------------------------------------------------------------
    # 7) Apenas o processo 0 calcula estatisticas e grava resultados.
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

        # Salva a imagem filtrada como PNG (OpenCV).
        saida_png = f"saida_{args.filtro}_{size}p.png"
        cv2.imwrite(saida_png, imagem_filtrada)

        print(f"[PAR {args.filtro} n={size}] media = {estat['media']:.4f}s  "
              f"variancia = {estat['variancia']:.2e}  "
              f"(amostras={estat['n_amostras']})  -> {caminho}")


if __name__ == "__main__":
    main()
