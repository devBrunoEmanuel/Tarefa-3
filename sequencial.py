"""
sequencial.py
=============

Versao SEQUENCIAL (1 unico processo, sem MPI). Aplica o filtro escolhido a
imagem inteira varias vezes, mede o tempo de cada repeticao, descarta o
aquecimento, remove outliers e grava as estatisticas em um arquivo JSON.

Este e o tempo de referencia (T_seq) usado para calcular o speedup da versao
paralela.

Uso:
    py sequencial.py --filtro media   --reps 15 --imagem imagem.npy --saida seq_media.json
    py sequencial.py --filtro mediana --reps 15 --imagem imagem.npy --saida seq_mediana.json
"""

import argparse
import time
import numpy as np

from filtros import FILTROS
from bench_utils import resumir_tempos, salvar_json


def main():
    p = argparse.ArgumentParser(description="Versao sequencial dos filtros.")
    p.add_argument("--filtro", choices=list(FILTROS.keys()), required=True)
    p.add_argument("--reps", type=int, default=15, help="repeticoes medidas (alem do aquecimento)")
    p.add_argument("--aquecimento", type=int, default=2, help="repeticoes de aquecimento descartadas")
    p.add_argument("--imagem", type=str, default="imagem.npy")
    p.add_argument("--saida", type=str, default=None, help="arquivo JSON de saida")
    args = p.parse_args()

    kernel = FILTROS[args.filtro]
    img = np.load(args.imagem)

    tempos = []
    saida = None
    for _ in range(args.aquecimento + args.reps):
        t0 = time.perf_counter()
        saida = kernel(img)
        t1 = time.perf_counter()
        tempos.append(t1 - t0)

    estat, validos = resumir_tempos(tempos, args.aquecimento)

    resultado = {
        "modo": "sequencial",
        "filtro": args.filtro,
        "n_processos": 1,
        "imagem": {"altura": int(img.shape[0]), "largura": int(img.shape[1])},
        "tempos_brutos": tempos,
        "tempos_validos": validos.tolist(),
        "estatisticas": estat,
    }

    caminho = args.saida or f"seq_{args.filtro}.json"
    salvar_json(caminho, resultado)

    print(f"[SEQ {args.filtro}] media = {estat['media']:.4f}s  "
          f"variancia = {estat['variancia']:.2e}  "
          f"(n={estat['n_amostras']})  -> {caminho}")


if __name__ == "__main__":
    main()
