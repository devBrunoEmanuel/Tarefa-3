"""
sequencial.py
=============

Versao SEQUENCIAL (1 processo, sem MPI). Aplica o filtro escolhido a imagem
inteira varias vezes, mede o tempo de cada repeticao, descarta o aquecimento
(warm-up), remove outliers e grava as estatisticas em JSON.

Este e o tempo de referencia (T_seq) usado para calcular o speedup.

Usa OpenCV (cv2) para carregar e salvar a imagem, alinhado com os slides do
professor (pagina 9: "pip install opencv-python").

Metodologia (pagina 16 dos slides):
    - warm-up : 2 primeiras repeticoes descartadas
    - isolamento de ruido: 30 repeticoes dentro do script (sem reiniciar processo)
    - remocao de outliers: metodo IQR

Uso:
    py sequencial.py --filtro media   --reps 30 --imagem imagem.png
    py sequencial.py --filtro mediana --reps 30 --imagem imagem.png
"""

import argparse
import time
import numpy as np
import cv2

from filtros import FILTROS
from bench_utils import resumir_tempos, salvar_json


def main():
    p = argparse.ArgumentParser(description="Versao sequencial dos filtros.")
    p.add_argument("--filtro", choices=list(FILTROS.keys()), required=True)
    p.add_argument("--reps", type=int, default=30,
                   help="repeticoes medidas alem do aquecimento (pagina 16 dos slides)")
    p.add_argument("--aquecimento", type=int, default=2,
                   help="warm-up: repeticoes descartadas no inicio")
    p.add_argument("--imagem", type=str, default="imagem.png")
    p.add_argument("--saida", type=str, default=None)
    args = p.parse_args()

    kernel = FILTROS[args.filtro]

    # Carrega a imagem com OpenCV em escala de cinza (como nos slides - pagina 9).
    img = cv2.imread(args.imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Imagem '{args.imagem}' nao encontrada ou formato nao suportado.")
    img = np.ascontiguousarray(img, dtype=np.uint8)

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

    # Salva a imagem filtrada como PNG.
    saida_png = f"saida_{args.filtro}_seq.png"
    cv2.imwrite(saida_png, saida)

    print(f"[SEQ {args.filtro}] media = {estat['media']:.4f}s  "
          f"variancia = {estat['variancia']:.2e}  "
          f"(n={estat['n_amostras']})  -> {caminho}")


if __name__ == "__main__":
    main()
