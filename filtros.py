"""
filtros.py
==========

Kernels dos dois filtros de suavizacao usados na atividade:

    - filtro_media_3x3   : filtro de media 3x3   (visto em aula)
    - filtro_mediana_3x3 : filtro de mediana 3x3 (2o filtro escolhido pelo grupo)

Os dois kernels recebem uma imagem 2D (escala de cinza) e devolvem uma imagem
do MESMO tamanho, tratando as bordas por REPLICACAO de borda ("edge"/clamp):
o pixel que ficaria "fora" da imagem e substituido pela borda mais proxima.

Esses mesmos kernels sao usados tanto pela versao sequencial quanto pela versao
paralela. Na versao paralela, cada processo passa ao kernel o seu bloco de linhas
JA acrescido das duas linhas-fantasma (halo) recebidas dos vizinhos; assim o
resultado fica IDENTICO ao da versao sequencial (ver paralelo_mpi.py).
"""

import numpy as np


def _replica_borda(img):
    """Acrescenta 1 pixel de borda em todos os lados, replicando a borda.

    Transforma uma imagem (H, W) em (H+2, W+2). Isso permite calcular a
    vizinhanca 3x3 de QUALQUER pixel (inclusive os das bordas) usando apenas
    fatiamento de arrays, sem ``if`` para tratar borda pixel a pixel.
    """
    return np.pad(img, pad_width=1, mode="edge")


def filtro_media_3x3(img):
    """Filtro de media 3x3.

    Cada pixel de saida e a media aritmetica dos 9 pixels da sua vizinhanca 3x3.
    Implementacao vetorizada: em vez de dois lacos ``for`` percorrendo a imagem,
    somamos 9 versoes deslocadas da imagem com padding e dividimos por 9.
    """
    a = _replica_borda(img).astype(np.float64)

    soma = (a[0:-2, 0:-2] + a[0:-2, 1:-1] + a[0:-2, 2:] +
            a[1:-1, 0:-2] + a[1:-1, 1:-1] + a[1:-1, 2:] +
            a[2:,   0:-2] + a[2:,   1:-1] + a[2:,   2:])

    return np.rint(soma / 9.0).astype(img.dtype)


def filtro_mediana_3x3(img):
    """Filtro de mediana 3x3.

    Cada pixel de saida e a MEDIANA dos 9 pixels da sua vizinhanca 3x3 (o valor
    do meio quando os 9 vizinhos sao ordenados). Diferente da media, a mediana
    nao mistura valores: ela escolhe um dos 9 valores existentes, o que remove
    ruido "sal e pimenta" sem borrar tanto as bordas dos objetos.

    Implementacao vetorizada: empilhamos as 9 vizinhancas deslocadas em um array
    (9, H, W) e tiramos a mediana ao longo do eixo 0 (os 9 vizinhos de cada pixel).
    """
    a = _replica_borda(img)

    vizinhos = np.stack([
        a[0:-2, 0:-2], a[0:-2, 1:-1], a[0:-2, 2:],
        a[1:-1, 0:-2], a[1:-1, 1:-1], a[1:-1, 2:],
        a[2:,   0:-2], a[2:,   1:-1], a[2:,   2:],
    ], axis=0)

    mediana = np.median(vizinhos, axis=0)

    return mediana.astype(img.dtype)


# Mapa de nome -> funcao, usado pelos scripts sequencial.py e paralelo_mpi.py
# para escolher o filtro a partir de um argumento de linha de comando.
FILTROS = {
    "media": filtro_media_3x3,
    "mediana": filtro_mediana_3x3,
}
