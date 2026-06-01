"""
gerar_imagem.py
===============

Gera a imagem de teste em escala de cinza e a salva como PNG.

Usa OpenCV (cv2) para salvar -- a mesma biblioteca que o professor ensinou
(pagina 9 dos slides: "pip install opencv-python"). Como fallback, tenta PIL.

Por que uma imagem grande e com ruido?

    - GRANDE: o enunciado pede uma imagem que "realmente demande paralelizacao".
      Uma imagem pequena terminaria tao rapido que o custo de comunicacao do MPI
      dominaria e nao haveria ganho. Por padrao geramos 3000x3000 = 9 megapixels.
    - COM RUIDO "SAL E PIMENTA": pixels aleatorios viram preto (0) ou branco (255).
      Esse e exatamente o tipo de ruido que o filtro de MEDIANA remove muito bem
      e o filtro de MEDIA apenas borra, tornando a comparacao entre os dois
      filtros mais interessante (slides paginas 4-8).

Uso:
    py gerar_imagem.py                 # 3000x3000, 5% de ruido, semente fixa
    py gerar_imagem.py --altura 6000 --largura 6000 --ruido 0.1
"""

import argparse
import numpy as np


def gerar(altura, largura, fracao_ruido, semente):
    """Cria uma imagem (altura x largura) uint8 com gradiente + circulos + ruido."""
    rng = np.random.default_rng(semente)

    # 1) Fundo: gradiente suave (de escuro para claro) para haver estrutura.
    linha = np.linspace(0, 255, largura, dtype=np.float64)
    img = np.tile(linha, (altura, 1))

    # 2) Circulos claros e escuros para ter bordas e objetos definidos.
    yy, xx = np.mgrid[0:altura, 0:largura]
    for _ in range(8):
        cy = rng.integers(0, altura)
        cx = rng.integers(0, largura)
        raio = rng.integers(min(altura, largura) // 20, min(altura, largura) // 6)
        cor = rng.integers(0, 256)
        dentro = (yy - cy) ** 2 + (xx - cx) ** 2 <= raio ** 2
        img[dentro] = cor

    img = img.astype(np.uint8)

    # 3) Ruido "sal e pimenta": fracao_ruido dos pixels vira 0 ou 255.
    mascara = rng.random((altura, largura)) < fracao_ruido
    sal = rng.random((altura, largura)) < 0.5
    img[mascara & sal] = 255
    img[mascara & ~sal] = 0

    return img


def salvar_imagem(img, caminho):
    """Salva imagem uint8 2D como PNG. Tenta cv2; fallback PIL; fallback npy."""
    try:
        import cv2
        cv2.imwrite(caminho, img)
        print(f"Imagem salva em {caminho} ({img.shape[0]}x{img.shape[1]}, via OpenCV)")
        return
    except ImportError:
        pass
    try:
        from PIL import Image
        Image.fromarray(img).save(caminho)
        print(f"Imagem salva em {caminho} ({img.shape[0]}x{img.shape[1]}, via PIL)")
        return
    except ImportError:
        pass
    # Ultimo recurso: salva como .npy (sequencial e paralelo aceitam qualquer formato)
    caminho_npy = caminho.rsplit(".", 1)[0] + ".npy"
    import numpy as np
    np.save(caminho_npy, img)
    print(f"cv2 e PIL nao instalados; salvo como {caminho_npy} (instale opencv-python)")


def main():
    p = argparse.ArgumentParser(description="Gera a imagem de teste em escala de cinza.")
    p.add_argument("--altura", type=int, default=3000)
    p.add_argument("--largura", type=int, default=3000)
    p.add_argument("--ruido", type=float, default=0.05)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--saida", type=str, default="imagem.png")
    args = p.parse_args()

    img = gerar(args.altura, args.largura, args.ruido, args.semente)
    salvar_imagem(img, args.saida)


if __name__ == "__main__":
    main()
