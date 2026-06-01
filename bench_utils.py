"""
bench_utils.py
==============

Funcoes de apoio ao benchmark, compartilhadas pela versao sequencial e pela
versao paralela:

    - remover_outliers : descarta tempos anomalos (metodo IQR / cercas de Tukey)
    - estatisticas     : media, variancia e desvio padrao das amostras
    - salvar_json      : grava o resultado de uma medicao em arquivo JSON
    - carregar_json    : le um arquivo JSON de resultado

Metodologia de medicao (alinhada com o exercicio de benchmark da aula):

    1. As repeticoes sao feitas DENTRO de um unico processo ja iniciado, em um
       laco. Nao reiniciamos o ambiente (no caso paralelo, nao chamamos
       ``mpiexec`` uma vez por repeticao). Assim o custo fixo de inicializacao
       e pago UMA vez e nao polui cada amostra.
    2. As primeiras repeticoes sao descartadas (aquecimento / "warm-up"), pois
       sofrem com cache frio e alocacao preguicosa de memoria.
    3. Depois removemos outliers e so entao calculamos media, variancia e speedup.
"""

import json
import numpy as np


def remover_outliers(amostras):
    """Remove outliers pelo metodo IQR (cercas de Tukey).

    Calcula o 1o quartil (Q1) e o 3o quartil (Q3). A faixa interquartil e
    IQR = Q3 - Q1. Consideramos outlier qualquer amostra fora do intervalo
    [Q1 - 1.5*IQR , Q3 + 1.5*IQR]. E um criterio robusto: nao depende da media
    (que e justamente o que os outliers distorcem).
    """
    a = np.asarray(amostras, dtype=float)
    if a.size < 4:
        # poucas amostras: nao da para estimar quartis de forma confiavel
        return a
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    mascara = (a >= limite_inf) & (a <= limite_sup)
    return a[mascara]


def estatisticas(amostras):
    """Devolve um dicionario com media, variancia, desvio padrao, min e max.

    A variancia e o desvio usam ddof=1 (variancia amostral), apropriada quando
    estimamos a variabilidade a partir de uma amostra e nao da populacao inteira.
    """
    a = np.asarray(amostras, dtype=float)
    n = int(a.size)
    return {
        "n_amostras": n,
        "media": float(a.mean()) if n > 0 else 0.0,
        "variancia": float(a.var(ddof=1)) if n > 1 else 0.0,
        "desvio_padrao": float(a.std(ddof=1)) if n > 1 else 0.0,
        "min": float(a.min()) if n > 0 else 0.0,
        "max": float(a.max()) if n > 0 else 0.0,
    }


def resumir_tempos(tempos_brutos, n_aquecimento):
    """Pipeline completo: descarta aquecimento -> remove outliers -> estatisticas.

    Retorna (estat, tempos_validos) onde ``estat`` e o dicionario de estatisticas
    e ``tempos_validos`` sao os tempos que sobraram apos descartar aquecimento e
    outliers (uteis para registrar no relatorio).
    """
    apos_aquecimento = np.asarray(tempos_brutos, dtype=float)[n_aquecimento:]
    validos = remover_outliers(apos_aquecimento)
    return estatisticas(validos), validos


def salvar_json(caminho, dados):
    """Grava ``dados`` (dicionario) em ``caminho`` no formato JSON."""
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def carregar_json(caminho):
    """Le e devolve o dicionario gravado em ``caminho`` (JSON)."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)
