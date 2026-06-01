"""
benchmark.py
============

Orquestrador do experimento. Ele:

  1. Garante que a imagem de teste existe (gera se preciso).
  2. Roda a versao SEQUENCIAL de cada filtro (tempo de referencia T_seq).
  3. Roda a versao PARALELA de cada filtro com 2, 4 e 8 processos.
  4. Le todos os resultados JSON, monta a TABELA DE SPEEDUP e desenha o
     GRAFICO de tempo por numero de processos.

IMPORTANTE (metodologia da aula): cada combinacao (filtro, n_processos) e
lancada com ``mpiexec`` UMA UNICA vez. As repeticoes acontecem DENTRO do
programa (laco em paralelo_mpi.py). Ou seja, NAO chamamos ``mpiexec`` uma vez
por repeticao -- isso forcaria um "inicio frio" do ambiente MPI a cada amostra,
adicionando tempo extra e simetrico que so atrapalha a medicao.

    speedup(P) = T_seq / T_paralelo(P)

Uso:
    py benchmark.py                          # 2,4,8 processos; filtros media e mediana
    py benchmark.py --processos 2,4 --reps 20
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from bench_utils import carregar_json


def executar(comando):
    """Executa um comando, repassando a saida para o terminal. Aborta se falhar."""
    print(">>", " ".join(comando))
    res = subprocess.run(comando)
    if res.returncode != 0:
        raise SystemExit(f"Comando falhou (codigo {res.returncode}): {' '.join(comando)}")


def main():
    p = argparse.ArgumentParser(description="Orquestra o benchmark sequencial vs MPI.")
    p.add_argument("--imagem", type=str, default="imagem.npy")
    p.add_argument("--filtros", type=str, default="media,mediana")
    p.add_argument("--processos", type=str, default="2,4,8")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--aquecimento", type=int, default=2)
    args = p.parse_args()

    filtros = [f.strip() for f in args.filtros.split(",") if f.strip()]
    lista_processos = [int(x) for x in args.processos.split(",") if x.strip()]

    python = sys.executable  # mesmo interpretador que esta rodando este script

    # 1) Garante a imagem de teste.
    if not os.path.exists(args.imagem):
        print("Imagem nao encontrada; gerando uma nova...")
        executar([python, "gerar_imagem.py", "--saida", args.imagem])

    resultados = {}  # (filtro, n_processos) -> dicionario JSON carregado

    # 2 e 3) Roda sequencial e paralelo para cada filtro.
    for filtro in filtros:
        # --- Sequencial (T_seq) ---
        saida_seq = f"seq_{filtro}.json"
        executar([python, "sequencial.py",
                  "--filtro", filtro, "--reps", str(args.reps),
                  "--aquecimento", str(args.aquecimento),
                  "--imagem", args.imagem, "--saida", saida_seq])
        resultados[(filtro, 1)] = carregar_json(saida_seq)

        # --- Paralelo (2, 4, 8 ...) ---
        for nproc in lista_processos:
            saida_par = f"par_{filtro}_{nproc}.json"
            executar(["mpiexec", "-n", str(nproc), python, "paralelo_mpi.py",
                      "--filtro", filtro, "--reps", str(args.reps),
                      "--aquecimento", str(args.aquecimento),
                      "--imagem", args.imagem, "--saida", saida_par])
            resultados[(filtro, nproc)] = carregar_json(saida_par)

    # 4) Tabela de speedup + grafico.
    montar_tabela(filtros, lista_processos, resultados)
    desenhar_grafico(filtros, lista_processos, resultados)


def montar_tabela(filtros, lista_processos, resultados):
    """Imprime e salva (CSV + Markdown) a tabela de speedup."""
    linhas_csv = ["filtro,n_processos,tempo_medio_s,variancia,speedup,eficiencia"]
    linhas_md = ["| Filtro | Processos | Tempo medio (s) | Variancia | Speedup | Eficiencia |",
                 "|--------|-----------|-----------------|-----------|---------|------------|"]

    print("\n================ TABELA DE SPEEDUP ================")
    for filtro in filtros:
        t_seq = resultados[(filtro, 1)]["estatisticas"]["media"]

        # Linha do sequencial (speedup de referencia = 1).
        var_seq = resultados[(filtro, 1)]["estatisticas"]["variancia"]
        print(f"{filtro:8s} | seq | {t_seq:8.4f}s | speedup 1.00")
        linhas_csv.append(f"{filtro},1,{t_seq:.6f},{var_seq:.3e},1.000,1.000")
        linhas_md.append(f"| {filtro} | 1 (seq) | {t_seq:.4f} | {var_seq:.2e} | 1.00 | 1.00 |")

        for nproc in lista_processos:
            estat = resultados[(filtro, nproc)]["estatisticas"]
            t_par = estat["media"]
            var = estat["variancia"]
            speedup = t_seq / t_par if t_par > 0 else float("nan")
            eficiencia = speedup / nproc
            print(f"{filtro:8s} | {nproc:2d}p | {t_par:8.4f}s | "
                  f"speedup {speedup:4.2f} | eficiencia {eficiencia:4.2f}")
            linhas_csv.append(f"{filtro},{nproc},{t_par:.6f},{var:.3e},{speedup:.3f},{eficiencia:.3f}")
            linhas_md.append(f"| {filtro} | {nproc} | {t_par:.4f} | {var:.2e} | {speedup:.2f} | {eficiencia:.2f} |")

    with open("resultados_speedup.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_csv) + "\n")
    with open("tabela_speedup.md", "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_md) + "\n")

    print("\nTabela salva em resultados_speedup.csv e tabela_speedup.md")


def desenhar_grafico(filtros, lista_processos, resultados):
    """Desenha o grafico de tempo medio por numero de processos (escala log2 em x)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend sem janela (so salva arquivo)
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib nao instalado; grafico nao gerado (a tabela foi salva).")
        return

    plt.figure(figsize=(8, 5))
    for filtro in filtros:
        xs = [1] + lista_processos
        ys = [resultados[(filtro, x)]["estatisticas"]["media"] for x in xs]
        plt.plot(xs, ys, marker="o", label=filtro)

    plt.xscale("log", base=2)
    plt.xticks([1] + lista_processos, ["1 (seq)"] + [str(x) for x in lista_processos])
    plt.xlabel("Numero de processos")
    plt.ylabel("Tempo medio (s)")
    plt.title("Tempo por numero de processos")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tempo_por_processos.png", dpi=130)
    print("Grafico salvo em tempo_por_processos.png")


if __name__ == "__main__":
    main()
