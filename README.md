# Comparação de filtros de imagem com MPI

Atividade de Sistemas/Programação Distribuída: comparar dois filtros de
suavização (**média 3×3** e **mediana 3×3**) em versão **sequencial** e
**paralela com MPI**, medindo **speedup** e **variância**.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `filtros.py` | Kernels dos dois filtros (média e mediana 3×3). |
| `gerar_imagem.py` | Gera a imagem de teste (grande, com ruído sal e pimenta). |
| `sequencial.py` | Versão sequencial + medição de tempo. |
| `paralelo_mpi.py` | Versão paralela MPI (Scatterv/Gatherv + halo) + medição. |
| `bench_utils.py` | Remoção de outliers, estatísticas e IO de JSON. |
| `benchmark.py` | Orquestra tudo, monta a tabela de speedup e o gráfico. |
| `EXPLICACAO_CODIGO.md` | **Explicação do código linha a linha.** |
| `RELATORIO.md` | Relatório (tabela, gráfico e respostas das reflexões). |

## Pré-requisitos

1. **MS-MPI** (runtime + SDK) instalado no Windows — fornece o `mpiexec`.
   Baixe "Microsoft MPI" no site da Microsoft (instale `msmpisetup.exe` e
   `msmpisdk.msi`). Confirme com:

   ```powershell
   mpiexec
   ```

2. Pacotes Python:

   ```powershell
   py -m pip install -r requirements.txt
   ```

   > Observação: `mpi4py` no Windows é compilado contra o MS-MPI; por isso o
   > MS-MPI precisa estar instalado **antes** do `pip install`.

## Como rodar

```powershell
# 1) Gera a imagem de teste (3000x3000 por padrão)
py gerar_imagem.py

# 2) Roda o experimento completo: sequencial + 2, 4 e 8 processos,
#    para os filtros de média e mediana, e gera tabela + gráfico
py benchmark.py
```

Saídas geradas:

- `resultados_speedup.csv` e `tabela_speedup.md` — tabela de speedup;
- `tempo_por_processos.png` — gráfico de tempo por número de processos;
- `seq_*.json`, `par_*.json` — medições brutas e estatísticas de cada execução.

### Rodar uma medição isolada (manual)

```powershell
# Sequencial
py sequencial.py --filtro mediana --reps 15

# Paralelo com 4 processos (lançado UMA vez; as repetições são internas)
mpiexec -n 4 py paralelo_mpi.py --filtro mediana --reps 15
```

## Metodologia de medição

- As repetições acontecem **dentro** de um único processo já iniciado. No caso
  paralelo, `mpiexec` é chamado **uma vez por número de processos**, não uma vez
  por repetição — assim o custo de inicialização do ambiente MPI (simétrico) é
  pago uma vez só e não polui cada amostra.
- Descartamos as primeiras repetições (**aquecimento**), removemos **outliers**
  (método IQR / cercas de Tukey) e só então calculamos média, variância e
  speedup.
