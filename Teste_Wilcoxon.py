import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# carregar dados
bart = pd.read_excel("caminho para o arquivo")
primera = pd.read_excel("caminho para o arquivo")
led = pd.read_excel("caminho para o arquivo")

# ordenar
bart = bart.sort_values("sample_id")
primera = primera.sort_values("sample_id")
led = led.sort_values("sample_id")

def clean_pair(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    return a[mask], b[mask]

def effect_size_wilcoxon(a, b, stat):
    # Calcula a diferença entre os pares
    d = a - b

    # O método 'wilcox' descarta as amostras onde a diferença é ZERO
    # Portanto, nosso "N efetivo" para a estatística Z deve ignorar os zeros
    d_nonzero = d[d != 0]
    n = len(d_nonzero)

    if n == 0:
        return 0.0

    # Fórmulas de aproximação normal para Wilcoxon
    mean_w = n * (n + 1) / 4.0
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)

    # Adicionamos uma pequena correção de continuidade (0.5) para melhorar a precisão
    z = (stat - mean_w + 0.5) / std_w

    # Rosenthal formula: r = |Z| / sqrt(N_total)
    # N_total é o número de observações NÃO NULAS (2 vezes o número de pares válidos)
    r = abs(z) / np.sqrt(2 * n)

    return r

def test(a, b, name):
    a, b = clean_pair(a, b)

    if len(a) < 10:
        print(f"{name}: poucos dados (N={len(a)})")
        return

    # Executa o teste
    stat, p = wilcoxon(a, b, zero_method='wilcox')

    # Diferença média bruta (apenas para referência)
    diff_mean = np.mean(a - b)

    # Effect size corrigido
    effect = effect_size_wilcoxon(a, b, stat)

    # Correção de Bonferroni (multiplica por 3 devido às 3 comparações)
    p_adjusted = min(p * 3, 1.0)

    # Interpretação do Effect Size (Cohen)
    if effect < 0.3:
        effect_label = "Pequeno"
    elif effect < 0.5:
        effect_label = "Médio"
    else:
        effect_label = "Grande"

    print(f"\n[{name}]")
    print(f"  W = {stat:.4f} | N_total_pares = {len(a)}")
    print(f"  p-valor Original = {p:.6f}")
    print(f"  p-valor Ajustado = {p_adjusted:.6f} ", end="")
    print("(Significativo!)" if p_adjusted < 0.05 else "(NÃO Significativo)")
    print(f"  Diferença média  = {diff_mean:.4f}")
    print(f"  Effect size (r)  = {effect:.4f} ({effect_label})")

# rodar testes
bart_score = bart["score"].values
primera_score = primera["score"].values
led_score = led["score"].values

test(bart_score, primera_score, "BART vs PRIMERA")
test(bart_score, led_score, "BART vs LED")
test(primera_score, led_score, "PRIMERA vs LED")