import sys
import pandas as pd
import ast
import re

# Ajuste o caminho conforme o seu ambiente
sys.path.append("/content/FENICE")
from metric.FENICE import FENICE

ARQUIVO = "govreport_erros_pred_corref_entity_realistas_v3.csv"

# =====================================================================
# 1. INICIALIZAÇÃO DO FENICE
# =====================================================================
print("Inicializando FENICE...")
fenice = FENICE()
fenice.alignments_cache = {}

# =====================================================================
# 2. CARREGAMENTO DO DADO (REGISTRO INDIVIDUAL)
# =====================================================================
print(f"\nCarregando dados do CSV: {ARQUIVO}")
df = pd.read_csv(ARQUIVO)


# Escolhendo a linha 0 para o teste individual
LINHA_TESTE = 0
linha = df.iloc[LINHA_TESTE]
fonte = linha['fonte']
resumo = linha['resumo_original'] 

document = fonte

summary = resumo

batch = [
    {"document": document, "summary": summary}
]

results = fenice.score_batch(batch)
print(results)
