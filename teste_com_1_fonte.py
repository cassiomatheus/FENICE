import sys
import pandas as pd
import ast

# Ajuste o caminho conforme o seu ambiente
sys.path.append("/content/FENICE")
from metric.FENICE import FENICE

# =====================================================================
# 1. INICIALIZAÇÃO DO FENICE (Seu código original)
# =====================================================================
print("Inicializando FENICE...")
avaliador = FENICE()
""" use_coref=False,
    num_sent_per_paragraph=5,
    sliding_paragraphs=True,
    doc_level_nli=True,
    paragraph_level_nli=True,
    claim_extractor_batch_size=16,
    nli_batch_size=16,
    nli_max_length=512
)"""
avaliador.alignments_cache = {}


# =====================================================================
# TOY DATASET: PROJETADO PARA FALHA DE SÍNTESE MULTI-PÁGINA (MULTI-HOP)
# =====================================================================

fonte_teste_confusao = """
In 2015, the National Aerospace Agency initiated the development of the Horizon Satellite, allocating an initial budget of 50 million dollars for its structural engineering phase. The program aimed to monitor atmospheric carbon levels over a ten-year operational lifecycle.

Meanwhile, the Department of Industrial Biotechnology conducted extensive research on advanced yeast fermentation for commercial baking. The implementation of automated thermal bioreactors optimized production efficiency by 15 percent, although raw material costs fluctuated significantly due to global grain export policies and supply chain constraints in agricultural markets.

Following a rigorous technical audit completed last year, the National Aerospace Agency officially canceled the Horizon Satellite program. The final report cited critical and irreparable failures discovered in the primary telemetry and communication subsystems during high-altitude simulations.
"""

resumo_teste_confusao = """
The Horizon Satellite, which began development in 2015 under the National Aerospace Agency, was officially canceled last year due to critical telemetry failures.
"""




#####################

# Inflamos o parágrafo do meio repetindo-o para estourar os 512 tokens e forçar o truncamento
paragrafo_intruso_repetido = """
The Department of Industrial Biotechnology conducted extensive research on advanced yeast fermentation for commercial baking. The implementation of automated thermal bioreactors optimized production efficiency by 15 percent, although raw material costs fluctuated significantly due to global grain export policies and supply chain constraints in agricultural markets.
""" * 4 # Repete 4 vezes para criar um "muro" de texto de mais de 500 tokens

fonte_teste_confusao = f"""
In 2015, the National Aerospace Agency initiated the development of the Horizon Satellite, allocating an initial budget of 50 million dollars for its structural engineering phase. The program aimed to monitor atmospheric carbon levels over a ten-year operational lifecycle.

{paragrafo_intruso_repetido}

Following a rigorous technical audit completed last year, the National Aerospace Agency officially canceled the Horizon Satellite program. The final report cited critical and irreparable failures discovered in the primary telemetry and communication subsystems during high-altitude simulations.
"""





# =====================================================================
# EXECUÇÃO DO TESTE
# =====================================================================
print("Rodando teste de estresse de contexto...")
resultado_toy = avaliador.score_batch([{'document': fonte_teste_confusao, 'summary': resumo_teste_confusao}])[0]

print("\n=== RESULTADO DO TOY DATASET ===")
print(f"Score global: {resultado_toy['score']:.4f}")
print(f"Status de factualidade: {resultado_toy['status_counts']}")

for i, al in enumerate(resultado_toy['alignments']):
    print(f"\nClaim {i}: {al['summary_claim']}")
    print(f"  -> Status atribuído pelo FENICE: {al['factual_status']}")
    print(f"  -> Chunk de Evidência Escolhido: {al['source_passage'][:150]}...")