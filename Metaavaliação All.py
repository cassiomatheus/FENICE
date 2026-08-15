import os
# Impede o crash de fragmentação de memória na RTX 3000
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "6"          
os.environ["MKL_NUM_THREADS"] = "6"          
os.environ["NUMEXPR_NUM_THREADS"] = "6"      

import sys
import pandas as pd
import ast
import re
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gc
import torch
import random
from tqdm import tqdm

# =====================================================================
# TRAVA DE REPRODUTIBILIDADE (SEED)
# =====================================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.set_num_threads(2) # Evita saltos de 100% de consumo na CPU
torch.set_grad_enabled(False) # Economiza VRAM

sys.path.append("/content/FENICE")
from metric.FENICE import FENICE

# =====================================================================
# CONFIGURAÇÕES E CARREGAMENTO GERAL
# =====================================================================
ARQUIVO = "Gerador_erros/govreport_erros_pred_corref_entity_realistas_v1.csv"
TIPOS_AVALIACAO = ["original", "entity", "predicate", "coreference"]

colunas_map = {
    "entity": ('resumo_entity', 'tokens_entity'),
    "predicate": ('resumo_predicado', 'tokens_predicado'),
    "coreference": ('resumo_corref', 'tokens_corref'),
    "original": ('resumo_original', None)
}

print("Carregando modelos base (spaCy, SBERT, FENICE)...")
nlp = spacy.load("en_core_web_md")
encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

fenice = FENICE(
    use_coref=False,
    doc_chunk_overlap=2,
    claim_extractor_batch_size=64,  
    nli_batch_size=64
)

def normalizar(txt):
    return re.sub(r'[^\w\s]', '', str(txt).lower().strip())

def get_injected_span(erro):
    return erro.get('token') or erro.get('injetado') or erro.get('novo_sujeito') or ""

def mapear_claims_para_sentencas_originais(summary_claims, sentencas_originais):
    if not summary_claims or not sentencas_originais:
        return {}
    emb_claims = encoder.encode(summary_claims)
    emb_sents = encoder.encode(sentencas_originais)
    matriz_similarity = cosine_similarity(emb_claims, emb_sents)
    
    mapeamento = {}
    for i, _ in enumerate(summary_claims):
        melhor_sent_idx = np.argmax(matriz_similarity[i])
        mapeamento[i] = melhor_sent_idx
    return mapeamento

# =====================================================================
# INICIALIZAÇÃO DOS ACUMULADORES GLOBAIS
# =====================================================================
global_stats = {
    tipo: {"TP": 0, "FN": 0, "FP": 0, "FP_CONTRADICAO": 0, "FP_ABSTRACAO": 0, "DIAG_CORRETO": 0}
    for tipo in TIPOS_AVALIACAO
}

linhas_real = ["predicate", "entity", "coreference", "other", "nenhum_fp"]
colunas_detectado = ["predicate", "entity", "coreference", "other"]

global_matrices = {
    tipo: pd.DataFrame(0, index=linhas_real, columns=colunas_detectado)
    for tipo in TIPOS_AVALIACAO
}

global_scores = {tipo: [] for tipo in TIPOS_AVALIACAO}

# =====================================================================
# PIPELINE DE PROCESSAMENTO (10 REGISTROS)
# =====================================================================
df = pd.read_csv(ARQUIVO)
registros_para_processar = df.head(10)

print(f"\nIniciando processamento automatizado de {len(registros_para_processar)} registros...\n")

for row_idx, linha in tqdm(registros_para_processar.iterrows(), total=len(registros_para_processar), desc="Processando Documentos"):
    fonte = linha['fonte']
    resumo_original = linha['resumo_original']
    sentencas_originais_texto = [s.text for s in nlp(resumo_original).sents]

    # Itera sobre os 4 cenários de erro para o mesmo documento
    for tipo_avaliacao in TIPOS_AVALIACAO:
        col_resumo, col_tokens = colunas_map[tipo_avaliacao]
        
        resumo_corrompido = linha[col_resumo]
        
        if col_tokens and pd.notna(linha[col_tokens]):
            tokens_injetados = ast.literal_eval(linha[col_tokens])
        else:
            tokens_injetados = []

        gt = {"tipo_erro": tipo_avaliacao}

        # Omitir barra de progresso individual para não poluir o console
        resultado_fenice = fenice.score_batch(
            [{'document': fonte, 'summary': resumo_corrompido}],
            ground_truths=[gt]
        )[0]

        # Coleta das predições
        alignments = resultado_fenice["alignments"]
        claims_text = [c["summary_claim"] for c in alignments]
        mapa_claim_sentenca = mapear_claims_para_sentencas_originais(claims_text, sentencas_originais_texto)

        erros_detectados_indices = set()

        for idx_claim, claim in enumerate(alignments):
            status = claim["factual_status"]
            classes_fenice = claim.get("error_types_all", [])
            sentenca_origem_idx = mapa_claim_sentenca.get(idx_claim, -1)
            
            # Pareamento de Erros Injetados
            erros_match = []
            for idx_erro, erro in enumerate(tokens_injetados):
                if erro.get("sentenca_index") == sentenca_origem_idx:
                    span_injetado = get_injected_span(erro)
                    if normalizar(span_injetado) in normalizar(claim["summary_claim"]):
                        erros_match.append((idx_erro, erro))

            tipo_detectado_defecto = classes_fenice[0] if classes_fenice else "other"
            if tipo_detectado_defecto not in colunas_detectado: 
                tipo_detectado_defecto = "other"

            # CONTAGEM DE TP, FN e MATRIZ
            if erros_match:
                for idx_e, _ in erros_match:
                    erros_detectados_indices.add(idx_e)
                
                if status in ["contradicted", "not_supported"]:
                    for _, erro in erros_match:
                        global_stats[tipo_avaliacao]["TP"] += 1
                        tipo_real = erro.get("tipo_erro", erro.get("tipo", "other"))
                        if tipo_real not in linhas_real: 
                            tipo_real = "other"
                        
                        if tipo_real in classes_fenice:
                            tipo_det = tipo_real
                            global_stats[tipo_avaliacao]["DIAG_CORRETO"] += 1
                        else:
                            tipo_det = tipo_detectado_defecto
                        
                        global_matrices[tipo_avaliacao].loc[tipo_real, tipo_det] += 1
                else:
                    global_stats[tipo_avaliacao]["FN"] += len(erros_match)
            
            # CONTAGEM DE FP NAS FRASES INOCENTES
            else:
                if status == "contradicted":
                    global_stats[tipo_avaliacao]["FP"] += 1
                    global_stats[tipo_avaliacao]["FP_CONTRADICAO"] += 1
                    global_matrices[tipo_avaliacao].loc["nenhum_fp", tipo_detectado_defecto] += 1
                elif status == "not_supported":
                    global_stats[tipo_avaliacao]["FP"] += 1
                    global_stats[tipo_avaliacao]["FP_ABSTRACAO"] += 1
                    global_matrices[tipo_avaliacao].loc["nenhum_fp", tipo_detectado_defecto] += 1

        # Falsos negativos de erros que nunca foram pareados
        for i, erro in enumerate(tokens_injetados):
            if i not in erros_detectados_indices:
                global_stats[tipo_avaliacao]["FN"] += 1

        global_scores[tipo_avaliacao].append(resultado_fenice['score'])

# =====================================================================
# RELATÓRIO FINAL CONSOLIDADO
# =====================================================================
print("\n\n" + "="*80)
print("🏆 RELATÓRIO DE METAVALIAÇÃO CONSOLIDADO (10 REGISTROS)")
print("="*80)

for tipo in TIPOS_AVALIACAO:
    stats = global_stats[tipo]
    score_medio_bruto = np.mean(global_scores[tipo])
    score_medio_norm = (score_medio_bruto + 1.0) / 2.0
    
    print(f"\n============================================================")
    print(f"📊 MATRIZ GLOBAIS: {tipo.upper()}")
    print(f"============================================================")
    print(global_matrices[tipo].to_string())
    print("------------------------------------------------------------")
    print(f" Verdadeiros Positivos (TP)    : {stats['TP']}")
    print(f" Diagnósticos Corretos        : {stats['DIAG_CORRETO']} (Diagonal Principal)")
    print(f" Falsos Negativos (FN)        : {stats['FN']}")
    print(f" Falsos Positivos Totais (FP) : {stats['FP']}")
    print(f"   ├─ Falsas Alucinações (Contradição) : {stats['FP_CONTRADICAO']}")
    print(f"   └─ Paráfrases Distantes (Neutro)    : {stats['FP_ABSTRACAO']}")
    print(f" Score Médio do FENICE (0 a 1) : {score_medio_norm:.4f}")
    print("============================================================\n")

print("Processamento concluído com sucesso!")