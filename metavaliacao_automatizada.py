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
TIPOS_AVALIACAO = ["predicate", "entity", "coreference", "original"] # Ordem ajustada para bater com o seu esboço

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
# INICIALIZAÇÃO DAS LISTAS DE REGISTROS PARA EXPORTAÇÃO
# =====================================================================
linhas_real = ["predicate", "entity", "coreference", "other", "nenhum_fp"]
colunas_detectado = ["predicate", "entity", "coreference", "other"]

registros_matrizes = []
registros_metricas = []

# =====================================================================
# PIPELINE DE PROCESSAMENTO
# =====================================================================
df = pd.read_csv(ARQUIVO)
registros_para_processar = df.head(10) # Processando 10 registros conforme solicitado

print(f"\nIniciando processamento automatizado de {len(registros_para_processar)} registros...\n")

for row_idx, linha in tqdm(registros_para_processar.iterrows(), total=len(registros_para_processar), desc="Processando Documentos"):
    fonte = linha['fonte']
    resumo_original = linha['resumo_original']
    sentencas_originais_texto = [s.text for s in nlp(resumo_original).sents]

    # Itera sobre os 4 cenários para o mesmo documento
    for tipo_avaliacao in TIPOS_AVALIACAO:
        col_resumo, col_tokens = colunas_map[tipo_avaliacao]
        resumo_corrompido = linha[col_resumo]
        
        if col_tokens and pd.notna(linha[col_tokens]):
            tokens_injetados = ast.literal_eval(linha[col_tokens])
        else:
            tokens_injetados = []

        gt = {"tipo_erro": tipo_avaliacao}

        # Roda o FENICE
        resultado_fenice = fenice.score_batch([{'document': fonte, 'summary': resumo_corrompido}], ground_truths=[gt])[0]

        # Configura as variáveis locais para este Registro específico
        local_stats = {"TP": 0, "FN": 0, "FP": 0, "FP_CONTRADICAO": 0, "FP_ABSTRACAO": 0, "DIAG_CORRETO": 0}
        local_matrix = pd.DataFrame(0, index=linhas_real, columns=colunas_detectado)
        erros_detectados_indices = set()

        alignments = resultado_fenice["alignments"]
        claims_text = [c["summary_claim"] for c in alignments]
        mapa_claim_sentenca = mapear_claims_para_sentencas_originais(claims_text, sentencas_originais_texto)

        for idx_claim, claim in enumerate(alignments):
            status = claim["factual_status"]
            classes_fenice = claim.get("error_types_all", [])
            sentenca_origem_idx = mapa_claim_sentenca.get(idx_claim, -1)
            
            erros_match = []
            for idx_erro, erro in enumerate(tokens_injetados):
                if erro.get("sentenca_index") == sentenca_origem_idx:
                    span_injetado = get_injected_span(erro)
                    if normalizar(span_injetado) in normalizar(claim["summary_claim"]):
                        erros_match.append((idx_erro, erro))

            tipo_detectado_defecto = classes_fenice[0] if classes_fenice else "other"
            if tipo_detectado_defecto not in colunas_detectado: 
                tipo_detectado_defecto = "other"

            # CONTAGEM LOCAL
            if erros_match:
                for idx_e, _ in erros_match:
                    erros_detectados_indices.add(idx_e)
                
                if status in ["contradicted", "not_supported"]:
                    for _, erro in erros_match:
                        local_stats["TP"] += 1
                        tipo_real = erro.get("tipo_erro", erro.get("tipo", "other"))
                        if tipo_real not in linhas_real: tipo_real = "other"
                        
                        if tipo_real in classes_fenice:
                            tipo_det = tipo_real
                            local_stats["DIAG_CORRETO"] += 1
                        else:
                            tipo_det = tipo_detectado_defecto
                        
                        local_matrix.loc[tipo_real, tipo_det] += 1
                else:
                    local_stats["FN"] += len(erros_match)
            else:
                if status == "contradicted":
                    local_stats["FP"] += 1
                    local_stats["FP_CONTRADICAO"] += 1
                    local_matrix.loc["nenhum_fp", tipo_detectado_defecto] += 1
                elif status == "not_supported":
                    local_stats["FP"] += 1
                    local_stats["FP_ABSTRACAO"] += 1
                    local_matrix.loc["nenhum_fp", tipo_detectado_defecto] += 1

        for i, erro in enumerate(tokens_injetados):
            if i not in erros_detectados_indices:
                local_stats["FN"] += 1

        # =====================================================================
        # 1. SALVANDO OS DADOS DA MATRIZ (TABELA 1 DO SEU ESBOÇO)
        # =====================================================================
        nome_cenario = "Normal" if tipo_avaliacao == "original" else tipo_avaliacao
        for erro_real in linhas_real:
            registros_matrizes.append({
                'Registro': row_idx,
                'Cenarios\Tipos': nome_cenario,
                'Erro Real (Injetado)': erro_real,
                'predicate': local_matrix.loc[erro_real, 'predicate'],
                'entity': local_matrix.loc[erro_real, 'entity'],
                'coreference': local_matrix.loc[erro_real, 'coreference'],
                'other': local_matrix.loc[erro_real, 'other']
            })

        # =====================================================================
        # 2. SALVANDO OS DADOS DE MÉTRICAS (TABELA 2 DO SEU ESBOÇO)
        # =====================================================================
        score_bruto = resultado_fenice['score']
        score_norm = (score_bruto + 1.0) / 2.0
        
        registros_metricas.append({
            'Registro': row_idx,
            'Cenario': nome_cenario,
            'Score Bruto': score_bruto,
            'Score Normalizado': score_norm,
            'Verdadeiros Positivos (TP)': local_stats['TP'],
            'Diagnósticos Corretos': local_stats['DIAG_CORRETO'],
            'Falsos Negativos (FN)': local_stats['FN'],
            'Falsos Positivos Totais (FP)': local_stats['FP'],
            'Falsas Alucinações (FP)': local_stats['FP_CONTRADICAO'],
            'Paráfrases Distantes (FP)': local_stats['FP_ABSTRACAO'],
            'Supported (Contagem)': resultado_fenice['status_counts'].get('supported', 0),
            'Contradicted (Contagem)': resultado_fenice['status_counts'].get('contradicted', 0)
        })

# =====================================================================
# GERAÇÃO DO ARQUIVO EXCEL
# =====================================================================
print("\nConvertendo dados e gerando arquivo Excel...")

df_matrizes = pd.DataFrame(registros_matrizes)
df_metricas = pd.DataFrame(registros_metricas)

# Cria o MultiIndex para mesclar visualmente as células como no seu esboço
df_matrizes.set_index(['Registro', 'Cenarios\Tipos', 'Erro Real (Injetado)'], inplace=True)

caminho_saida = 'Resultados_Avaliacao_Por_Registro.xlsx'

with pd.ExcelWriter(caminho_saida, engine='xlsxwriter') as writer:
    # Salva a primeira aba (Matrizes)
    df_matrizes.to_excel(writer, sheet_name='Matrizes de Confusao')
    
    # Salva a segunda aba (Métricas Gerais)
    df_metricas.to_excel(writer, sheet_name='Metricas e Scores', index=False)
    
    # Ajuste visual das larguras das colunas
    workbook = writer.book
    worksheet_mat = writer.sheets['Matrizes de Confusao']
    worksheet_met = writer.sheets['Metricas e Scores']
    
    worksheet_mat.set_column('A:A', 10)
    worksheet_mat.set_column('B:B', 20)
    worksheet_mat.set_column('C:C', 25)
    worksheet_mat.set_column('D:G', 15)
    
    worksheet_met.set_column('A:B', 15)
    worksheet_met.set_column('C:L', 20)

print(f"✅ Sucesso! O arquivo '{caminho_saida}' foi gerado.")