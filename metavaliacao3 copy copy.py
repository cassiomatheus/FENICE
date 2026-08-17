import os
# Impede o crash de fragmentação de memória na RTX 3000
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import time

import random

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

    import os
# Força o sistema a usar apenas 6 threads para pré-processamento

os.environ["OMP_NUM_THREADS"] = "6"          # limite de threads OpenMP
os.environ["MKL_NUM_THREADS"] = "6"          # limite do MKL
os.environ["NUMEXPR_NUM_THREADS"] = "6"      # numexpr

torch.set_num_threads(2) # Evita que a CPU dê saltos de 100% de consumo

# Desliga o cálculo de gradientes (economiza uns 2GB de VRAM)
torch.set_grad_enabled(False)

# Ajuste o caminho conforme o seu ambiente
sys.path.append("/content/FENICE")
from metric.FENICE import FENICE

# Força o uso do modelo médio para os logs textuais
nlp = spacy.load("en_core_web_md")

# =====================================================================
# CONFIGURAÇÃO E CHAVE MESTRA
# =====================================================================
TIPO_AVALIACAO = "entity" 
#ARQUIVO = "Gerador_erros/govreport_erros_pred_corref_entity_realistas_v1.csv"
ARQUIVO = "Gerador_erros/govreport_erros_pred_corref_entity_controlados_v5.csv"

if TIPO_AVALIACAO == "entity":
    col_resumo = 'resumo_entity'
    col_tokens = 'tokens_entity'

elif TIPO_AVALIACAO == "predicate":
    col_resumo = 'resumo_predicado'
    col_tokens = 'tokens_predicado'

elif TIPO_AVALIACAO == "coreference":
    col_resumo = 'resumo_corref'
    col_tokens = 'tokens_corref'

elif TIPO_AVALIACAO == "original":
    col_resumo = 'resumo_original'
    col_tokens = None  # <-- Ajuste crucial: O original não tem tokens injetados
else:
    raise ValueError("TIPO_AVALIACAO inválido.")

# =====================================================================
# CARREGAMENTO DE MODELOS
# =====================================================================
print("Carregando modelos de alinhamento...")
# 1. Carrega o FENICE na GPU com impacto mínimo
fenice = FENICE(
    use_coref=False,
    doc_chunk_overlap=2,
    claim_extractor_batch_size=64,  
    nli_batch_size=64,  # Processa as sentenças de 2 em 2 (mantém a carga elétrica super suave)

    num_sent_per_paragraph=3, # Janelas de exatas 3 sentenças
    sliding_paragraphs=True,  # Ativa o formato Sliding Window
    sliding_stride=1,         # Avança 1 sentença por vez (sobreposição máxima)
    doc_level_nli=False       # (Opcional) Pode desligar o chunk de doc_level se quiser economizar RAM, pois a janela de 3 já cobre 99% dos gaps.
)


# 2. Carrega o Alinhador TAMBÉM na GPU (Substitua o 'cpu' por 'cuda')
#encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu") #MODIFICADO

#encoder = SentenceTransformer("all-MiniLM-L6-v2")
#encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

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
# PROCESSAMENTO DO REGISTRO
# =====================================================================
df = pd.read_csv(ARQUIVO)
linha = df.iloc[0] # Registro 0 (Cell-cultured meat)

fonte = linha['fonte']
resumo_original = linha['resumo_original']
resumo_corrompido = linha[col_resumo]
#tokens_injetados = ast.literal_eval(linha[col_tokens])

if col_tokens:
    tokens_injetados = ast.literal_eval(linha[col_tokens])
else:
    tokens_injetados = []  # Garante lista vazia para o resumo original

# Extrai as frases do resumo original e do corrompido para o relatório de prints
sentencas_originais_texto = [s.text for s in nlp(resumo_original).sents]
sentencas_corrompidas_texto = [s.text for s in nlp(resumo_corrompido).sents]

# Faxina preventiva ANTES de processar
#gc.collect()
#torch.cuda.empty_cache()
#torch.cuda.synchronize()

print("\nExecutando FENICE (Processamento NLI)...")
#resultado_fenice = fenice.score_batch([{'document': fonte, 'summary': resumo_corrompido}])[0]

# Para cada linha, crie um dicionário com o tipo real
gt = {"tipo_erro": TIPO_AVALIACAO}   # "entity", "predicate", "coreference"

resultado_fenice = fenice.score_batch(
    [{'document': fonte, 'summary': resumo_corrompido}],
    ground_truths=[gt]
)[0]

# =====================================================================
# AVALIAÇÃO E MATRIZ DE CONFUSÃO CRUZADA (CROSS-TABULATION)
# =====================================================================
alignments = resultado_fenice["alignments"]
claims_text = [c["summary_claim"] for c in alignments]

# Alinhamento vetorial
mapa_claim_sentenca = mapear_claims_para_sentencas_originais(claims_text, sentencas_originais_texto)

# SUBSTUTUA ESTA LINHA:
# stats = {"TP": 0, "FN": 0, "FP": 0, "DIAG_CORRETO": 0}

# POR ESTA VERSÃO ATUALIZADA:
stats = {
    "TP": 0, 
    "FN": 0, 
    "FP": 0, 
    "FP_CONTRADICAO": 0, 
    "FP_ABSTRACAO": 0, 
    "DIAG_CORRETO": 0
}
erros_detectados_indices = set()

# 1. INICIALIZA A MATRIZ CRUZADA (Injetado x Detectado)
#categorias_erro = ["predicate", "entity", "coreference", "other"]
#matriz_cruzada = pd.DataFrame(0, index=categorias_erro, columns=categorias_erro)

# =====================================================================
# 1. INICIALIZA A MATRIZ CRUZADA (INCLUINDO FALSOS POSITIVOS)
# =====================================================================
# Colunas: O que o FENICE diz que achou
#colunas_detectado = ["predicate", "entity", "referential_inconsistency", "other"]

# Linhas: O que realmente estava na frase (incluindo o cenário onde não injetamos nada)
#linhas_real = ["predicate", "entity", "referential_inconsistency", "other", "nenhum_fp"]


# Colunas: O que o FENICE diz que achou (Single-Label via Argmax)
colunas_detectado = ["predicate", "entity", "coreference", "other"]

# Linhas: O que realmente estava na frase (incluindo o cenário onde não injetamos nada)
linhas_real = ["predicate", "entity", "coreference", "other", "nenhum_fp"]

matriz_cruzada = pd.DataFrame(0, index=linhas_real, columns=colunas_detectado)
matriz_cruzada.index.name = "Erro Real (Injetado)"
matriz_cruzada.columns.name = "Detectado (FENICE)"

#print("\n" + "="*80)
#print(f"🔬 INVESTIGAÇÃO MICROSCÓPICA DE CLAIMS: {TIPO_AVALIACAO.upper()}")
#print("="*80)

# =====================================================================
# 2. AUDITORIA DOS CLAIMS
# =====================================================================
for idx, claim in enumerate(alignments):
    status = claim["factual_status"]
    classes_fenice = claim.get("error_types_all", [])
    sentenca_origem_idx = mapa_claim_sentenca.get(idx, -1)
    
    # 1. Coleta TODOS os erros injetados pertencentes a esta claim
    erros_match = []
    for idx_erro, erro in enumerate(tokens_injetados):
        if erro.get("sentenca_index") == sentenca_origem_idx:
            span_injetado = get_injected_span(erro)
            if normalizar(span_injetado) in normalizar(claim["summary_claim"]):
                erros_match.append((idx_erro, erro))
                
    # Extrai o tipo principal detectado como fallback
    tipo_detectado_defecto = classes_fenice[0] if classes_fenice else "other"
    if tipo_detectado_defecto not in colunas_detectado: 
        tipo_detectado_defecto = "other"

    # -----------------------------------------------------------------
    # CASO A: A CLAIM CONTÉM ERROS INJETADOS
    # -----------------------------------------------------------------
    if erros_match:
        # Marca os índices de todos os erros da frase como mapeados
        for idx_e, _ in erros_match:
            erros_detectados_indices.add(idx_e)
        
        if status in ["contradicted", "not_supported"]:
            # Processa cada erro presente na frase individualmente
            for _, erro in erros_match:
                stats["TP"] += 1
                
                tipo_real = erro.get("tipo_erro", erro.get("tipo", "other"))
                if tipo_real not in linhas_real: 
                    tipo_real = "other"
                
                # Regra Multi-rótulo (Any-Match) por erro
                if tipo_real in classes_fenice:
                    tipo_det = tipo_real  # Força o acerto para a Diagonal Principal
                    stats["DIAG_CORRETO"] += 1
                else:
                    tipo_det = tipo_detectado_defecto
                
                matriz_cruzada.loc[tipo_real, tipo_det] += 1
        else:
            # Se o status foi SUPPORTED, a IA foi enganada -> soma FN para cada erro da frase
            stats["FN"] += len(erros_match)

    # -----------------------------------------------------------------
    # CASO B: FALSOS POSITIVOS (Claim inocente)
    # -----------------------------------------------------------------
    else:
        if status == "contradicted":
            stats["FP"] += 1
            stats["FP_CONTRADICAO"] += 1  # Alucinação grave (1 alarme falso)
            matriz_cruzada.loc["nenhum_fp", tipo_detectado_defecto] += 1
            
        elif status == "not_supported":
            stats["FP"] += 1
            stats["FP_ABSTRACAO"] += 1    # Desvio por paráfrase/abstração
            matriz_cruzada.loc["nenhum_fp", tipo_detectado_defecto] += 1

# Limpeza final: computa Falsos Negativos de erros que nem chegaram a ser pareados com um claim
for i, erro in enumerate(tokens_injetados):
    if i not in erros_detectados_indices:
        stats["FN"] += 1















# =====================================================================
# 3. RAIO-X DAS COMPARAÇÕES (VISUALIZAÇÃO DETALHADA)
# =====================================================================
print("\n" + "="*80)
print("🔍 RAIO-X DAS COMPARAÇÕES (CLAIM POR CLAIM)")
print("="*80)

for idx, claim in enumerate(alignments):
    print(f"\n[{idx + 1}] CLAIM (Frase do Resumo):")
    print(f"    {claim['summary_claim']}")
    
    print(f"\n    SENTENÇA RECUPERADA (A Prova do Texto Original):")
    print(f"    {claim['source_passage']}")
    
    # Extrai as probabilidades da rede neural
    probs = claim.get("probs", [0, 0, 0])
    ent = probs[0] * 100
    contr = probs[1] * 100
    neut = probs[2] * 100
    
    print(f"\n    🧠 NLI (DeBERTa): Apoio: {ent:.1f}% | Contradição: {contr:.1f}% | Neutro: {neut:.1f}%")
    print(f"    📊 Status Final:  {claim['factual_status'].upper()}")
    print(f"    🔬 Regras spaCy:  {claim.get('error_types_all', [])}")
    
    # Procura se nós havíamos injetado um erro nesta frase específica
    sentenca_origem_idx = mapa_claim_sentenca.get(idx, -1)
    erro_match = None
    for erro in tokens_injetados:
        if erro.get("sentenca_index") == sentenca_origem_idx:
            span_injetado = get_injected_span(erro)
            if normalizar(span_injetado) in normalizar(claim["summary_claim"]):
                erro_match = erro
                break
    
    if erro_match:
        print(f"    ⚠️ ERRO INJETADO PELO GERADOR: {erro_match}")
    else:
        print(f"    ✅ FRASE INOCENTE (Nenhum erro injetado aqui)")
    print("-" * 80)











# =====================================================================
# EXIBIÇÃO DOS RESULTADOS
# =====================================================================
print("\n" + "="*60)
print(f"📊 MATRIZ DE CONFUSÃO POR TIPO DE ERRO ({TIPO_AVALIACAO.upper()})")
print("="*60)
print(matriz_cruzada.to_string())
print("="*50)

print("="*50)
print(f" Verdadeiros Positivos (TP)    : {stats['TP']}")
print(f" Diagnósticos Corretos        : {stats['DIAG_CORRETO']} (Diagonal Principal)")
print(f" Falsos Negativos (FN)        : {stats['FN']}")
print(f" Falsos Positivos Totais (FP) : {stats['FP']}")
print(f"   ├─ Falsas Alucinações (Contradição) : {stats['FP_CONTRADICAO']}")
print(f"   └─ Paráfrases Distantes (Neutro)    : {stats['FP_ABSTRACAO']}")
print("="*50)

# =====================================================================
# NORMALIZAÇÃO DO SCORE [-1, 1] PARA A ESCALA [0, 1]
# Fórmula: S_norm = (S_fenice - min) / (max - min) = (S_fenice + 1) / 2
# =====================================================================
score_bruto = resultado_fenice['score']
score_normalizado = (score_bruto + 1.0) / 2.0

print("\n=== RESUMO GERAL DO FENICE ===")
print(f"Score Bruto (Padrão do Artigo [-1 a 1]) : {score_bruto:.4f}")
print(f"Score Normalizado (Gráficos [0 a 1])    : {score_normalizado:.4f}  <-- (Escala de 0% a 100% de Fidelidade)")
print(f"Status counts: {resultado_fenice['status_counts']}")
print(f"Error type counts: {resultado_fenice['error_type_counts']}")

