import sys
import pandas as pd
import ast
import re
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Ajuste o caminho conforme o seu ambiente
sys.path.append("/content/FENICE")
from metric.FENICE import FENICE

# Força o uso do modelo médio para os logs textuais
nlp = spacy.load("en_core_web_md")

# =====================================================================
# CONFIGURAÇÃO E CHAVE MESTRA
# =====================================================================
TIPO_AVALIACAO = "predicate" 
ARQUIVO = "govreport_erros_pred_corref_entity_realistas_v3.csv"

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
    col_tokens = 'tokens_corref'
else:
    raise ValueError("TIPO_AVALIACAO inválido.")

# =====================================================================
# CARREGAMENTO DE MODELOS
# =====================================================================
print("Carregando modelos de alinhamento...")
fenice = FENICE(
    use_coref=True, 
    doc_chunk_overlap=2 # Repete as 2 últimas frases do bloco anterior no novo bloco
)
#encoder = SentenceTransformer("all-MiniLM-L6-v2")

encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

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
linha = df.iloc[1] # Registro 0 (Cell-cultured meat)

fonte = linha['fonte']
resumo_original = linha['resumo_original']
resumo_corrompido = linha[col_resumo]
tokens_injetados = ast.literal_eval(linha[col_tokens])

# Extrai as frases do resumo original e do corrompido para o relatório de prints
sentencas_originais_texto = [s.text for s in nlp(resumo_original).sents]
sentencas_corrompidas_texto = [s.text for s in nlp(resumo_corrompido).sents]

print("\nExecutando FENICE (Processamento NLI)...")
resultado_fenice = fenice.score_batch([{'document': fonte, 'summary': resumo_corrompido}])[0]

# =====================================================================
# AVALIAÇÃO E PRINT DETALHADO (AUDITORIA DO TERMINAL)
# =====================================================================
alignments = resultado_fenice["alignments"]
claims_text = [c["summary_claim"] for c in alignments]

# Alinhamento vetorial
mapa_claim_sentenca = mapear_claims_para_sentencas_originais(claims_text, sentencas_originais_texto)

stats = {"TP": 0, "FN": 0, "FP": 0, "DIAG_CORRETO": 0}
erros_detectados_indices = set()

print("\n" + "="*80)
print(f"🔬 INVESTIGAÇÃO MICROSCÓPICA DE CLAIMS: {TIPO_AVALIACAO.upper()}")
print("="*80)

for idx, claim in enumerate(alignments):
    status = claim["factual_status"]
    classes_fenice = claim.get("error_types_all", [])
    sentenca_origem_idx = mapa_claim_sentenca[idx]
    
    # Busca se a sentença de origem mapeada pelo SentenceTransformer tem um erro injetado
    erro_match = None
    for erro in tokens_injetados:
        if erro.get("sentenca_index") == sentenca_origem_idx:
            span_injetado = get_injected_span(erro)
            if normalizar(span_injetado) in normalizar(claim["summary_claim"]):
                erro_match = erro
                break
                
    # CASO A: O claim contém um erro que nós injetamos
    if erro_match:
        erros_detectados_indices.add(tokens_injetados.index(erro_match))
        span_txt = get_injected_span(erro_match)
        verbo_orig = erro_match.get('verbo_original', 'N/A')
        
        print(f"\n🚨 [CLAIM MODIFICADO MAPPED] -> Claim ID: {idx} | Relacionado à Sentença Original: {sentenca_origem_idx}")
        print(f" 📝 Texto do Claim Extraído: '{claim['summary_claim']}'")
        print(f" 🔄 Mutação Aplicada: Verbo Original '{verbo_orig}' $\rightarrow$ Injetado: '{span_txt}'")
        print(f" 📖 Frase Original Completa : {sentencas_originais_texto[sentenca_origem_idx]}")
        print(f" 🛑 Frase Corrompida Completa: {sentencas_corrompidas_texto[sentenca_origem_idx]}")
        print(f" ⚖️ Veredito FENICE: Factual Status = '{status}' | Erros Atribuídos = {classes_fenice}")
        
        if status in ["contradicted", "not_supported"]:
            stats["TP"] += 1
            if erro_match.get("tipo_erro", TIPO_AVALIACAO) in classes_fenice:
                print(" 🟢 [AUDITORIA]: SUCESSO. O FENICE detectou a contradição e rotulou a classe certa.")
                stats["DIAG_CORRETO"] += 1
            else:
                print(" 🟡 [AUDITORIA]: IMPRECISO. Detectou a inconsistência, mas errou a tipagem do erro.")
        else:
            print(" 🔴 [AUDITORIA]: FALSO NEGATIVO. O erro passou batido pelo avaliador.")
            stats["FN"] += 1
            
    # CASO B: O claim NÃO foi modificado por nós, mas o FENICE acusou inconsistência (Falso Positivo / Ruído Nativo)
    else:
        if status in ["contradicted", "not_supported"]:
            stats["FP"] += 1
            print(f"\n🔎 [ALUCINAÇÃO NATIVA / FP] -> Claim ID: {idx} | Relacionado à Sentença Original: {sentenca_origem_idx}")
            print(f" 📝 Texto do Claim: '{claim['summary_claim']}'")
            print(f" 📖 Baseado na Frase Original: {sentencas_originais_texto[sentenca_origem_idx]}")
            print(f" ⚖️ Motivo do FENICE: Status = '{status}' | Erros Atribuídos = {classes_fenice}")

# Verifica se sobrou algum erro sem claim associado
for i, erro in enumerate(tokens_injetados):
    if i not in erros_detectados_indices:
        print(f"\n👻 [ERRO FANTASMA] -> O erro '{get_injected_span(erro)}' (Sentença {erro.get('sentenca_index')}) foi omitido pelo ClaimExtractor.")
        stats["FN"] += 1

print("\n" + "="*40)
print("📊 MATRIZ DE CONFUSÃO FINAL DO DOCUMENTO")
print("="*40)
print(f" Verdadeiros Positivos (TP) : {stats['TP']}")
print(f" Diagnósticos Corretos     : {stats['DIAG_CORRETO']}")
print(f" Falsos Negativos (FN)     : {stats['FN']}")
print(f" Falsos Positivos (FP)     : {stats['FP']}")
print("="*40)

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