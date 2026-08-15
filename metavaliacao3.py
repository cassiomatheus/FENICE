import sys
import pandas as pd
import ast
import re

# Ajuste o caminho conforme o seu ambiente
sys.path.append("/content/FENICE")
from metric.FENICE import FENICE

# =====================================================================
# CHAVE MESTRA: Defina o tipo de erro que deseja avaliar
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
else:
    raise ValueError("TIPO_AVALIACAO inválido.")

# =====================================================================
# 1. INICIALIZAÇÃO DO FENICE
# =====================================================================
print("Inicializando FENICE...")
avaliador = FENICE()
avaliador.alignments_cache = {}

# =====================================================================
# 2. CARREGAMENTO DO DADO (REGISTRO INDIVIDUAL)
# =====================================================================
print(f"\nCarregando dados do CSV: {ARQUIVO}")
df = pd.read_csv(ARQUIVO)

# Escolhendo a linha 0 para o teste individual
LINHA_TESTE = 0
linha = df.iloc[LINHA_TESTE]
fonte = linha['fonte']
resumo_corrompido = linha[col_resumo] 

try:
    tokens_injetados = ast.literal_eval(linha[col_tokens]) 
except (ValueError, SyntaxError):
    tokens_injetados = [linha[col_tokens]]

print(f"\n--- INFORMAÇÕES DA INJEÇÃO ({TIPO_AVALIACAO.upper()}) ---")
print(f"ID do Documento: {linha.get('id', 'N/A')}")
print(f"Erros injetados na linha {LINHA_TESTE}:")
for t in tokens_injetados:
    print(f" - {t}")
print("------------------------------\n")

# =====================================================================
# 3. EXECUÇÃO DO FENICE
# =====================================================================
print("Rodando a avaliação do FENICE...")
resultado = avaliador.score_batch([{'document': fonte, 'summary': resumo_corrompido}])[0]

# =====================================================================
# 4. FUNÇÃO DE CHECAGEM POR SPAN OVERLAP
# =====================================================================
def get_injected_span(erro):
    """Recupera o texto corrompido independentemente do gerador V1, V2 ou V3."""
    return erro.get('token') or erro.get('injetado') or erro.get('novo_sujeito') or ""

def normalize_text(text):
    """Remove pontuação e coloca em minúsculo para busca de overlap segura."""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip())

def checar_acerto_fenice_por_span(lista_erros_injetados, json_do_fenice, tipo_esperado):
    resultados = {"TP": 0, "FN": 0, "FP": 0, "Diag_Correto": 0}
    alignments = json_do_fenice.get('alignments', [])
    
    # Cria uma lista de spans injetados que precisamos encontrar nos claims
    spans_pendentes = [erro for erro in lista_erros_injetados if isinstance(erro, dict)]
    
    print(f"\n--- RELATÓRIO DE DIAGNÓSTICO POR SPAN (Tipo: {tipo_esperado}) ---")
    
    for i, alignment in enumerate(alignments):
        claim_text = alignment['summary_claim']
        status = alignment['factual_status']
        erros_detectados = alignment.get('error_types_all', [])
        
        claim_norm = normalize_text(claim_text)
        
        # Verifica se o claim atual absorveu algum dos nossos spans corrompidos
        erro_no_claim = None
        for erro in spans_pendentes:
            span_injetado = get_injected_span(erro)
            span_norm = normalize_text(span_injetado)
            
            # OVERLAP MATCH: O span corrompido está textualmente dentro do claim?
            if span_norm and span_norm in claim_norm:
                erro_no_claim = erro
                break
                
        if erro_no_claim:
            tipo_injetado = erro_no_claim.get('tipo_erro', tipo_esperado)
            span_injetado = get_injected_span(erro_no_claim)
            
            # Remove da lista de pendentes para não contar o mesmo erro duas vezes
            spans_pendentes.remove(erro_no_claim)
            
            if status in ['contradicted', 'not_supported']:
                resultados["TP"] += 1
                if tipo_injetado in erros_detectados:
                    print(f"✅ [SUCESSO] Claim {i} contém '{span_injetado}'. FENICE cravou o diagnóstico: {tipo_injetado}.")
                    resultados["Diag_Correto"] += 1
                else:
                    print(f"⚠️ [IMPRECISO] Claim {i} contém '{span_injetado}'. Deu erro, mas classificou como {erros_detectados}.")
            else:
                print(f"❌ [FALHA] Claim {i} absorveu a alucinação '{span_injetado}', mas deixou passar (Status: {status}).")
                resultados["FN"] += 1
                
        else:
            # Se não tem nenhum span nosso, mas apontou erro, contamos como FP (ou ruído nativo)
            if status in ['contradicted', 'not_supported']:
                resultados["FP"] += 1

    # Os erros que sobraram em 'spans_pendentes' são Falsos Negativos fantasmas
    # Isso ocorre se o extrator de claims fragmentou mal a frase e apagou a nossa palavra injetada.
    for erro_perdido in spans_pendentes:
        span_perdido = get_injected_span(erro_perdido)
        print(f"👻 [FANTASMA] A alucinação '{span_perdido}' não sobreviveu à extração de claims. (FN computado)")
        resultados["FN"] += 1

    return resultados

# =====================================================================
# 5. RESULTADOS FINAIS
# =====================================================================
print("\n=== RESUMO GERAL DO FENICE ===")
print(f"Score global: {resultado['score']:.4f}")
print(f"Status counts: {resultado['status_counts']}")
print(f"Error type counts: {resultado['error_type_counts']}")

metricas = checar_acerto_fenice_por_span(tokens_injetados, resultado, tipo_esperado=TIPO_AVALIACAO)

print("\n=== MATRIZ DE CONFUSÃO PARA O REGISTRO INDIVIDUAL ===")
print(f"Verdadeiros Positivos (TP) - Span pego: {metricas['TP']}")
print(f"Diagnósticos Precisos (Categoria exata): {metricas['Diag_Correto']}")
print(f"Falsos Negativos (FN) - Span ignorado ou Fantasma: {metricas['FN']}")
print(f"Falsos Positivos / Alucinações Nativas: {metricas['FP']}")