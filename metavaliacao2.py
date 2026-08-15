import sys
import pandas as pd
import ast

# Ajuste o caminho conforme o seu ambiente
sys.path.append("/content/FENICE")
from metric.FENICE import FENICE

# =====================================================================
# CHAVE MESTRA: Defina o tipo de erro que deseja avaliar
# Opções válidas: "entity", "predicate", "coreference"
# =====================================================================
TIPO_AVALIACAO = "predicate" 

if TIPO_AVALIACAO == "entity":
    col_resumo = 'resumo_entity'  # Ou 'resumo_entidade', dependendo de como nomeou
    col_tokens = 'tokens_entity'
elif TIPO_AVALIACAO == "predicate":
    col_resumo = 'resumo_predicado'
    col_tokens = 'tokens_predicado'
elif TIPO_AVALIACAO == "coreference":
    col_resumo = 'resumo_corref'
    col_tokens = 'tokens_corref'
else:
    raise ValueError("TIPO_AVALIACAO inválido. Escolha 'entity', 'predicate' ou 'coreference'.")

# =====================================================================
# 1. INICIALIZAÇÃO DO FENICE
# =====================================================================
print("Inicializando FENICE...")
avaliador = FENICE()
# Configurações do avaliador mantidas no padrão
avaliador.alignments_cache = {}

# =====================================================================
# 2. CARREGAMENTO DOS DADOS (Automatizado pela Chave Mestra)
# =====================================================================
print("Carregando dados do CSV...")
df = pd.read_csv("govreport_erros_pred_corref_entity_realistas.csv")

# Pegando a linha específica
linha = df.iloc[2]
fonte = linha['fonte']

# Puxa dinamicamente a coluna certa do resumo corrompido
resumo_corrompido = linha[col_resumo] 

try:
    # Puxa dinamicamente a coluna certa dos tokens injetados
    tokens_injetados = ast.literal_eval(linha[col_tokens]) 
except (ValueError, SyntaxError):
    # Fallback caso não seja uma lista interpretável
    tokens_injetados = [linha[col_tokens]]

print(f"\n--- INFORMAÇÕES DA INJEÇÃO ({TIPO_AVALIACAO.upper()}) ---")
print(f"ID do Documento: {linha.get('id', 'N/A')}")
print(f"Tokens EXATOS puxados do CSV: {tokens_injetados}")
print(f"Tipo do dado dos tokens: {type(tokens_injetados[0]) if tokens_injetados else 'Vazio'}")
print("------------------------------\n")

# =====================================================================
# 3. EXECUÇÃO DO FENICE
# =====================================================================
print("Rodando a avaliação do FENICE...")
resultado = avaliador.score_batch([{'document': fonte, 'summary': resumo_corrompido}])[0]

# =====================================================================
# 4. FUNÇÃO DE CHECAGEM (Ancoragem por Sentença + Diagnóstico)
# =====================================================================
def checar_acerto_fenice(lista_erros_injetados, json_do_fenice, tipo_esperado="entity"):
    resultados = {"TP": 0, "FN": 0, "FP": 0, "Diag_Correto": 0}
    alignments = json_do_fenice.get('alignments', [])
    
    print(f"\n--- RELATÓRIO DE DIAGNÓSTICO (Tipo Injetado: {tipo_esperado}) ---")
    
    for i, alignment in enumerate(alignments):
        claim_text = alignment['summary_claim']
        status = alignment['factual_status']
        erros_detectados = alignment.get('error_types_all', [])
        
        # Procura se a sentença atual (claim i) sofreu injeção
        erro_na_sentenca = None
        for erro in lista_erros_injetados:
            if isinstance(erro, dict) and erro.get('sentenca_index') == i:
                erro_na_sentenca = erro
                break
                
        # CASO 1: NÓS INJETAMOS UM ERRO NESTA SENTENÇA
        if erro_na_sentenca:
            tipo_injetado = erro_na_sentenca.get('tipo_erro', tipo_esperado)
            
            if status in ['contradicted', 'not_supported']:
                resultados["TP"] += 1
                if tipo_injetado in erros_detectados:
                    print(f"✅ [SUCESSO] Claim {i} ({tipo_injetado}): FENICE detectou e diagnosticou corretamente.")
                    resultados["Diag_Correto"] += 1
                else:
                    print(f"⚠️ [IMPRECISO] Claim {i} ({tipo_injetado}): FENICE achou o erro, mas diagnosticou como {erros_detectados}.")
            else:
                print(f"❌ [FALHA] Claim {i} ({tipo_injetado}): FENICE deixou a alucinação passar (Marcou como {status}).")
                resultados["FN"] += 1
                
        # CASO 2: NÓS *NÃO* INJETAMOS ERRO NESTA SENTENÇA (Pode ser FP ou Alucinação original do documento)
        else:
            if status in ['contradicted', 'not_supported']:
                print(f"🔎 [FALSO POSITIVO OU DESCOBERTA] Claim {i}: Marcado como {status}. Motivo FENICE: {erros_detectados}.")
                resultados["FP"] += 1

    return resultados

# =====================================================================
# 5. RESULTADOS FINAIS
# =====================================================================
print("\n=== RESUMO GERAL DO FENICE ===")
print(f"Score global: {resultado['score']:.4f}")
print(f"Status counts: {resultado['status_counts']}")
print(f"Error type counts: {resultado['error_type_counts']}")

# Passando a chave mestre também para a função de checagem
metricas = checar_acerto_fenice(tokens_injetados, resultado, tipo_esperado=TIPO_AVALIACAO)

print("\n=== MATRIZ DE CONFUSÃO PARA ESTE DOCUMENTO ===")
print(f"Verdadeiros Positivos (TP) - Erros injetados e pegos: {metricas['TP']}")
print(f"Diagnósticos Precisos (O FENICE cravou a categoria certa): {metricas['Diag_Correto']}")
print(f"Falsos Negativos (FN) - Erros injetados que passaram batidos: {metricas['FN']}")
print(f"Falsos Positivos / Descobertas: {metricas['FP']}")

# =====================================================================
# 6. RAIO-X DE ERROS DE PREDICADO E CORREFERÊNCIA
# =====================================================================
print("\n" + "="*50)
print("🔎 INVESTIGAÇÃO DETALHADA: PREDICATE E COREFERENCE")
print("="*50)

alignments = resultado.get('alignments', [])
encontrou_algum = False

for i, alignment in enumerate(alignments):
    erros_detectados = alignment.get('error_types_all', [])
    
    if alignment['factual_status'] == 'contradicted' and ('predicate' in erros_detectados or 'coreference' in erros_detectados):
        encontrou_algum = True
        print(f"\n🚨 CLAIM {i} | Tipos de Erro Mapeados: {erros_detectados}")
        print(f"👉 Frase do Resumo: {alignment['summary_claim']}")
        
        evidencia = alignment.get('source_passage', 'Não disponível')
        probs = alignment.get('probs', [0, 0, 0])
        print(f"📖 Evidência original: {evidencia}")
        print(f"⚖️ Probabilidades (Apoio, Contradição, Neutro): [{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]")

if not encontrou_algum:
    print("\nNenhum erro de 'predicate' ou 'coreference' foi encontrado neste documento específico.")