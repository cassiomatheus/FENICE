import spacy
import nltk
from nltk.corpus import wordnet as wn
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import json

# =======================================================
# 1. PREPARAÇÃO
# =======================================================
nltk.download('wordnet')
nltk.download('omw-1.4')

print("Carregando spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.max_length = 2000000 

# =======================================================
# 2. MINERAÇÃO DE VERBOS (Alvo: Conjunto de TESTE)
# =======================================================
print("Baixando/Carregando o GovReport (Conjunto de Teste)...")
# Usamos todo o conjunto de teste para extrair a distribuição real do que vamos avaliar
dataset = load_dataset("ccdv/govreport-summarization", split="test")

contador_verbos = Counter()

print(f"Minerando verbos de {len(dataset)} documentos de teste...")
for example in tqdm(dataset, desc="Analisando textos"):
    try:
        doc = nlp(example['report'])
        for token in doc:
            if token.pos_ == "VERB" and token.is_alpha:
                contador_verbos[token.lemma_.lower()] += 1
    except ValueError:
        continue # Pula arquivos excepcionalmente grandes se houver erro de memória

# O SEGREDO: Em vez de top 500, pegamos qualquer verbo que apareça pelo menos 5 vezes no dataset!
# Isso elimina erros de digitação, mas mantém milhares de verbos técnicos raros.
verbos_validos = {verbo for verbo, count in contador_verbos.items() if count >= 2}
print(f"\nExtraídos {len(verbos_validos)} verbos válidos do domínio.")

# =======================================================
# 3. CRUZAMENTO LÉXICO (O Novo Dicionário)
# =======================================================
print("Mapeando Antônimos via WordNet...")
dicionario_antonimos_dominio = {}

for verbo in verbos_validos:
    antonimos_encontrados = []
    
    for syn in wn.synsets(verbo, pos=wn.VERB):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                ant = lemma.antonyms()[0].name().lower()
                antonimos_encontrados.append(ant)
    
    if antonimos_encontrados:
        melhor_antonimo = Counter(antonimos_encontrados).most_common(1)[0][0]
        
        # O antônimo só precisa fazer parte dos verbos válidos do GovReport
        if melhor_antonimo in verbos_validos and melhor_antonimo != verbo:
            dicionario_antonimos_dominio[verbo] = melhor_antonimo

# =======================================================
# 4. EXPORTAÇÃO
# =======================================================
print("\n=== SUCESSO! NOVO DICIONÁRIO GERADO ===")
print(f"Encontrados {len(dicionario_antonimos_dominio)} pares de antônimos de domínio.")

with open('antonimos_govreport.json', 'w', encoding='utf-8') as f:
    json.dump(dicionario_antonimos_dominio, f, indent=4)

print("\nO arquivo 'antonimos_govreport.json' foi atualizado e está pronto para o seu gerador!")