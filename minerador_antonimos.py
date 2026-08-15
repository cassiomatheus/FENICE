import spacy
import nltk
from nltk.corpus import wordnet as wn
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import json

# =======================================================
# 1. PREPARAÇÃO (Baixa as bibliotecas necessárias)
# =======================================================
nltk.download('wordnet')
nltk.download('omw-1.4')

# Carrega o spaCy desabilitando NER e Parser para ser 10x mais rápido!
print("Carregando spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# --- CORREÇÃO: Aumenta o limite de caracteres para documentos longos ---
nlp.max_length = 2000000

# =======================================================
# 2. MINERAÇÃO DE VERBOS (Lendo 500 documentos reais)
# =======================================================
print("Baixando/Carregando o GovReport...")
# Puxa 400 documentos da base de treino
dataset = load_dataset("ccdv/govreport-summarization", split="test[:970]")

contador_verbos = Counter()

print(f"Minerando verbos de {len(dataset)} documentos...")

for example in tqdm(dataset, desc="Analisando textos"):
    try:
        doc = nlp(example['report'])
        for token in doc:
            # Pega apenas verbos puros (lematizados) em minúsculo
            if token.pos_ == "VERB" and token.is_alpha:
                contador_verbos[token.lemma_.lower()] += 1
    except ValueError:
        # Se o documento for maior que 2 milhões, ele pula e continua
        continue

# Pega os 500 verbos mais comuns da burocracia governamental
top_verbos = {verbo for verbo, count in contador_verbos.most_common(9000)}
print(f"\nExtraídos os {len(top_verbos)} verbos mais comuns.")

# =======================================================
# 3. CRUZAMENTO LÉXICO (Encontrando os Antônimos de Domínio)
# =======================================================
print("Mapeando Antônimos via WordNet...")
dicionario_antonimos_dominio = {}

for verbo in top_verbos:
    antonimos_encontrados = []
    
    # Procura na base do WordNet por sinsets (grupos de significado) do tipo VERBO
    for syn in wn.synsets(verbo, pos=wn.VERB):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                # Pega o nome do antônimo
                ant = lemma.antonyms()[0].name().lower()
                antonimos_encontrados.append(ant)
    
    if antonimos_encontrados:
        # Pega o antônimo mais comum sugerido pelo WordNet para este verbo
        melhor_antonimo = Counter(antonimos_encontrados).most_common(1)[0][0]
        
        # O FILTRO DE REALISMO: O antônimo TAMBÉM tem que ser uma palavra comum no GovReport!
        if melhor_antonimo in top_verbos and melhor_antonimo != verbo:
            dicionario_antonimos_dominio[verbo] = melhor_antonimo

# =======================================================
# 4. EXPORTAÇÃO
# =======================================================
print("\n=== SUCESSO! DICIONÁRIO GERADO ===")
print(f"Encontrados {len(dicionario_antonimos_dominio)} pares perfeitos de antônimos de domínio.")

# Salva o resultado em um arquivo JSON e imprime no console
with open('antonimos_govreport.json', 'w', encoding='utf-8') as f:
    json.dump(dicionario_antonimos_dominio, f, indent=4)

print("\nSeu novo dicionário para colar no script de injeção:")
print("ANTONIMOS_DOMINIO = {")
for v, ant in sorted(dicionario_antonimos_dominio.items()):
    print(f'    "{v}": "{ant}",')
print("}")