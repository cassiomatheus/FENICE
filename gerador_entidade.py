#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerador_entidades_com_rastreio.py
Gera resumos corrompidos EXCLUSIVAMENTE com erros de Entidades Nomeadas.
Garante que os erros sejam espalhados em sentenças DIFERENTES.
Salva os tokens exatos que foram injetados para futura validação.
"""

import random
import pandas as pd
from typing import List, Tuple
import spacy
from datasets import load_dataset

# Carrega modelo de língua inglesa (GovReport em inglês)
nlp = spacy.load("en_core_web_sm")

# =============================================
# GERADORES DE VALORES FALSOS (ENTIDADES NOMEADAS)
# =============================================

GERADORES_ENTIDADES = {
    # Nomes comuns e corporativos/acadêmicos plausíveis
    "PERSON": lambda: random.choice([
        "James Harrison", "Sarah Jenkins", "Dr. William Carter", 
        "Eleanor Vance", "Marcus Reed", "Dr. Olivia Chen", 
        "Director Amanda Richardson", "David M. Sullivan"
    ]),
    
    # Agências governamentais falsas, ONGs, empresas contratadas e institutos
    "ORG": lambda: random.choice([
        "National Institute of Public Policy", "Vanguard Healthcare Solutions", 
        "Department of Regional Planning", "Apex Financial Corp", 
        "American Civic Coalition", "Keystone Defense Industries",
        "Office of Civic Integrity", "Horizon Medical Center"
    ]),
    
    # Condados, cidades e estados com nomes extremamente comuns/plausíveis (ou reais genéricos)
    "GPE": lambda: random.choice([
        "Harrison County", "City of Westford", "State of New Liberty", 
        "Springfield", "Montgomery County", "Republic of Valoria", 
        "Belleville", "Franklin County"
    ]),
    
    # Bacias, vales, reservas e acidentes geográficos plausíveis para relatórios de infraestrutura/ambiente
    "LOC": lambda: random.choice([
        "Colorado River Basin", "Appalachian Valley", "Lake Evergreen", 
        "Cascade Mountain Range", "Pine Ridge Reservation", 
        "Mississippi River Delta", "Silver Lake District"
    ]),
}

NAMED_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC"}

# =============================================
# FUNÇÕES DE EXTRAÇÃO E INJEÇÃO
# =============================================

def extrair_entidades_por_tipo(texto: str, labels_permitidos: set) -> List[Tuple[str, str]]:
    """Extrai entidades do texto que pertencem aos labels permitidos (Usado para a fonte)."""
    doc = nlp(texto)
    entidades = []
    for ent in doc.ents:
        if ent.label_ in labels_permitidos:
            entidades.append((ent.text, ent.label_))
    return entidades

def inject_entity_errors(
    resumo_original: str,
    fonte: str,
    num_errors: int = 1
) -> Tuple[str, int, List[str]]:
    """
    Substitui entidades nomeadas por valores falsos, garantindo no máximo 1 erro por sentença.
    RETORNO: (resumo_corrompido, quantidade_de_erros_injetados, lista_de_tokens_injetados)
    """
    entidades_fonte = extrair_entidades_por_tipo(fonte, NAMED_ENTITY_LABELS)
    
    # Conjunto de valores da fonte (para evitar criar uma "mentira" que seja verdade no texto base)
    valores_fonte_por_tipo = {}
    for texto, label in entidades_fonte:
        valores_fonte_por_tipo.setdefault(label, set()).add(texto)

    # Processa o resumo para separar por sentenças
    doc_resumo = nlp(resumo_original)
    
    # Lista de sentenças que possuem pelo menos uma entidade válida
    sentencas_com_entidades = []
    for sent in doc_resumo.sents:
        ents_validas = [ent for ent in sent.ents if ent.label_ in NAMED_ENTITY_LABELS]
        if ents_validas:
            sentencas_com_entidades.append(ents_validas) # Salva a lista de entidades desta sentença

    # Se não houver entidades no resumo para corromper, retorna intacto
    if not sentencas_com_entidades:
        return resumo_original, 0, []

    # Escolhe sentenças DISTINTAS aleatoriamente
    qtd_alvos = min(len(sentencas_com_entidades), num_errors)
    sentencas_alvo = random.sample(sentencas_com_entidades, qtd_alvos)

    erros = 0
    tokens_injetados = []
    alvos_para_corromper = [] # Lista de tuplas (Entidade do spaCy, Novo Valor)

    for ents_na_sentenca in sentencas_alvo:
        # Escolhe apenas 1 entidade aleatória dentro desta sentença específica
        ent_alvo = random.choice(ents_na_sentenca)
        
        gerador = GERADORES_ENTIDADES.get(ent_alvo.label_)
        if not gerador:
            continue

        novo_valor = None
        for _ in range(10):
            candidato = gerador()
            if not any(candidato in vals for vals in valores_fonte_por_tipo.values()):
                novo_valor = candidato
                break
        
        if novo_valor:
            alvos_para_corromper.append((ent_alvo, novo_valor))
            tokens_injetados.append(novo_valor)
            erros += 1

    # Ordena as substituições de trás para frente (reverse=True) usando o índice (start_char).
    # Isso impede que a alteração do tamanho de uma palavra quebre o índice das palavras seguintes!
    alvos_para_corromper.sort(key=lambda x: x[0].start_char, reverse=True)
    
    texto_modificado = resumo_original
    for ent_alvo, novo_valor in alvos_para_corromper:
        start = ent_alvo.start_char
        end = ent_alvo.end_char
        # Fatiamento cirúrgico da string original
        texto_modificado = texto_modificado[:start] + novo_valor + texto_modificado[end:]

    return texto_modificado, erros, tokens_injetados

# =============================================
# PIPELINE PRINCIPAL
# =============================================

def processar_dataset_entidades(dataset_split, num_erros_entidade: int = 1, seed: int = 42) -> pd.DataFrame:
    """Itera sobre o dataset e gera os resumos corrompidos com logs de rastreio."""
    random.seed(seed)
    resultados = []

    for example in dataset_split:
        fonte = example['report']
        resumo_orig = example['summary']

        res_named, cnt_named, tokens_entidade = inject_entity_errors(resumo_orig, fonte, num_erros_entidade)

        resultados.append({
            'id': example.get('id', 'N/A'),
            'fonte': fonte,
            'resumo_original': resumo_orig,
            'resumo_corrompido': res_named,
            'qtd_erros_injetados': cnt_named,
            'tokens_injetados': tokens_entidade
        })

    return pd.DataFrame(resultados)

# =============================================
# EXECUÇÃO (GERAÇÃO DO DATASET DE TESTE)
# =============================================

if __name__ == "__main__":
    # Carrega uma pequena amostra (ex: 5 registros) do GovReport
    dataset = load_dataset("ccdv/govreport-summarization", split="test")
    amostra = dataset.select(range(5))

    # Pede para injetar 2 erros, o código garantirá que eles vão para sentenças separadas
    df = processar_dataset_entidades(amostra, num_erros_entidade=6, seed=42)

    colunas_exibicao = ['id', 'qtd_erros_injetados', 'tokens_injetados']
    print("\n--- AMOSTRA DOS TOKENS INJETADOS ---")
    print(df[colunas_exibicao].to_string())
    
    # Salva o arquivo CSV que será consumido pelo FENICE no outro script
    df.to_csv("govreport_erros_entidade.csv", index=False, encoding='utf-8')
    print("\nDataset salvo como 'govreport_erros_entidade_1.csv'.")