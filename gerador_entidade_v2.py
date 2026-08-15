#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerador_entidade.py
Gera resumos corrompidos EXCLUSIVAMENTE com erros de Entidades Nomeadas (Entity Swapping).
Garante que os erros sejam espalhados em sentenças DIFERENTES e sejam baseados em entidades
reais do documento fonte, aumentando o realismo da alucinação.
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
# INJEÇÃO DE ERRO REALISTA (ENTITY SWAPPING)
# =============================================

def inject_realistic_entity_errors(resumo_orig: str, fonte_orig: str, num_erros: int = 2) -> Tuple[str, int, List[str]]:
    doc_resumo = nlp(resumo_orig)
    doc_fonte = nlp(fonte_orig)
    
    # Extrai todas as entidades REAIS do texto fonte e agrupa por tipo (ORG, PERSON, GPE, DATE)
    entidades_fonte = {}
    for ent in doc_fonte.ents:
        if ent.label_ not in entidades_fonte:
            entidades_fonte[ent.label_] = []
        entidades_fonte[ent.label_].append(ent.text)
        
    sents_resumo = list(doc_resumo.sents)
    indices_sents_com_entidades = [
        i for i, s in enumerate(sents_resumo) if len(s.ents) > 0
    ]
    
    if len(indices_sents_com_entidades) < num_erros:
        num_erros = len(indices_sents_com_entidades)
        
    sents_escolhidas = random.sample(indices_sents_com_entidades, num_erros)
    
    resumo_modificado = []
    tokens_injetados = []
    erros_injetados = 0
    
    for i, sent in enumerate(sents_resumo):
        if i in sents_escolhidas:
            entidades_na_sentenca = list(sent.ents)
            entidade_alvo = random.choice(entidades_na_sentenca)
            tipo_alvo = entidade_alvo.label_
            
            if tipo_alvo in entidades_fonte and len(entidades_fonte[tipo_alvo]) > 1:
                opcoes_validas = [e for e in entidades_fonte[tipo_alvo] if e.lower() not in entidade_alvo.text.lower()]
                
                if opcoes_validas:
                    nova_entidade = random.choice(opcoes_validas)
                    texto_sent = sent.text.replace(entidade_alvo.text, nova_entidade, 1)
                    resumo_modificado.append(texto_sent)
                    
                    # --- MUDANÇA AQUI: Salva o índice da sentença e a entidade ---
                    tokens_injetados.append({
                        "sentenca_index": i, 
                        "tipo_erro": "entity",
                        "token": nova_entidade
                    })
                    erros_injetados += 1
                    continue
                    
        resumo_modificado.append(sent.text)
        
    return " ".join(resumo_modificado), erros_injetados, tokens_injetados

# =============================================
# PROCESSAMENTO DO DATASET
# =============================================

def processar_dataset_entidades(dataset_split, num_erros_entidade: int = 2, seed: int = 42):
    random.seed(seed)
    resultados = []

    for example in dataset_split:
        fonte = example['report']
        resumo_orig = example['summary']

        # Chama a nova função realista
        res_named, cnt_named, tokens_entidade = inject_realistic_entity_errors(resumo_orig, fonte, num_erros_entidade)

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

    # Pede para injetar 6 erros (ou o máximo que a sentença permitir)
    df = processar_dataset_entidades(amostra, num_erros_entidade=6, seed=42)

    colunas_exibicao = ['id', 'qtd_erros_injetados', 'tokens_injetados']
    print(df[colunas_exibicao].head())
    
    # Salva o dataset com apenas erros de entidades realistas
    df.to_csv('govreport_erros_entidades_realistas.csv', index=False)
    print("\nDataset gerado e salvo como 'govreport_erros_entidades_realistas.csv'")