#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerador_predicado_corref.py
Gera resumos corrompidos com erros de PREDICADO (Verbos) e CORREFERÊNCIA (Pronomes).
Garante que os erros sejam espalhados em sentenças DIFERENTES.
Salva os tokens exatos que foram injetados para a ancoragem do FENICE no metavaliacao2.py.
"""

import random
import pandas as pd
from typing import List, Tuple
import spacy
from datasets import load_dataset

# Carrega modelo de língua inglesa
nlp = spacy.load("en_core_web_sm")

# =============================================
# DICIONÁRIOS E LISTAS DE VALORES FALSOS
# =============================================

# Verbos burocráticos/técnicos plausíveis para relatórios do GovReport
VERBOS_EXTERNOS = [
    "calibrate", "simulate", "quantify", "dismantle", "aggregate",
    "parameterize", "optimize", "catalyze", "fabricate", "modulate",
    "attenuate", "propagate", "oscillate", "radiate", "converge",
    "diverge", "transduce", "encrypt", "decouple", "replicate"
]

# Sintagmas nominais (Noun Phrases) plausíveis para substituir pronomes
SN_EXTERNOS = [
    "the fiscal projection model",
    "the compliance audit framework",
    "the external oversight committee",
    "the regional data validation system",
    "the independent personnel log",
    "the strategic risk assessment matrix",
    "the autonomous administrative council"
]

# =============================================
# FUNÇÕES LINGUÍSTICAS AUXILIARES
# =============================================

def conjugar_verbo(token_spacy, novo_lemma: str) -> str:
    """Conjuga 'novo_lemma' no mesmo tempo/pessoa/número do verbo original."""
    tag = token_spacy.tag_
    irreg_past = {"be": "was", "have": "had", "do": "did", "say": "said",
                  "go": "went", "make": "made", "know": "knew", "see": "saw",
                  "come": "came", "find": "found", "give": "gave", "tell": "told"}
    
    if tag == "VB":      # Base form
        return novo_lemma
    elif tag == "VBD":   # Past simple
        if novo_lemma in irreg_past:
            return irreg_past[novo_lemma]
        return novo_lemma + "ed" if not novo_lemma.endswith('e') else novo_lemma + "d"
    elif tag == "VBG":   # Gerund / present participle
        if novo_lemma.endswith('e'):
            return novo_lemma[:-1] + "ing"
        return novo_lemma + "ing"
    elif tag == "VBN":   # Past participle
        if novo_lemma in irreg_past:
            return irreg_past[novo_lemma]
        return novo_lemma + "ed" if not novo_lemma.endswith('e') else novo_lemma + "d"
    elif tag == "VBZ":   # 3rd person singular present
        if novo_lemma.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return novo_lemma + "es"
        elif novo_lemma.endswith('y') and novo_lemma[-2] not in 'aeiou':
            return novo_lemma[:-1] + "ies"
        return novo_lemma + "s"
    elif tag == "VBP":   # Non-3rd person present
        return novo_lemma
    else:
        return novo_lemma

# =============================================
# INJETOR DE PREDICADO (VERBOS)
# =============================================

def inject_predicate_errors(
    resumo_original: str, fonte: str, num_errors: int = 1
) -> Tuple[str, int, List[str]]:
    
    # 1. Mapear verbos que existem na fonte para não usá-los acidentalmente
    doc_fonte = nlp(fonte)
    verbos_fonte = {t.lemma_.lower() for t in doc_fonte if t.pos_ == "VERB"}

    # 2. Separar as sentenças do resumo que contêm verbos válidos
    doc_resumo = nlp(resumo_original)
    sentencas_com_verbos = []
    
    for sent in doc_resumo.sents:
        verbos_validos = [t for t in sent if t.pos_ == "VERB" and t.is_alpha]
        if verbos_validos:
            sentencas_com_verbos.append(verbos_validos)

    if not sentencas_com_verbos:
        return resumo_original, 0, []

    # 3. Sortear as sentenças distintas
    qtd_alvos = min(len(sentencas_com_verbos), num_errors)
    sentencas_alvo = random.sample(sentencas_com_verbos, qtd_alvos)

    erros = 0
    tokens_injetados = []
    alvos_para_corromper = []

    # 4. Processar substituições
    for verbos_na_sentenca in sentencas_alvo:
        token_alvo = random.choice(verbos_na_sentenca)
        
        # Filtra os verbos externos garantindo que não estão na fonte
        candidatos = [v for v in VERBOS_EXTERNOS if v not in verbos_fonte]
        if not candidatos:
            continue
            
        novo_lemma = random.choice(candidatos)
        novo_valor = conjugar_verbo(token_alvo, novo_lemma)

        # Preserva capitalização se necessário
        if token_alvo.text.istitle() or token_alvo.is_sent_start:
            novo_valor = novo_valor.capitalize()

        alvos_para_corromper.append((token_alvo, novo_valor))
        tokens_injetados.append(novo_valor)
        erros += 1

    # 5. Substituição cirúrgica de trás para frente usando o índice de caracteres originais (idx)
    alvos_para_corromper.sort(key=lambda x: x[0].idx, reverse=True)
    texto_modificado = resumo_original
    
    for token_alvo, novo_valor in alvos_para_corromper:
        start = token_alvo.idx
        end = start + len(token_alvo.text)
        texto_modificado = texto_modificado[:start] + novo_valor + texto_modificado[end:]

    return texto_modificado, erros, tokens_injetados

# =============================================
# INJETOR DE CORREFERÊNCIA (PRONOMES)
# =============================================

def inject_coreference_errors(
    resumo_original: str, fonte: str, num_errors: int = 1
) -> Tuple[str, int, List[str]]:
    
    # 1. Extrair sujeitos da fonte para evitar injetar um sujeito que já é verdade lá
    doc_fonte = nlp(fonte)
    sujeitos_fonte = {t.text.lower() for t in doc_fonte if t.dep_ in ("nsubj", "nsubjpass")}

    # 2. Separar as sentenças do resumo que contêm pronomes
    doc_resumo = nlp(resumo_original)
    sentencas_com_pronomes = []
    
    for sent in doc_resumo.sents:
        # PRP = Pronome Pessoal (it, they), PRP$ = Possessivo (its, their)
        pronomes_validos = [t for t in sent if t.pos_ == "PRON" and t.tag_ in ("PRP", "PRP$")]
        if pronomes_validos:
            sentencas_com_pronomes.append(pronomes_validos)

    if not sentencas_com_pronomes:
        return resumo_original, 0, []

    # 3. Sortear as sentenças distintas
    qtd_alvos = min(len(sentencas_com_pronomes), num_errors)
    sentencas_alvo = random.sample(sentencas_com_pronomes, qtd_alvos)

    erros = 0
    tokens_injetados = []
    alvos_para_corromper = []

    # 4. Processar substituições
    for pronomes_na_sentenca in sentencas_alvo:
        token_alvo = random.choice(pronomes_na_sentenca)
        
        candidatos = [sn for sn in SN_EXTERNOS if sn.lower() not in sujeitos_fonte]
        if not candidatos:
            continue
            
        novo_sn = random.choice(candidatos)

        # Trata o caso de pronome possessivo (adicionando 's)
        if token_alvo.tag_ == "PRP$":
            novo_valor = novo_sn + "'s" if not novo_sn.endswith('s') else novo_sn + "'"
        else:
            novo_valor = novo_sn

        # Preserva capitalização (ex: se o pronome começar a frase)
        if token_alvo.text.istitle() or token_alvo.is_sent_start:
            novo_valor = novo_valor[0].upper() + novo_valor[1:]

        alvos_para_corromper.append((token_alvo, novo_valor))
        tokens_injetados.append(novo_valor)
        erros += 1

    # 5. Substituição cirúrgica de trás para frente
    alvos_para_corromper.sort(key=lambda x: x[0].idx, reverse=True)
    texto_modificado = resumo_original
    
    for token_alvo, novo_valor in alvos_para_corromper:
        start = token_alvo.idx
        end = start + len(token_alvo.text)
        texto_modificado = texto_modificado[:start] + novo_valor + texto_modificado[end:]

    return texto_modificado, erros, tokens_injetados

# =============================================
# PIPELINE PRINCIPAL E SALVAMENTO
# =============================================

def processar_dataset_erros(dataset_split, num_erros: int = 1, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    resultados = []

    for example in dataset_split:
        fonte = example['report']
        resumo_orig = example['summary']

        # Gera o resumo focado em Predicado
        res_pred, cnt_pred, toks_pred = inject_predicate_errors(resumo_orig, fonte, num_erros)
        
        # Gera o resumo focado em Correferência
        res_corref, cnt_corref, toks_corref = inject_coreference_errors(resumo_orig, fonte, num_erros)

        resultados.append({
            'id': example.get('id', 'N/A'),
            'fonte': fonte,
            'resumo_original': resumo_orig,
            'resumo_predicado': res_pred,
            'qtd_erros_predicado': cnt_pred,
            'tokens_predicado': toks_pred,
            'resumo_corref': res_corref,
            'qtd_erros_corref': cnt_corref,
            'tokens_corref': toks_corref
        })

    return pd.DataFrame(resultados)

# =============================================
# EXECUÇÃO
# =============================================

if __name__ == "__main__":
    # Carrega uma pequena amostra do GovReport para teste
    dataset = load_dataset("ccdv/govreport-summarization", split="test")
    amostra = dataset.select(range(5))

    # Pede para injetar 2 erros de cada tipo (espalhados em sentenças diferentes)
    df = processar_dataset_erros(amostra, num_erros=6, seed=42)

    colunas_exibicao = ['id', 'tokens_predicado', 'tokens_corref']
    print("\n--- AMOSTRA DOS TOKENS INJETADOS ---")
    print(df[colunas_exibicao].to_string())
    
    df.to_csv("govreport_erros_pred_corref.csv", index=False, encoding='utf-8')
    print("\nDataset salvo como 'govreport_erros_pred_corref.csv'")