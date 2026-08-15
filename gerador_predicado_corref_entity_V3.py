#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerador_predicado_corref_entity_V3.py
Gera resumos corrompidos com erros avançados de PREDICADO, CORREFERÊNCIA e ENTIDADE.
Utiliza Seleção Prioritária, Parse de Dependência S-V-O, Similaridade Semântica e Isolamento Lexical
para gerar anomalias factuais cognitivamente realistas para testes de LLMs.
"""

import random
import pandas as pd
from typing import List, Tuple
import spacy
from datasets import load_dataset
import json
import lemminflect
from spacy.tokens import Doc

# Carrega modelo de língua inglesa MÉDIO (Necessário para os word vectors do Entity Swap)
nlp = spacy.load("en_core_web_md")

# Carrega o dicionário de antônimos do arquivo JSON
try:
    with open('antonimos_govreport.json', 'r') as f:
        ANTONIMOS_DOMINIO = json.load(f)
except FileNotFoundError:
    print("AVISO: 'antonimos_govreport.json' não encontrado. Usando fallback básico.")
    ANTONIMOS_DOMINIO = {}

def apply_token_replacement_char(sent_text, target_token, replacement_string):
    """
    Substitui o token preservando 100% da formatação e pontuação original da sentença.
    Usa o índice de caracteres (char offset) relativo à sentença.
    """
    start_char = target_token.idx - target_token.sent.start_char
    end_char = start_char + len(target_token.text)
    
    return sent_text[:start_char] + replacement_string + sent_text[end_char:]

def estrategizar_predicado(verbo_alvo, sent, antonimos):
    lemma = verbo_alvo.lemma_.lower()

    # -----------------------------
    # 1. ANTÔNIMO
    # -----------------------------
    if lemma in antonimos:
        antonym_lemma = antonimos[lemma]
        infl = lemminflect.getInflection(antonym_lemma, tag=verbo_alvo.tag_)
        novo_verbo = infl[0] if infl else antonym_lemma
        
        return apply_token_replacement_char(sent.text, verbo_alvo, novo_verbo), novo_verbo

    # -----------------------------
    # 2. NEGAÇÃO CORRETA (Auxiliary Support)
    # -----------------------------
    if verbo_alvo.tag_ in ("VBD", "VBP", "VBZ"):
        if verbo_alvo.tag_ == "VBD":
            replacement = f"did not {lemma}"
            return apply_token_replacement_char(sent.text, verbo_alvo, replacement), "did not"

        elif verbo_alvo.tag_ == "VBZ":
            replacement = f"does not {lemma}"
            return apply_token_replacement_char(sent.text, verbo_alvo, replacement), "does not"

        else:
            replacement = f"do not {lemma}"
            return apply_token_replacement_char(sent.text, verbo_alvo, replacement), "do not"

    # -----------------------------
    # 3. SHIFT DE TEMPO
    # -----------------------------
    replacement = f"will {lemma}"
    return apply_token_replacement_char(sent.text, verbo_alvo, replacement), "tense_shift"


def inject_realistic_predicate_errors(
    resumo_orig: str,
    fonte_orig: str,
    num_erros: int = 2
):
    doc = nlp(resumo_orig)
    sents = list(doc.sents)

    candidatos_vip = []
    candidatos_normais = []

    for i, sent in enumerate(sents):
        # NOVO FILTRO S-V-O: O verbo precisa ter um Sujeito (nsubj) e um Objeto/Complemento
        verbos_svo = []
        for t in sent:
            if t.pos_ == "VERB" and t.is_alpha:
                # Checa dependentes do verbo
                dependentes = [child.dep_ for child in t.children]
                tem_sujeito = any(dep in ("nsubj", "nsubjpass", "csubj") for dep in dependentes)
                tem_complemento = any(dep in ("dobj", "pobj", "acomp", "xcomp") for dep in dependentes)
                
                if tem_sujeito and tem_complemento:
                    verbos_svo.append(t)
                    
        if verbos_svo:
            if any(v.lemma_.lower() in ANTONIMOS_DOMINIO for v in verbos_svo):
                candidatos_vip.append((i, sent, verbos_svo))
            else:
                candidatos_normais.append((i, sent, verbos_svo))

    if not candidatos_vip and not candidatos_normais:
        return resumo_orig, 0, []

    escolhidos = []
    
    if candidatos_vip:
        qtd_vip = min(len(candidatos_vip), num_erros)
        escolhidos.extend(random.sample(candidatos_vip, qtd_vip))
        
    faltam = num_erros - len(escolhidos)
    if faltam > 0 and candidatos_normais:
        qtd_normais = min(len(candidatos_normais), faltam)
        escolhidos.extend(random.sample(candidatos_normais, qtd_normais))

    resultado = []
    logs = []
    usados = set()

    for i, sent in enumerate(sents):
        match = next((c for c in escolhidos if c[0] == i), None)

        if match:
            _, sent_ref, verbos = match
            verbos_validos = [v for v in verbos if v.lemma_.lower() not in usados]

            if not verbos_validos:
                resultado.append(sent.text)
                continue

            verbos_no_dict = [v for v in verbos_validos if v.lemma_.lower() in ANTONIMOS_DOMINIO]
            verbo_alvo = random.choice(verbos_no_dict) if verbos_no_dict else random.choice(verbos_validos)
            
            usados.add(verbo_alvo.lemma_.lower())

            novo_texto, token = estrategizar_predicado(verbo_alvo, sent, ANTONIMOS_DOMINIO)
            resultado.append(novo_texto)

            logs.append({
                "sentenca_index": i,
                "verbo_original": verbo_alvo.text,
                "tipo_erro": "predicate",
                "injetado": token
            })
        else:
            resultado.append(sent.text)

    return " ".join(resultado), len(logs), logs


def inject_realistic_coreference_errors_antigo(
    resumo_orig: str,
    fonte_orig: str,
    num_erros=2
):
    doc_resumo = nlp(resumo_orig)
    doc_fonte = nlp(fonte_orig)

    # 1. Sujeitos reais da fonte
    sujeitos_fonte = set(
        t.lemma_.lower().strip()
        for t in doc_fonte
        if t.dep_ in ("nsubj", "nsubjpass")
    )

    # 2. Mapeia candidatos NOMINAIS da fonte
    candidatos = []
    for chunk in doc_fonte.noun_chunks:
        if 1 < len(chunk.text.split()) < 6:
            lemma = chunk.root.lemma_.lower().strip()
            if lemma not in sujeitos_fonte:
                candidatos.append({
                    "text": chunk.text,
                    "lemma": lemma
                })

    if not candidatos:
        return resumo_orig, 0, []

    # 3. Detecta pronomes sujeitos no resumo
    sentencas = list(doc_resumo.sents)
    alvos = []

    for i, sent in enumerate(sentencas):
        for t in sent:
            if (
                t.pos_ == "PRON"
                and t.dep_ in ("nsubj", "nsubjpass")
                and t.tag_ in ("PRP", "PRP$")
                and t.lemma_.lower() in {"he", "she", "they", "it"}
            ):
                alvos.append((i, sent, t))

    if len(alvos) == 0:
        return resumo_orig, 0, []

    if num_erros > len(alvos):
        num_erros = len(alvos)

    escolhidos = random.sample(alvos, num_erros)
    escolhidos.sort(key=lambda x: x[2].idx, reverse=True)

    resumo_texto_mutavel = resumo_orig
    erros = []
    usados = set()

    # 4. Injeção controlada e precisa
    for sent_idx, sent_ref, pron in escolhidos:
        antecedente_real = None

        # --- NOVA ETAPA: ISOLAMENTO DO ANTECEDENTE REAL ---
        # Mapeia todas as raízes nominais do contexto imediato (frase atual e a anterior)
        context_lemmas = {t.lemma_.lower() for t in sent_ref}
        if sent_idx > 0:
            context_lemmas.update({t.lemma_.lower() for t in sentencas[sent_idx - 1]})
            
        # Filtra candidatos garantindo ausência de vizinhança semântica
        candidatos_validos = []
        for c in candidatos:
            if c["lemma"] not in usados:
                if c["lemma"] not in context_lemmas:
                    candidatos_validos.append(c)

        if not candidatos_validos:
            continue

        novo = random.choice(candidatos_validos)
        usados.add(novo["lemma"])

        replacement = novo["text"]
        if pron.is_sent_start:
            replacement = replacement[0].upper() + replacement[1:]

        start_char = pron.idx
        end_char = pron.idx + len(pron.text)
        
        resumo_texto_mutavel = (
            resumo_texto_mutavel[:start_char] + 
            replacement + 
            resumo_texto_mutavel[end_char:]
        )

        erros.append({
            "sentenca_index": sent_idx,
            "pronome": pron.text,
            #"antecedente_original": getattr(antecedente_real, "text", None),
            "novo_sujeito": replacement,
            "tipo_erro": "coreference"  # Alinhado com a nova taxonomia estrita e padronizado com as outras funções
        })

    erros.reverse()
    return resumo_texto_mutavel, len(erros), erros



def inject_realistic_coreference_errors(
    resumo_orig: str,
    fonte_orig: str,
    num_erros=2
):
    doc_resumo = nlp(resumo_orig)
    doc_fonte = nlp(fonte_orig)

    # 1. Sujeitos reais da fonte
    sujeitos_fonte = set(
        t.lemma_.lower().strip()
        for t in doc_fonte
        if t.dep_ in ("nsubj", "nsubjpass")
    )

    # 2. Mapeia candidatos NOMINAIS da fonte
    candidatos = []
    for chunk in doc_fonte.noun_chunks:
        if 1 < len(chunk.text.split()) < 6:
            lemma = chunk.root.lemma_.lower().strip()
            if lemma not in sujeitos_fonte:
                candidatos.append({
                    "text": chunk.text,
                    "lemma": lemma
                })

    if not candidatos:
        return resumo_orig, 0, []

    # -------------------------------------------------------------
    # 3. Detecta e AGRUPA pronomes por SENTENÇA (Garante Máx. 1 Erro/Frase)
    # -------------------------------------------------------------
    sentencas = list(doc_resumo.sents)
    pronomes_por_sentenca = {}

    for i, sent in enumerate(sentencas):
        prons_frase = []
        for t in sent:
            if (
                t.pos_ == "PRON"
                and t.dep_ in ("nsubj", "nsubjpass")
                and t.tag_ in ("PRP", "PRP$")
                and t.lemma_.lower() in {"he", "she", "they", "it"}
            ):
                # 🛡️ FILTRO ANTI-EXPLETIVO: Descarta 'it' seguido de verbos de ligação + adjetivos/orações (ex: it is clear, it is necessary)
                if t.lemma_.lower() == "it":
                    # Se o verbo for 'be' e tiver complemento oracional (ex: 'whether', 'that'), é dummy 'it'
                    head_verb = t.head
                    has_clausal_complement = any(child.dep_ in ("ccomp", "xcomp", "advcl") for child in head_verb.children)
                    if has_clausal_complement:
                        continue # Pula este 'it' pois é expletivo fictício!
                        
                prons_frase.append(t)
        
        if prons_frase:
            pronomes_por_sentenca[i] = (sent, prons_frase)

    if not pronomes_por_sentenca:
        return resumo_orig, 0, []

    # Sorteia FRASES ÚNICAS (impede repetição de sentença)
    sent_indices_elegiveis = list(pronomes_por_sentenca.keys())
    qtd_sentencas = min(num_erros, len(sent_indices_elegiveis))
    sent_indices_escolhidos = random.sample(sent_indices_elegiveis, qtd_sentencas)

    # Para cada frase escolhida, sorteia APENAS 1 pronome alvo
    escolhidos = []
    for idx in sent_indices_escolhidos:
        sent_ref, prons = pronomes_por_sentenca[idx]
        pron_alvo = random.choice(prons)
        escolhidos.append((idx, sent_ref, pron_alvo))

    # Ordena reverso por offset de caractere para manter os índices intactos na substituição
    escolhidos.sort(key=lambda x: x[2].idx, reverse=True)

    resumo_texto_mutavel = resumo_orig
    erros = []
    usados = set()

    # 4. Injeção controlada e precisa
    for sent_idx, sent_ref, pron in escolhidos:
        context_lemmas = {t.lemma_.lower() for t in sent_ref}
        if sent_idx > 0:
            context_lemmas.update({t.lemma_.lower() for t in sentencas[sent_idx - 1]})
            
        candidatos_validos = [
            c for c in candidatos 
            if c["lemma"] not in usados and c["lemma"] not in context_lemmas
        ]

        if not candidatos_validos:
            continue

        novo = random.choice(candidatos_validos)
        usados.add(novo["lemma"])

        replacement = novo["text"]
        if pron.is_sent_start:
            replacement = replacement[0].upper() + replacement[1:]

        start_char = pron.idx
        end_char = pron.idx + len(pron.text)
        
        resumo_texto_mutavel = (
            resumo_texto_mutavel[:start_char] + 
            replacement + 
            resumo_texto_mutavel[end_char:]
        )

        erros.append({
            "sentenca_index": sent_idx,
            "pronome": pron.text,
            "novo_sujeito": replacement,
            "tipo_erro": "coreference"
        })

    erros.reverse()
    return resumo_texto_mutavel, len(erros), erros


def normalize_ent(ent):
    return " ".join(ent.lower().strip().split())


def inject_realistic_entity_errors(resumo_orig, fonte_orig, num_erros=2):
    doc_resumo = nlp(resumo_orig)
    doc_fonte = nlp(fonte_orig)

    # -----------------------------
    # 1. Extração de Entidades da Fonte
    # -----------------------------
    entidades_fonte = {}

    for ent in doc_fonte.ents:
        label = ent.label_
        entidades_fonte.setdefault(label, [])
        entidades_fonte[label].append(ent.text)

    # -----------------------------
    # 2. Seleção de Sentenças Candidatas no Resumo
    # -----------------------------
    sentencas = list(doc_resumo.sents)
    sent_validas = [
        i for i, s in enumerate(sentencas)
        if len(s.ents) > 0
    ]

    if not sent_validas:
        return resumo_orig, 0, []

    escolhidas = random.sample(sent_validas, min(num_erros, len(sent_validas)))

    resultado = []
    logs = []

    # -----------------------------
    # 3. Processamento e Injeção por Fatiamento
    # -----------------------------
    for i, sent in enumerate(sentencas):

        if i in escolhidas:
            ents = list(sent.ents)
            alvo = random.choice(ents)
            tipo = alvo.label_

            if tipo not in entidades_fonte:
                resultado.append(sent.text)
                continue

            candidatos = entidades_fonte[tipo]
            norm_alvo = normalize_ent(alvo.text)
            alvo_doc = nlp(alvo.text) # Processa o alvo para pegar os vetores

            # Calcula o "Confusion Score" (Similaridade) para cada candidato
            candidatos_com_score = []
            for c in candidatos:
                if normalize_ent(c) != norm_alvo:
                    c_doc = nlp(c)
                    # Verifica se ambos têm vetores válidos para evitar avisos
                    if alvo_doc.has_vector and c_doc.has_vector:
                        sim_score = alvo_doc.similarity(c_doc)
                    else:
                        sim_score = 0.5 # fallback se não houver vetor
                    candidatos_com_score.append((c, sim_score))

            if not candidatos_com_score:
                resultado.append(sent.text)
                continue

            # ORDENA PELO MAIOR CONFUSION SCORE (Os mais parecidos semanticamente)
            candidatos_com_score.sort(key=lambda x: x[1], reverse=True)
            
            # Pega o Top 3 mais confusos e sorteia entre eles para manter variabilidade
            top_confusos = [x[0] for x in candidatos_com_score[:3]]
            nova_entidade = random.choice(top_confusos)

            # --- CORREÇÃO CRÍTICA: Substituição Cirúrgica por Offset ---
            start_char = alvo.start_char - sent.start_char
            end_char = alvo.end_char - sent.start_char
            novo_texto_sentenca = sent.text[:start_char] + nova_entidade + sent.text[end_char:]

            resultado.append(novo_texto_sentenca)

            # --- PADRONIZAÇÃO: Chaves alinhadas com o metavaliacao2.py ---
            logs.append({
                "sentenca_index": i,
                "tipo_erro": "entity",
                "original": alvo.text,
                "token": nova_entidade
            })

        else:
            resultado.append(sent.text)

    return " ".join(resultado), len(logs), logs


def processar_dataset_erros(dataset_split, num_erros: int = 2, seed: int = 42):
    random.seed(seed)
    resultados = []

    for example in dataset_split:
        fonte = example['report']
        resumo_orig = example['summary']

        res_pred, cnt_pred, toks_pred = inject_realistic_predicate_errors(resumo_orig, fonte, num_erros)
        res_corref, cnt_corref, toks_corref = inject_realistic_coreference_errors(resumo_orig, fonte, num_erros)
        res_entity, cnt_entity, toks_entity = inject_realistic_entity_errors(resumo_orig, fonte, num_erros)

        resultados.append({
            'id': example.get('id', 'N/A'),
            'fonte': fonte,
            'resumo_original': resumo_orig,

            #Predicado
            'resumo_predicado': res_pred,
            'qtd_erros_predicado': cnt_pred,
            'tokens_predicado': toks_pred,

            #Correferencia
            'resumo_corref': res_corref,
            'qtd_erros_corref': cnt_corref,
            'tokens_corref': toks_corref,
            
            #Entidade
            'resumo_entity': res_entity,
            'qtd_erros_entity': cnt_entity,
            'tokens_entity': toks_entity
        })

    return pd.DataFrame(resultados)


if __name__ == "__main__":
    print("Iniciando pipeline de geração de erros (V3 - Alta Fidelidade)...")
    dataset = load_dataset("ccdv/govreport-summarization", split="test")
    amostra = dataset.select(range(10))

    df = processar_dataset_erros(amostra, num_erros=6, seed=42)
    
    arquivo_saida = 'Gerador_erros/govreport_erros_pred_corref_entity_realistas_v1.csv'

    df.to_csv(arquivo_saida, index=False)
    print(f"\n✅ Pipeline concluído! Dataset exportado para: {arquivo_saida}")