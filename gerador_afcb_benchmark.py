#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import json
import spacy
import lemminflect
from datasets import load_dataset

# Carrega modelo MÉDIO (Necessário para os word vectors do Entity Swap)
nlp = spacy.load("en_core_web_md")

# Carrega o dicionário de antônimos do arquivo JSON
try:
    with open('antonimos_govreport.json', 'r') as f:
        ANTONIMOS_DOMINIO = json.load(f)
except FileNotFoundError:
    print("AVISO: 'antonimos_govreport.json' não encontrado. Usando fallback básico.")
    ANTONIMOS_DOMINIO = {}
    

def estrategizar_predicado(verbo_alvo, antonimos):
    lemma = verbo_alvo.lemma_.lower()
    if lemma in antonimos:
        antonym_lemma = antonimos[lemma]
        infl = lemminflect.getInflection(antonym_lemma, tag=verbo_alvo.tag_)
        return (infl[0] if infl else antonym_lemma), "antonym"

    if verbo_alvo.tag_ in ("VBD", "VBP", "VBZ"):
        if verbo_alvo.tag_ == "VBD": return f"did not {lemma}", "did not"
        elif verbo_alvo.tag_ == "VBZ": return f"does not {lemma}", "does not"
        else: return f"do not {lemma}", "do not"

    return f"will {lemma}", "tense_shift"

def normalize_ent(ent):
    return " ".join(ent.lower().strip().split())

# ==========================================
# 1. INJEÇÃO DE PREDICADO (S-V-O + Offsets)
# ==========================================
def inject_afcb_predicate(resumo_orig: str, num_erros: int = 1):
    doc = nlp(resumo_orig)
    candidatos_vip, candidatos_normais = [], []

    for t in doc:
        if t.pos_ == "VERB" and t.is_alpha and t.pos_ != "AUX":
            dependentes = [child.dep_ for child in t.children]
            if any(dep in ("nsubj", "nsubjpass", "csubj") for dep in dependentes) and \
               any(dep in ("dobj", "pobj", "acomp", "xcomp") for dep in dependentes):
                if t.lemma_.lower() in ANTONIMOS_DOMINIO:
                    candidatos_vip.append(t)
                else:
                    candidatos_normais.append(t)

    escolhidos = []
    if candidatos_vip:
        escolhidos.extend(random.sample(candidatos_vip, min(len(candidatos_vip), num_erros)))
    faltam = num_erros - len(escolhidos)
    if faltam > 0 and candidatos_normais:
        escolhidos.extend(random.sample(candidatos_normais, min(len(candidatos_normais), faltam)))

    if not escolhidos:
        return resumo_orig, []

    # ORDENA DA ESQUERDA PARA A DIREITA PARA O OFFSET FUNCIONAR
    escolhidos.sort(key=lambda t: t.idx)

    corrupted_summary = resumo_orig
    corruptions = []
    offset_shift = 0 

    for verbo in escolhidos:
        replacement, rule = estrategizar_predicado(verbo, ANTONIMOS_DOMINIO)
        
        start_char = verbo.idx + offset_shift
        end_char = start_char + len(verbo.text)
        
        corrupted_summary = corrupted_summary[:start_char] + replacement + corrupted_summary[end_char:]
        
        corruptions.append({
            "type": "predicate",
            "rule_applied": rule,
            "original": verbo.text,
            "replacement": replacement,
            "span_start": start_char,
            "span_end": start_char + len(replacement)
        })
        offset_shift += len(replacement) - len(verbo.text)

    return corrupted_summary, corruptions

# ==========================================
# 2. INJEÇÃO DE CORREFERÊNCIA (Noun Chunks + Offsets)
# ==========================================
def inject_afcb_coreference(resumo_orig: str, fonte_orig: str, num_erros: int = 1):
    doc_resumo = nlp(resumo_orig)
    doc_fonte = nlp(fonte_orig)

    sujeitos_fonte = {t.lemma_.lower().strip() for t in doc_fonte if t.dep_ in ("nsubj", "nsubjpass")}
    candidatos_falsos = [chunk.text for chunk in doc_fonte.noun_chunks 
                         if 1 < len(chunk.text.split()) < 6 and chunk.root.lemma_.lower().strip() not in sujeitos_fonte]

    pronomes = [t for t in doc_resumo if t.pos_ == "PRON" and t.dep_ in ("nsubj", "nsubjpass") 
                and t.lemma_.lower() in {"he", "she", "they", "it"}]
    
    if not pronomes or not candidatos_falsos:
        return resumo_orig, []

    escolhidos = random.sample(pronomes, min(num_erros, len(pronomes)))
    escolhidos.sort(key=lambda t: t.idx)

    corrupted_summary = resumo_orig
    corruptions = []
    offset_shift = 0

    for pron in escolhidos:
        replacement = random.choice(candidatos_falsos)
        if pron.is_sent_start:
            replacement = replacement[0].upper() + replacement[1:]
        
        start_char = pron.idx + offset_shift
        end_char = start_char + len(pron.text)
        
        corrupted_summary = corrupted_summary[:start_char] + replacement + corrupted_summary[end_char:]
        
        corruptions.append({
            "type": "coreference",
            "original": pron.text,
            "replacement": replacement,
            "span_start": start_char,
            "span_end": start_char + len(replacement)
        })
        offset_shift += len(replacement) - len(pron.text)

    return corrupted_summary, corruptions

# ==========================================
# 3. INJEÇÃO DE ENTIDADE (Word Vectors + Offsets)
# ==========================================
def inject_afcb_entity(resumo_orig: str, fonte_orig: str, num_erros: int = 1):
    doc_resumo = nlp(resumo_orig)
    doc_fonte = nlp(fonte_orig)

    entidades_fonte = {}
    for ent in doc_fonte.ents:
        entidades_fonte.setdefault(ent.label_, []).append(ent.text)

    entidades_resumo = [ent for ent in doc_resumo.ents if ent.label_ in entidades_fonte]
    
    if not entidades_resumo:
        return resumo_orig, []

    escolhidos = random.sample(entidades_resumo, min(num_erros, len(entidades_resumo)))
    escolhidos.sort(key=lambda e: e.start_char)

    corrupted_summary = resumo_orig
    corruptions = []
    offset_shift = 0

    for alvo in escolhidos:
        candidatos = entidades_fonte[alvo.label_]
        norm_alvo = normalize_ent(alvo.text)
        
        candidatos_com_score = []
        for c in candidatos:
            if normalize_ent(c) != norm_alvo:
                c_doc = nlp(c)
                sim_score = alvo.similarity(c_doc) if alvo.has_vector and c_doc.has_vector else 0.5
                candidatos_com_score.append((c, sim_score))

        if not candidatos_com_score:
            continue

        candidatos_com_score.sort(key=lambda x: x[1], reverse=True)
        nova_entidade = random.choice([x[0] for x in candidatos_com_score[:3]])

        start_char = alvo.start_char + offset_shift
        end_char = alvo.end_char + offset_shift
        
        corrupted_summary = corrupted_summary[:start_char] + nova_entidade + corrupted_summary[end_char:]

        corruptions.append({
            "type": "entity",
            "entity_label": alvo.label_,
            "original": alvo.text,
            "replacement": nova_entidade,
            "span_start": start_char,
            "span_end": start_char + len(nova_entidade)
        })
        offset_shift += len(nova_entidade) - len(alvo.text)

    return corrupted_summary, corruptions

# ==========================================
# GERADOR DE BENCHMARK
# ==========================================
def gerar_benchmark_afcb(dataset_split, seed: int = 42):
    random.seed(seed)
    afcb_records = []

    for example in dataset_split:
        doc_id = example.get('id', str(random.randint(1000, 9999)))
        fonte = example['report']
        resumo_orig = example['summary']

        # 1. PREDICADO
        res_pred, corrup_pred = inject_afcb_predicate(resumo_orig, num_erros=1)
        if corrup_pred:
            afcb_records.append({
                "id": f"{doc_id}_pred", "document": fonte, "original_summary": resumo_orig,
                "corrupted_summary": res_pred, "corruptions": corrup_pred, "label": "contradiction"
            })

        # 2. CORREFERÊNCIA
        res_corref, corrup_corref = inject_afcb_coreference(resumo_orig, fonte, num_erros=1)
        if corrup_corref:
            afcb_records.append({
                "id": f"{doc_id}_coref", "document": fonte, "original_summary": resumo_orig,
                "corrupted_summary": res_corref, "corruptions": corrup_corref, "label": "contradiction"
            })
            
        # 3. ENTIDADE
        res_ent, corrup_ent = inject_afcb_entity(resumo_orig, fonte, num_erros=1)
        if corrup_ent:
            afcb_records.append({
                "id": f"{doc_id}_ent", "document": fonte, "original_summary": resumo_orig,
                "corrupted_summary": res_ent, "corruptions": corrup_ent, "label": "contradiction"
            })

        # 4. FACTUAL (Entailment / Base)
        afcb_records.append({
            "id": f"{doc_id}_factual", "document": fonte, "original_summary": resumo_orig,
            "corrupted_summary": resumo_orig, "corruptions": [], "label": "entailment"
        })

    return afcb_records

if __name__ == "__main__":
    print("Iniciando geração do AFCB Benchmark (V4 - Realista + Estruturado)...")
    dataset = load_dataset("ccdv/govreport-summarization", split="test")
    amostra = dataset.select(range(5)) 
    
    benchmark_data = gerar_benchmark_afcb(amostra)
    
    output_file = "AFCB_govreport_FULL_v4.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for record in benchmark_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"✅ Benchmark gerado com {len(benchmark_data)} instâncias em {output_file}")