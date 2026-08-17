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
import random
from typing import Dict, List, Tuple, Optional, Any

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

def inject_realistic_coreference_errors_v1(
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


def inject_realistic_coreference_errors(
    resumo_orig: str,
    fonte_orig: str,
    num_erros: int = 2
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Injeta inconsistências controladas de correferência em um resumo.

    Estratégia:
    1. Identifica pronomes sujeitos referenciais no resumo.
    2. Exige a existência de um antecedente nominal identificável
       nas sentenças anteriores do resumo.
    3. Não substitui o pronome por uma entidade nominal, evitando
       confundir erro de correferência com erro de entidade.
    4. Substitui o pronome por outro pronome semanticamente incompatível
       dentro de um conjunto controlado, preservando sua função sintática.
    5. Mantém no máximo uma alteração por sentença.
    6. Registra o antecedente original e a transformação aplicada,
       permitindo construir o ground truth da perturbação.

    Observação:
        A implementação é deliberadamente restrita a pronomes sujeitos
        com antecedente nominal identificável no contexto local. Isso evita
        tentar resolver correferência geral, que exigiria um resolvedor
        especializado e introduziria maior complexidade.
    """

    if not resumo_orig or not fonte_orig or num_erros <= 0:
        return resumo_orig, 0, []

    doc_resumo = nlp(resumo_orig)

    sentencas = list(doc_resumo.sents)

    if not sentencas:
        return resumo_orig, 0, []

    # ============================================================
    # 1. PRONOMES ELEGÍVEIS
    # ============================================================
    #
    # Restrição:
    # - somente pronomes pessoais;
    # - somente função de sujeito;
    # - somente he, she ou they;
    # - it é excluído para evitar construções expletivas/dummy it;
    # - PRP$ não é considerado, pois a proposta atual trabalha apenas
    #   com pronomes sujeitos.
    #
    # Isso evita:
    #     They -> The Department of Labor
    #
    # e passa a produzir:
    #     They -> He
    #
    # ou:
    #     She -> He
    #
    # mantendo a mesma função sintática.

    PRONOMES_ELEGIVEIS = {
        "he",
        "she",
        "they",
    }

    # ============================================================
    # 2. REGRAS DE SUBSTITUIÇÃO
    # ============================================================
    #
    # A troca é deliberadamente restrita.
    #
    # he   -> she
    # she  -> he
    # they -> he/she
    #
    # Isso produz uma alteração referencial sem inserir uma entidade
    # nominal nova e sem modificar o predicado.
    #
    # A escolha de he/she para "they" também cria uma incompatibilidade
    # explícita de número quando "they" representa um antecedente plural.
    #
    SUBSTITUICOES = {
        "he": ["she"],
        "she": ["he"],
        "they": ["he", "she"],
    }

    # ============================================================
    # 3. FUNÇÕES AUXILIARES
    # ============================================================

    def normalizar(texto: str) -> str:
        """Normaliza texto para comparação lexical simples."""
        return " ".join(texto.lower().strip().split())

    def obter_numero(token) -> Optional[str]:
        """
        Retorna o número morfológico de um token quando disponível.
        Possíveis valores: 'Sing', 'Plur' ou None.
        """
        try:
            valores = token.morph.get("Number")
        except Exception:
            return None

        if not valores:
            return None

        if "Plur" in valores:
            return "Plur"

        if "Sing" in valores:
            return "Sing"

        return None

    def chunk_numero(chunk) -> Optional[str]:
        """
        Obtém o número morfológico do núcleo de um noun chunk.
        """
        try:
            root = chunk.root
            return obter_numero(root)
        except Exception:
            return None

    def eh_chunk_referencial_valido(chunk) -> bool:
        """
        Filtra candidatos a antecedente.

        O antecedente deve:
        - possuir conteúdo lexical;
        - não ser uma estrutura excessivamente longa;
        - não ser um pronome;
        - possuir núcleo nominal.

        A regra é deliberadamente conservadora.
        """

        texto = normalizar(chunk.text)

        if not texto:
            return False

        palavras = texto.split()

        # Evita constituintes excessivamente grandes.
        if len(palavras) > 8:
            return False

        # Evita chunks puramente pronominais.
        if chunk.root.pos_ == "PRON":
            return False

        return True

    def encontrar_antecedente(
        sent_idx: int,
        pronome
    ):
        """
        Procura um antecedente nominal em até duas sentenças anteriores.

        O antecedente mais próximo que:
        1. não seja pronome;
        2. seja nominal;
        3. seja compatível com o número do pronome, quando possível;

        é utilizado como referência.

        Não é um resolvedor geral de correferência. É uma heurística
        deliberadamente limitada para construção do conjunto controlado.
        """

        numero_pronome = obter_numero(pronome)

        candidatos_antecedentes = []

        # Procuramos nas duas sentenças anteriores.
        inicio = max(0, sent_idx - 2)

        for idx in range(sent_idx - 1, inicio - 1, -1):
            sent_anterior = sentencas[idx]

            for chunk in sent_anterior.noun_chunks:
                if not eh_chunk_referencial_valido(chunk):
                    continue

                # Evita chunks que sejam claramente objetos isolados
                # sem potencial de referência.
                if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                    continue

                numero_chunk = chunk_numero(chunk)

                # Primeiro, prioriza compatibilidade de número quando
                # essa informação estiver disponível em ambos.
                compatibilidade_numero = (
                    numero_pronome is not None
                    and numero_chunk is not None
                    and numero_pronome == numero_chunk
                )

                candidatos_antecedentes.append({
                    "sent_idx": idx,
                    "chunk": chunk,
                    "numero": numero_chunk,
                    "compatibilidade_numero": compatibilidade_numero
                })

        if not candidatos_antecedentes:
            return None

        # Prioridade:
        # 1. compatibilidade de número;
        # 2. sentença mais recente.
        candidatos_antecedentes.sort(
            key=lambda x: (
                not x["compatibilidade_numero"],
                -x["sent_idx"]
            )
        )

        return candidatos_antecedentes[0]

    def pronome_substituto(pronome_lemma: str) -> Optional[str]:
        """
        Seleciona uma substituição controlada.
        """
        candidatos = SUBSTITUICOES.get(pronome_lemma.lower())

        if not candidatos:
            return None

        return random.choice(candidatos)

    def ajustar_capitalizacao(original: str, substituto: str) -> str:
        """
        Preserva a capitalização inicial do pronome.
        """
        if not original:
            return substituto

        if original[0].isupper():
            return substituto.capitalize()

        return substituto.lower()

    # ============================================================
    # 4. IDENTIFICAÇÃO DOS PRONOMES COM ANTECEDENTE
    # ============================================================

    candidatos_injecao = []

    for sent_idx, sent in enumerate(sentencas):

        # Máximo de um erro por sentença.
        candidatos_sentenca = []

        for token in sent:

            lemma = token.lemma_.lower().strip()

            # Somente pronomes sujeitos referenciais.
            if token.pos_ != "PRON":
                continue

            if token.dep_ not in {"nsubj", "nsubjpass"}:
                continue

            if token.tag_ != "PRP":
                continue

            if lemma not in PRONOMES_ELEGIVEIS:
                continue

            # Procura antecedente nominal.
            antecedente = encontrar_antecedente(
                sent_idx=sent_idx,
                pronome=token
            )

            if antecedente is None:
                continue

            candidatos_sentenca.append({
                "sent_idx": sent_idx,
                "sent": sent,
                "pronome": token,
                "antecedente": antecedente
            })

        if candidatos_sentenca:
            # Apenas um pronome por sentença.
            escolhido = random.choice(candidatos_sentenca)
            candidatos_injecao.append(escolhido)

    if not candidatos_injecao:
        return resumo_orig, 0, []

    # ============================================================
    # 5. SELEÇÃO DAS SENTENÇAS
    # ============================================================

    qtd_erros = min(
        num_erros,
        len(candidatos_injecao)
    )

    selecionados = random.sample(
        candidatos_injecao,
        qtd_erros
    )

    # Ordenação reversa para preservar offsets de caracteres.
    selecionados.sort(
        key=lambda x: x["pronome"].idx,
        reverse=True
    )

    # ============================================================
    # 6. INJEÇÃO
    # ============================================================

    resumo_mutavel = resumo_orig
    erros = []

    for item in selecionados:

        sent_idx = item["sent_idx"]
        pron = item["pronome"]
        antecedente_info = item["antecedente"]

        pronome_original = pron.text
        pronome_lemma = pron.lemma_.lower().strip()

        # Seleciona pronome incompatível.
        novo_pronome = pronome_substituto(pronome_lemma)

        if novo_pronome is None:
            continue

        # --------------------------------------------------------
        # Controle adicional:
        #
        # Se "they" possui antecedente singular e a substituição for
        # he/she, a mudança altera o número da referência.
        #
        # Se "he"/"she" possui antecedente e ocorre a troca entre
        # he <-> she, a alteração mantém a função sintática, mas
        # modifica a compatibilidade referencial.
        # --------------------------------------------------------

        numero_antecedente = antecedente_info["numero"]
        numero_original = obter_numero(pron)

        # Para "they", somente produzimos a alteração quando:
        # - o antecedente é claramente plural; OU
        # - o antecedente não permite determinar o número.
        #
        # Se o antecedente for claramente singular, "they" pode ser
        # singular they e a alteração poderia não produzir erro factual
        # suficientemente claro.
        if pronome_lemma == "they":
            if numero_antecedente == "Sing":
                continue

        replacement = ajustar_capitalizacao(
            pronome_original,
            novo_pronome
        )

        start_char = pron.idx
        end_char = pron.idx + len(pron.text)

        resumo_mutavel = (
            resumo_mutavel[:start_char]
            + replacement
            + resumo_mutavel[end_char:]
        )

        antecedente = antecedente_info["chunk"]

        erros.append({
            "sentenca_index": sent_idx,
            "pronome_original": pronome_original,
            "pronome_injetado": replacement,
            "antecedente_original": antecedente.text,
            "antecedente_sentenca_index": antecedente_info["sent_idx"],
            "tipo_erro": "coreference",
            "estrategia": "pronoun_swap",
            "numero_pronome_original": numero_original,
            "numero_antecedente": numero_antecedente,
        })

    # Como as alterações foram realizadas do final para o início,
    # os registros são revertidos para manter a ordem textual.
    erros.reverse()

    return (
        resumo_mutavel,
        len(erros),
        erros
    )

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
    amostra = dataset.select(range(5))

    df = processar_dataset_erros(amostra, num_erros=6, seed=42)
    
    arquivo_saida = 'Gerador_erros/govreport_erros_pred_corref_entity_realistas_v2.csv'

    df.to_csv(arquivo_saida, index=False)
    print(f"\n✅ Pipeline concluído! Dataset exportado para: {arquivo_saida}")