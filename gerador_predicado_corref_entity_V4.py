#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerador_predicado_corref_entity_V4.py

Gerador de inconsistências controladas para resumos do GovReport.

Categorias:
    1. predicate
    2. coreference
    3. entity

Princípios gerais:
    - alterações somente quando os critérios de elegibilidade são satisfeitos;
    - no máximo uma alteração por sentença em cada versão;
    - nenhuma alteração é forçada quando não há candidato válido;
    - todas as alterações são registradas para construção do ground truth;
    - preservação de offsets por substituições aplicadas da direita para a esquerda.

Estratégias de PREDICADO:
    1. antonym
    2. negation
    3. tense_shift

Estratégia de COREFERENCE:
    - pronoun swapping controlado;
    - antecedente nominal local identificável;
    - apenas pronomes sujeitos elegíveis;
    - não substitui o pronome por uma entidade nominal.

Estratégia de ENTITY:
    - entidade do resumo -> entidade do documento-fonte;
    - mesmo tipo NER;
    - candidata semanticamente próxima;
    - candidata fora do contexto local da sentença modificada;
    - nenhuma substituição arbitrária.
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple

import lemminflect
import pandas as pd
import spacy
from datasets import load_dataset


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_NAME = "en_core_web_md"

try:
    nlp = spacy.load(MODEL_NAME)
except OSError as exc:
    raise RuntimeError(
        f"Modelo spaCy '{MODEL_NAME}' não encontrado. "
        f"Instale-o antes de executar o script."
    ) from exc


# ============================================================
# CARREGAMENTO DO DICIONÁRIO DE ANTÔNIMOS
# ============================================================

try:
    with open("antonimos_govreport.json", "r", encoding="utf-8") as f:
        ANTONIMOS_DOMINIO = json.load(f)
except FileNotFoundError:
    print(
        "AVISO: 'antonimos_govreport.json' não encontrado. "
        "A estratégia de antônimos ficará indisponível."
    )
    ANTONIMOS_DOMINIO = {}


# ============================================================
# FUNÇÕES GERAIS
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparações lexicais.
    """
    return " ".join(text.lower().strip().split())


def safe_float(value: Any) -> Optional[float]:
    """
    Converte valor para float quando possível.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_token_replacement_char(
    sent_text: str,
    target_token,
    replacement_string: str
) -> str:
    """
    Substitui um token usando offset relativo à sentença.
    """
    start_char = target_token.idx - target_token.sent.start_char
    end_char = start_char + len(target_token.text)

    return (
        sent_text[:start_char]
        + replacement_string
        + sent_text[end_char:]
    )


def replace_span_in_text(
    text: str,
    start_char: int,
    end_char: int,
    replacement: str
) -> str:
    """
    Substituição genérica por offsets absolutos.
    """
    return text[:start_char] + replacement + text[end_char:]


def adjust_capitalization_1(
    original: str,
    replacement: str
) -> str:
    """
    Preserva a capitalização inicial.
    """
    if not original:
        return replacement

    if original[0].isupper():
        return replacement.capitalize()

    return replacement.lower()

def adjust_capitalization(original: str, replacement: str) -> str:
    if not original:
        return replacement
    # NOVA REGRA: Se o original for uma sigla (todo em maiúsculo), mantém o substituto igual.
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement.lower()

def token_has_negation(token) -> bool:
    """
    Verifica se o token possui negação sintática.
    """
    for child in token.children:
        if child.dep_ == "neg":
            return True

    return False


def sentence_has_negation(sent) -> bool:
    """
    Verifica se existe negação explícita na sentença.
    """
    for token in sent:
        if token.dep_ == "neg":
            return True

        if token.lower_ in {"n't", "not", "never", "no"}:
            return True

    return False


def is_auxiliary_or_modal(token) -> bool:
    """
    Evita modificar verbos que não representam adequadamente o
    predicado lexical principal.
    """
    if token.dep_ in {"aux", "auxpass"}:
        return True

    if token.lemma_.lower() in {
        "be",
        "have",
        "do",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
    }:
        return True

    return False


# ============================================================
# ============================================================
# PREDICADO
# ============================================================
# ============================================================

PREDICATE_SUBJECT_DEPS = {
    "nsubj",
    "nsubjpass",
    "csubj",
}

PREDICATE_COMPLEMENT_DEPS = {
    "obj",
    "dobj",
    "pobj",
    "acomp",
    "xcomp",
    "ccomp",
    "attr",
    "oprd",
}

FINITE_VERB_TAGS = {
    "VBD",
    "VBP",
    "VBZ",
}

TENSE_SHIFT_TAGS = {
    "VBD",
    "VBP",
    "VBZ",
}

NEGATION_TAGS = {
    "VBD",
    "VBP",
    "VBZ",
}


def has_predicate_structure(token) -> bool:
    """
    Verifica se o verbo possui uma estrutura argumental mínima.

    Exigência:
        verbo + sujeito + objeto/complemento

    Essa restrição reduz alterações em verbos isolados ou estruturas
    que não expressem uma proposição factual suficientemente clara.
    """

    if token.pos_ != "VERB":
        return False

    if not token.is_alpha:
        return False

    if is_auxiliary_or_modal(token):
        return False

    has_subject = any(
        child.dep_ in PREDICATE_SUBJECT_DEPS
        for child in token.children
    )

    has_complement = any(
        child.dep_ in PREDICATE_COMPLEMENT_DEPS
        for child in token.children
    )

    return has_subject and has_complement


def get_antonym_lemma(
    lemma: str,
    antonyms: Dict[str, Any]
) -> Optional[str]:
    """
    Obtém um antônimo do dicionário.

    Suporta:
        "approve": "reject"
        ou
        "approve": ["reject", ...]
    """

    value = antonyms.get(lemma)

    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None
        return str(value[0]).strip()

    if isinstance(value, str):
        return value.strip()

    return None


def get_inflected_antonym(
    verb_token,
    antonym_lemma: str
) -> Optional[str]:
    """
    Tenta gerar o antônimo mantendo a flexão do verbo original.
    """

    try:
        inflections = lemminflect.getInflection(
            antonym_lemma,
            tag=verb_token.tag_
        )
    except Exception:
        inflections = None

    if inflections:
        candidate = inflections[0].strip()

        if candidate.lower() != verb_token.text.lower():
            return candidate

    # fallback
    if antonym_lemma.lower() != verb_token.text.lower():
        return antonym_lemma

    return None


def build_negated_predicate(
    verb_token
) -> Optional[Tuple[str, str]]:
    """
    Constrói negação adequada para formas verbais simples.

    Retorna:
        (texto_substituto, descrição_da_transformação)
    """

    if verb_token.tag_ not in NEGATION_TAGS:
        return None

    if token_has_negation(verb_token):
        return None

    lemma = verb_token.lemma_.lower()

    if verb_token.tag_ == "VBD":
        return f"did not {lemma}", "negation"

    if verb_token.tag_ == "VBZ":
        return f"does not {lemma}", "negation"

    if verb_token.tag_ == "VBP":
        return f"do not {lemma}", "negation"

    return None


def build_tense_shift(
    verb_token
) -> Optional[Tuple[str, str]]:
    """
    Altera o predicado para uma construção futura.

    A operação só é aplicada quando:
        - o verbo é finito;
        - o verbo não é auxiliar/modal;
        - a sentença não apresenta já uma construção futura explícita.
    """

    if verb_token.tag_ not in TENSE_SHIFT_TAGS:
        return None

    # Evita deslocamentos redundantes.
    for child in verb_token.children:
        if child.lemma_.lower() == "will":
            return None

    lemma = verb_token.lemma_.lower()

    if lemma in {"be", "have", "do"}:
        return None

    replacement = f"will {lemma}"

    if replacement.lower() == verb_token.text.lower():
        return None

    return replacement, "tense_shift"


def generate_predicate_transformation(
    verb_token,
    sent,
    antonyms: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Gera uma transformação de predicado elegível.

    Ordem de prioridade:
        1. antônimo;
        2. negação;
        3. mudança temporal.

    A prioridade evita escolher uma transformação mais fraca quando
    existe uma alternativa lexical explicitamente definida.
    """

    lemma = verb_token.lemma_.lower()

    # --------------------------------------------------------
    # 1. ANTÔNIMO
    # --------------------------------------------------------

    antonym_lemma = get_antonym_lemma(
        lemma,
        antonyms
    )

    if antonym_lemma:
        new_verb = get_inflected_antonym(
            verb_token,
            antonym_lemma
        )

        if new_verb:
            return {
                "replacement": new_verb,
                "strategia": "antonym",
                "original": verb_token.text,
                "injetado": new_verb,
                "lemma_original": lemma,
                "lemma_injetado": antonym_lemma,
            }

    # --------------------------------------------------------
    # 2. NEGAÇÃO
    # --------------------------------------------------------

    negation_result = build_negated_predicate(
        verb_token
    )

    if negation_result:
        replacement, strategy = negation_result

        return {
            "replacement": replacement,
            "strategia": strategy,
            "original": verb_token.text,
            "injetado": replacement,
            "lemma_original": lemma,
            "lemma_injetado": f"NOT_{lemma}",
        }

    # --------------------------------------------------------
    # 3. MUDANÇA TEMPORAL
    # --------------------------------------------------------

    tense_result = build_tense_shift(
        verb_token
    )

    if tense_result:
        replacement, strategy = tense_result

        return {
            "replacement": replacement,
            "strategia": strategy,
            "original": verb_token.text,
            "injetado": replacement,
            "lemma_original": lemma,
            "lemma_injetado": f"FUTURE_{lemma}",
        }

    return None


def inject_realistic_predicate_errors(
    resumo_orig: str,
    fonte_orig: str,
    num_erros: int = 2
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Injeta inconsistências de predicado de maneira controlada.

    A função:
        - identifica apenas verbos com sujeito e complemento;
        - evita auxiliares/modal/copulares;
        - permite no máximo uma alteração por sentença;
        - prioriza antônimos;
        - usa negação quando não há antônimo;
        - usa alteração temporal como último recurso;
        - não força a alteração quando não existe transformação válida.

    Observação:
        fonte_orig é mantido na assinatura para manter compatibilidade
        com o pipeline principal. Nesta estratégia, a transformação do
        predicado é baseada na estrutura do resumo, não na seleção de
        um predicado externo do documento-fonte.
    """

    if not resumo_orig or num_erros <= 0:
        return resumo_orig, 0, []

    doc = nlp(resumo_orig)
    sentencas = list(doc.sents)

    if not sentencas:
        return resumo_orig, 0, []

    # --------------------------------------------------------
    # 1. CONSTRUIR CANDIDATOS
    # --------------------------------------------------------

    candidatos = []

    for sent_idx, sent in enumerate(sentencas):

        verbos_elegiveis = []

        for token in sent:
            if not has_predicate_structure(token):
                continue

            transformation = generate_predicate_transformation(
                token,
                sent,
                ANTONIMOS_DOMINIO
            )

            if transformation is None:
                continue

            verbos_elegiveis.append(
                {
                    "token": token,
                    "transformacao": transformation,
                }
            )

        # Máximo de uma alteração por sentença.
        if verbos_elegiveis:
            candidatos.append(
                {
                    "sent_idx": sent_idx,
                    "sent": sent,
                    "verbos": verbos_elegiveis,
                }
            )

    if not candidatos:
        return resumo_orig, 0, []

    # --------------------------------------------------------
    # 2. SELECIONAR SENTENÇAS
    # --------------------------------------------------------

    qtd = min(
        num_erros,
        len(candidatos)
    )

    selecionados = random.sample(
        candidatos,
        qtd
    )

    # --------------------------------------------------------
    # 3. ESCOLHER UM VERBO POR SENTENÇA
    # --------------------------------------------------------

    alteracoes = []

    for item in selecionados:

        verbo_escolhido = random.choice(
            item["verbos"]
        )

        alteracoes.append(
            {
                "sent_idx": item["sent_idx"],
                "sent": item["sent"],
                "token": verbo_escolhido["token"],
                "transformacao": verbo_escolhido["transformacao"],
            }
        )

    # --------------------------------------------------------
    # 4. APLICAÇÃO POR OFFSETS ABSOLUTOS
    # --------------------------------------------------------

    alteracoes.sort(
        key=lambda x: x["token"].idx,
        reverse=True
    )

    resumo_mutavel = resumo_orig
    logs = []

    for item in alteracoes:

        token = item["token"]
        transformation = item["transformacao"]

        start_char = token.idx
        end_char = token.idx + len(token.text)

        replacement = adjust_capitalization(
            token.text,
            transformation["replacement"]
        )

        resumo_mutavel = replace_span_in_text(
            resumo_mutavel,
            start_char,
            end_char,
            replacement
        )

        logs.append(
            {
                "sentenca_index": item["sent_idx"],
                "token_original": token.text,
                "lemma_original": transformation["lemma_original"],
                "tipo_erro": "predicate",
                "estrategia": transformation["strategia"],
                "token_injetado": replacement,
                "lemma_injetado": transformation["lemma_injetado"],
            }
        )

    logs.reverse()

    return (
        resumo_mutavel,
        len(logs),
        logs
    )


# ============================================================
# ============================================================
# COREFERENCE
# ============================================================
# ============================================================

import random
from typing import Tuple, List, Dict, Any, Optional
# Assumindo que 'nlp' e 'replace_span_in_text' estão no seu escopo global

def inject_realistic_coreference_errors(
    resumo_orig: str,
    fonte_orig: str,
    num_erros: int = 2
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Injeta inconsistências controladas de correferência forçando a substituição
    pronominal por meio de divergência de número gramatical (Singular <-> Plural).

    Regras de controle
    ------------------
        - Somente troca de pronome para pronome (ex: it -> they, they -> he).
        - O referente alternativo DEVE ter número gramatical diferente do original.
        - A substituição (X -> X) é estritamente proibida.
        - O candidato alternativo deve estar presente no documento-fonte.
        - Não é permitida mais de uma alteração por sentença.
    """

    if not resumo_orig or not fonte_orig or num_erros <= 0:
        return resumo_orig, 0, []

    doc_resumo = nlp(resumo_orig)
    doc_fonte = nlp(fonte_orig)

    sentencas_resumo = list(doc_resumo.sents)
    sentencas_fonte = list(doc_fonte.sents)

    if not sentencas_resumo or not sentencas_fonte:
        return resumo_orig, 0, []

    # ============================================================
    # CONFIGURAÇÕES
    # ============================================================

    PRONOMES_SUJEITO = {"he", "she", "they", "it"}
    PRONOMES_POSSESSIVOS = {"his", "her", "their", "its"}

    # ============================================================
    # FUNÇÕES AUXILIARES
    # ============================================================

    def normalizar(texto: str) -> str:
        return " ".join(texto.lower().strip().split())

    def obter_numero_token(token) -> Optional[str]:
        try:
            valores = token.morph.get("Number")
        except Exception:
            return None
        if not valores: return None
        if "Plur" in valores: return "Plur"
        if "Sing" in valores: return "Sing"
        return None

    def obter_numero_chunk(chunk) -> Optional[str]:
        try:
            return obter_numero_token(chunk.root)
        except Exception:
            return None

    def determinar_funcao_referencial(token) -> Optional[str]:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            return "subject"
        if token.dep_ == "poss":
            return "possessive"
        return None

    def eh_chunk_referencial_valido(chunk) -> bool:
        texto = normalizar(chunk.text)
        if not texto or len(texto.split()) > 8:
            return False
        if chunk.root.pos_ not in {"NOUN", "PROPN"}:
            return False
        return True

    def eh_antecedente_sujeito(chunk) -> bool:
        return chunk.root.dep_ in {"nsubj", "nsubjpass"}

    def encontrar_antecedente_original(sent_idx: int, expressao) -> Optional[Dict[str, Any]]:
        numero_expressao = obter_numero_token(expressao)
        candidatos = []
        inicio = max(0, sent_idx - 2)

        for idx in range(sent_idx - 1, inicio - 1, -1):
            for chunk in sentencas_resumo[idx].noun_chunks:
                if not eh_chunk_referencial_valido(chunk):
                    continue
                numero_chunk = obter_numero_chunk(chunk)
                compatibilidade_numero = (
                    numero_expressao is not None
                    and numero_chunk is not None
                    and numero_expressao == numero_chunk
                )
                candidatos.append({
                    "sent_idx": idx,
                    "chunk": chunk,
                    "numero": numero_chunk,
                    "is_subject": eh_antecedente_sujeito(chunk),
                    "compatibilidade_numero": compatibilidade_numero,
                })

        if not candidatos: return None
        candidatos.sort(key=lambda x: (not x["compatibilidade_numero"], not x["is_subject"], -x["sent_idx"]))
        return candidatos[0]

    def encontrar_alternativas_locais(sent_idx: int, antecedente_original: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidatos = []
        antecedente_texto = normalizar(antecedente_original["chunk"].text)
        inicio = max(0, sent_idx - 2)

        for idx in range(sent_idx - 1, inicio - 1, -1):
            for chunk in sentencas_resumo[idx].noun_chunks:
                if not eh_chunk_referencial_valido(chunk):
                    continue
                texto = normalizar(chunk.text)
                if texto == antecedente_texto:
                    continue
                
                numero = obter_numero_chunk(chunk)
                
                # REGRA CRUCIAL: O número gramatical DEVE ser diferente para forçar a mudança do pronome
                if numero is None or numero == antecedente_original["numero"]:
                    continue

                candidatos.append({
                    "sent_idx": idx,
                    "chunk": chunk,
                    "numero": numero,
                    "is_subject": eh_antecedente_sujeito(chunk),
                })
        return candidatos

    def localizar_entidade_fonte(texto_alternativo: str) -> Optional[Any]:
        alvo = normalizar(texto_alternativo)
        if not alvo: return None
        for ent in doc_fonte.ents:
            if normalizar(ent.text) == alvo:
                return ent
        for chunk in doc_fonte.noun_chunks:
            if normalizar(chunk.text) == alvo:
                return chunk
        return None

    def extrair_pronome_alternativo(antecedente_alternativo: Dict[str, Any], funcao: str) -> Optional[str]:
        numero = antecedente_alternativo["numero"]
        chunk = antecedente_alternativo["chunk"]
        
        # Verifica se o referente é uma Pessoa usando NER para inferir he/she
        is_person = False
        for ent in doc_resumo.ents:
            if ent.label_ == "PERSON" and normalizar(ent.text) in normalizar(chunk.text):
                is_person = True
                break

        if funcao == "subject":
            if numero == "Plur": return "they"
            if numero == "Sing":
                return random.choice(["he", "she"]) if is_person else "it"

        if funcao == "possessive":
            if numero == "Plur": return "their"
            if numero == "Sing":
                return random.choice(["his", "her"]) if is_person else "its"
                
        return None

    def ajustar_capitalizacao(original: str, replacement: str) -> str:
        if not original: return replacement
        if original[0].isupper(): return replacement.capitalize()
        return replacement.lower()

    # ============================================================
    # IDENTIFICAÇÃO DAS EXPRESSÕES ANAFÓRICAS
    # ============================================================

    candidatos_injecao = []

    for sent_idx, sent in enumerate(sentencas_resumo):
        candidatos_sentenca = []

        for token in sent:
            token_lower = token.text.lower().strip()
            lemma = token.lemma_.lower().strip()
            funcao = determinar_funcao_referencial(token)

            if funcao is None:
                continue

            if funcao == "subject":
                chave = token_lower if token_lower in PRONOMES_SUJEITO else lemma
                if chave not in PRONOMES_SUJEITO: continue
            elif funcao == "possessive":
                chave = token_lower if token_lower in PRONOMES_POSSESSIVOS else lemma
                if chave not in PRONOMES_POSSESSIVOS: continue

            antecedente_original = encontrar_antecedente_original(sent_idx, token)
            if antecedente_original is None: continue

            alternativas = encontrar_alternativas_locais(sent_idx, antecedente_original)
            if not alternativas: continue

            for alternativa in alternativas:
                fonte_ref = localizar_entidade_fonte(alternativa["chunk"].text)
                if fonte_ref is None: continue

                pronome_alternativo = extrair_pronome_alternativo(alternativa, funcao)
                if pronome_alternativo is None: continue

                # REGRA CRUCIAL 2: A injeção deve ser textualmente observável (nunca X -> X)
                if pronome_alternativo.lower() == token.text.lower().strip():
                    continue

                candidatos_sentenca.append({
                    "sent_idx": sent_idx,
                    "sent": sent,
                    "expressao": token,
                    "funcao": funcao,
                    "expressao_original": token.text,
                    "antecedente_original": antecedente_original,
                    "antecedente_alternativo": alternativa,
                    "fonte_alternativo": fonte_ref,
                    "pronome_alternativo": pronome_alternativo,
                })

        # Adiciona no máximo uma alteração válida por sentença
        if candidatos_sentenca:
            candidatos_injecao.append(random.choice(candidatos_sentenca))

    if not candidatos_injecao:
        return resumo_orig, 0, []

    # ============================================================
    # SELEÇÃO E APLICAÇÃO DAS PERTURBAÇÕES
    # ============================================================

    qtd_erros = min(num_erros, len(candidatos_injecao))
    selecionados = random.sample(candidatos_injecao, qtd_erros)
    selecionados.sort(key=lambda x: x["expressao"].idx, reverse=True)

    resumo_mutavel = resumo_orig
    logs = []

    for item in selecionados:
        expressao = item["expressao"]
        expressao_original = item["expressao_original"]
        pronome_alternativo = item["pronome_alternativo"]
        antecedente_original = item["antecedente_original"]
        antecedente_alternativo = item["antecedente_alternativo"]

        replacement = ajustar_capitalizacao(expressao_original, pronome_alternativo)

        start_char = expressao.idx
        end_char = expressao.idx + len(expressao.text)

        resumo_mutavel = replace_span_in_text(
            resumo_mutavel,
            start_char,
            end_char,
            replacement
        )

        logs.append({
            "sentenca_index": item["sent_idx"],
            "tipo_erro": "coreference",
            "estrategia": "pronoun_to_pronoun_number_shift", # Nome atualizado para clareza
            "funcao_referencial": item["funcao"],
            "expressao_original": expressao_original,
            "referente_original": antecedente_original["chunk"].text,
            "referente_alternativo": antecedente_alternativo["chunk"].text,
            "expressao_injetada": replacement,
            "sentenca_antecedente_original": antecedente_original["sent_idx"],
            "sentenca_referente_alternativo": antecedente_alternativo["sent_idx"],
            "numero_referente_original": antecedente_original["numero"],
            "numero_referente_alternativo": antecedente_alternativo["numero"],
            "fonte_referente_alternativo": antecedente_alternativo["chunk"].text,
        })

    logs.reverse()

    return resumo_mutavel, len(logs), logs

# ============================================================
# ============================================================
# ENTITY
# ============================================================
# ============================================================

def get_sentence_entity_occurrences(doc) -> Dict[int, List[Any]]:
    """
    Organiza entidades do documento por índice de sentença.
    """

    sent_map = {}

    for sent_idx, sent in enumerate(doc.sents):
        sent_entities = []

        for ent in sent.ents:
            sent_entities.append(ent)

        sent_map[sent_idx] = sent_entities

    return sent_map


def entity_sentence_index(
    entity,
    sentence_map
) -> Optional[int]:
    """
    Retorna o índice da sentença em que a entidade está.
    """

    for idx, entities in sentence_map.items():
        for ent in entities:
            if ent.start_char == entity.start_char:
                return idx

    return None


def inject_realistic_entity_errors(
    resumo_orig: str,
    fonte_orig: str,
    num_erros: int = 2
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Injeta inconsistências de entidade de forma controlada.

    Regras:

    1. a sentença do resumo precisa conter pelo menos uma entidade;
    2. a entidade substituta deve existir no documento-fonte;
    3. original e substituta devem possuir o mesmo label NER;
    4. a entidade substituta deve ser lexicalmente diferente;
    5. a substituta não pode ocorrer na sentença alvo;
    6. a substituta não pode ocorrer nas sentenças imediatamente
       adjacentes;
    7. a substituta é escolhida entre as três candidatas semanticamente
       mais próximas;
    8. a mesma entidade substituta não é reutilizada;
    9. no máximo uma alteração por sentença;
    10. se não houver candidato válido, a sentença não é alterada.
    """

    if not resumo_orig or not fonte_orig or num_erros <= 0:
        return resumo_orig, 0, []

    doc_resumo = nlp(resumo_orig)
    doc_fonte = nlp(fonte_orig)

    sentencas_resumo = list(doc_resumo.sents)
    sentencas_fonte = list(doc_fonte.sents)

    if not sentencas_resumo or not sentencas_fonte:
        return resumo_orig, 0, []

    # --------------------------------------------------------
    # 1. MAPEAMENTO DAS ENTIDADES DA FONTE
    # --------------------------------------------------------

    entidades_por_label: Dict[str, List[Any]] = {}

    for ent in doc_fonte.ents:

        texto_normalizado = normalize_text(
            ent.text
        )

        if not texto_normalizado:
            continue

        entidades_por_label.setdefault(
            ent.label_,
            []
        ).append(ent)

    if not entidades_por_label:
        return resumo_orig, 0, []

    # --------------------------------------------------------
    # 2. MAPA SENTENÇA -> ENTIDADES
    # --------------------------------------------------------

    entidades_resumo_por_sentenca = (
        get_sentence_entity_occurrences(
            doc_resumo
        )
    )

    # --------------------------------------------------------
    # 3. CANDIDATOS DE INJEÇÃO
    # --------------------------------------------------------

    candidatos_injecao = []

    for sent_idx, sent in enumerate(
        sentencas_resumo
    ):

        entidades = (
            entidades_resumo_por_sentenca
            .get(sent_idx, [])
        )

        if not entidades:
            continue

        # Apenas entidades que possuem candidatos
        # válidos são consideradas.
        for entidade_alvo in entidades:

            label = entidade_alvo.label_

            candidatos_fonte = (
                entidades_por_label.get(
                    label,
                    []
                )
            )

            if not candidatos_fonte:
                continue

            texto_alvo = normalize_text(
                entidade_alvo.text
            )

            # Conjunto lexical do contexto local.
            contexto_local = set(
                normalize_text(token.text)
                for token in sent
                if token.is_alpha
            )

            if sent_idx > 0:
                sent_anterior = (
                    sentencas_resumo[
                        sent_idx - 1
                    ]
                )

                contexto_local.update(
                    normalize_text(token.text)
                    for token in sent_anterior
                    if token.is_alpha
                )

            if sent_idx < len(sentencas_resumo) - 1:
                sent_proxima = (
                    sentencas_resumo[
                        sent_idx + 1
                    ]
                )

                contexto_local.update(
                    normalize_text(token.text)
                    for token in sent_proxima
                    if token.is_alpha
                )

            # ------------------------------------------------
            # CANDIDATOS DO MESMO TIPO
            # ------------------------------------------------

            candidatos_validos = []

            for candidato in candidatos_fonte:

                candidato_norm = normalize_text(
                    candidato.text
                )

                # Não pode ser a própria entidade.
                if candidato_norm == texto_alvo:
                    continue

                # Precisa ter texto diferente.
                if not candidato_norm:
                    continue

                # O candidato não pode aparecer no contexto local.
                partes_candidato = set(
                    candidato_norm.split()
                )

                if partes_candidato & contexto_local:
                    continue

                # ------------------------------------------------
                # DISTÂNCIA SENTENCIAL
                # ------------------------------------------------
                #
                # O candidato deve estar pelo menos duas sentenças
                # distante da sentença modificada.
                #
                # Isso implementa explicitamente a ideia de
                # "confusão entre informações do documento",
                # evitando trocar pela entidade já presente no
                # contexto imediato.
                # ------------------------------------------------

                distancia_minima_ok = True

                for fonte_sent_idx, fonte_sent in enumerate(
                    sentencas_fonte
                ):
                    if (
                        candidato.start_char >= fonte_sent.start_char
                        and candidato.end_char <= fonte_sent.end_char
                    ):
                        distancia = abs(
                            fonte_sent_idx - sent_idx
                        )

                        if distancia < 2:
                            distancia_minima_ok = False

                        break

                if not distancia_minima_ok:
                    continue

                # ------------------------------------------------
                # SIMILARIDADE SEMÂNTICA
                # ------------------------------------------------

                try:
                    alvo_doc = nlp(
                        entidade_alvo.text
                    )

                    candidato_doc = nlp(
                        candidato.text
                    )

                    if (
                        alvo_doc.has_vector
                        and candidato_doc.has_vector
                    ):
                        similarity = (
                            alvo_doc.similarity(
                                candidato_doc
                            )
                        )
                    else:
                        similarity = 0.0

                except Exception:
                    similarity = 0.0

                candidatos_validos.append(
                    {
                        "entity": candidato,
                        "similarity": similarity,
                    }
                )

            if not candidatos_validos:
                continue

            candidatos_injecao.append(
                {
                    "sent_idx": sent_idx,
                    "sent": sent,
                    "alvo": entidade_alvo,
                    "candidatos": candidatos_validos,
                }
            )

            # No máximo uma entidade alvo por sentença.
            break

    if not candidatos_injecao:
        return resumo_orig, 0, []

    # --------------------------------------------------------
    # 4. SELECIONAR SENTENÇAS
    # --------------------------------------------------------

    qtd_erros = min(
        num_erros,
        len(candidatos_injecao)
    )

    selecionados = random.sample(
        candidatos_injecao,
        qtd_erros
    )

    selecionados.sort(
        key=lambda x: x["alvo"].start_char,
        reverse=True
    )

    # --------------------------------------------------------
    # 5. INJEÇÃO
    # --------------------------------------------------------

    resumo_mutavel = resumo_orig
    logs = []
    usados = set()

    for item in selecionados:

        entidade_alvo = item["alvo"]

        candidatos = [
            c
            for c in item["candidatos"]
            if normalize_text(
                c["entity"].text
            ) not in usados
        ]

        if not candidatos:
            continue

        # ----------------------------------------------------
        # TOP-3 MAIS SEMELHANTES
        # ----------------------------------------------------

        candidatos.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        top_k = min(
            3,
            len(candidatos)
        )

        top_candidatos = candidatos[:top_k]

        escolhido = random.choice(
            top_candidatos
        )

        nova_entidade = escolhido[
            "entity"
        ]

        nova_entidade_texto = (
            nova_entidade.text
        )

        nova_entidade_norm = normalize_text(
            nova_entidade_texto
        )

        usados.add(
            nova_entidade_norm
        )

        # ----------------------------------------------------
        # PRESERVAÇÃO DE CAPITALIZAÇÃO
        # ----------------------------------------------------

        replacement = adjust_capitalization(
            entidade_alvo.text,
            nova_entidade_texto
        )

        # ----------------------------------------------------
        # SUBSTITUIÇÃO POR OFFSET ABSOLUTO
        # ----------------------------------------------------

        resumo_mutavel = replace_span_in_text(
            resumo_mutavel,
            entidade_alvo.start_char,
            entidade_alvo.end_char,
            replacement
        )

        # ----------------------------------------------------
        # DISTÂNCIA NO DOCUMENTO-FONTE
        # ----------------------------------------------------

        fonte_sent_idx = None

        for idx, fonte_sent in enumerate(
            sentencas_fonte
        ):
            if (
                nova_entidade.start_char
                >= fonte_sent.start_char
                and
                nova_entidade.end_char
                <= fonte_sent.end_char
            ):
                fonte_sent_idx = idx
                break

        if fonte_sent_idx is not None:
            distancia = abs(
                fonte_sent_idx
                - item["sent_idx"]
            )
        else:
            distancia = None

        logs.append(
            {
                "sentenca_index": item["sent_idx"],
                "tipo_erro": "entity",
                "label_ner": entidade_alvo.label_,
                "token_original": entidade_alvo.text,
                "token_injetado": replacement,
                "entidade_fonte": nova_entidade_texto,
                "similaridade": safe_float(
                    escolhido["similarity"]
                ),
                "sentenca_fonte_index": fonte_sent_idx,
                "distancia_sentencial": distancia,
                "estrategia": "same_ner_label_semantic_confusion",
            }
        )

    logs.reverse()

    return (
        resumo_mutavel,
        len(logs),
        logs
    )


# ============================================================
# PROCESSAMENTO DO DATASET
# ============================================================

def processar_dataset_erros(
    dataset_split,
    num_erros: int = 2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Gera as três versões perturbadas de cada resumo.
    """

    random.seed(seed)

    resultados = []

    for example in dataset_split:

        fonte = example["report"]
        resumo_orig = example["summary"]

        # ----------------------------------------------------
        # PREDICADO
        # ----------------------------------------------------

        (
            res_pred,
            cnt_pred,
            toks_pred
        ) = inject_realistic_predicate_errors(
            resumo_orig,
            fonte,
            num_erros
        )

        # ----------------------------------------------------
        # CORREFERÊNCIA
        # ----------------------------------------------------

        (
            res_corref,
            cnt_corref,
            toks_corref
        ) = inject_realistic_coreference_errors(
            resumo_orig,
            fonte,
            num_erros
        )

        # ----------------------------------------------------
        # ENTIDADE
        # ----------------------------------------------------

        (
            res_entity,
            cnt_entity,
            toks_entity
        ) = inject_realistic_entity_errors(
            resumo_orig,
            fonte,
            num_erros
        )

        resultados.append(
            {
                "id": example.get(
                    "id",
                    "N/A"
                ),

                "fonte": fonte,

                "resumo_original": resumo_orig,

                # ------------------------------
                # PREDICADO
                # ------------------------------

                "resumo_predicado": res_pred,

                "qtd_erros_predicado": cnt_pred,

                "tokens_predicado": toks_pred,

                # ------------------------------
                # CORREFERÊNCIA
                # ------------------------------

                "resumo_corref": res_corref,

                "qtd_erros_corref": cnt_corref,

                "tokens_corref": toks_corref,

                # ------------------------------
                # ENTIDADE
                # ------------------------------

                "resumo_entity": res_entity,

                "qtd_erros_entity": cnt_entity,

                "tokens_entity": toks_entity,
            }
        )

    return pd.DataFrame(
        resultados
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    print(
        "Iniciando pipeline de geração de erros "
        "(V4 - Avaliação Controlada)..."
    )

    try:
        dataset = load_dataset(
            "ccdv/govreport-summarization",
            split="test"
        )
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível carregar o dataset "
            "'ccdv/govreport-summarization'."
        ) from exc

    # --------------------------------------------------------
    # AMOSTRA
    # --------------------------------------------------------

    amostra = dataset.select(
        range(
            min(5, len(dataset))
        )
    )

    df = processar_dataset_erros(
        amostra,
        num_erros=5,
        seed=42
    )

    # --------------------------------------------------------
    # SAÍDA
    # --------------------------------------------------------

    arquivo_saida = (
        "Gerador_erros/"
        "govreport_erros_pred_corref_entity_"
        "controlados_v6.csv"
    )

    df.to_csv(
        arquivo_saida,
        index=False
    )

    print(
        f"\n✅ Pipeline concluído!"
        f"\n✅ Registros gerados: {len(df)}"
        f"\n✅ Arquivo: {arquivo_saida}"
    )

    # --------------------------------------------------------
    # RESUMO DAS QUANTIDADES
    # --------------------------------------------------------

    print("\nDistribuição de alterações:")

    print(
        "Predicado:",
        int(
            df["qtd_erros_predicado"].sum()
        )
    )

    print(
        "Correferência:",
        int(
            df["qtd_erros_corref"].sum()
        )
    )

    print(
        "Entidade:",
        int(
            df["qtd_erros_entity"].sum()
        )
    )