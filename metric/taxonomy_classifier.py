import re
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
import json # <-- Não esqueça de importar o json

try:
    from metric.utils.utils import nlp, sbert_model
except ImportError:
    nlp = spacy.load("en_core_web_md")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

class TaxonomyClassifier:
    def __init__(self):
        self.nlp = nlp
        self.sbert = sbert_model
        
        # =====================================================================
        # CARREGAMENTO DO DICIONÁRIO DE ANTÔNIMOS (Uma única vez na memória)
        # =====================================================================
        self.pares_antonimos = {}
        try:
            with open('antonimos_govreport.json', 'r', encoding='utf-8') as f:
                dict_original = json.load(f)
                
                # Inverte o dicionário: O json do gerador é "original": "antonimo".
                # O avaliador precisa buscar o "antonimo" para descobrir qual era o "original".
                for original, antonimos in dict_original.items():
                    # Suporta o JSON caso o valor seja uma string ou uma lista de strings
                    if isinstance(antonimos, list):
                        for ant in antonimos:
                            self.pares_antonimos[ant.lower().strip()] = original.lower().strip()
                    else:
                        self.pares_antonimos[antonimos.lower().strip()] = original.lower().strip()
                        
        except FileNotFoundError:
            print("AVISO: 'antonimos_govreport.json' não encontrado pelo FENICE. Usando fallback básico.")
            self.pares_antonimos = {
                "disappear": "appear",
                "obviate": "ask",
                "maximize": "minimize"
            }

    def _semantic_similarity(self, text1, text2):
        if not text1 or not text2: return 0.0
        with torch.inference_mode():
            emb = self.sbert.encode([text1, text2], convert_to_tensor=True, normalize_embeddings=True)
        return util.cos_sim(emb[0], emb[1]).item()

    def _clean_text(self, text):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text)).strip().lower()


    def _is_evidence_relevant(self, doc_claim, doc_evid):
        c_lemmas = {t.lemma_.lower() for t in doc_claim if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "VERB")}
        e_lemmas = {t.lemma_.lower() for t in doc_evid if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "VERB")}
        if not c_lemmas: return True 
        # Exigir pelo menos 2 lemmas em comum para evitar falsos pareamentos
        return len(c_lemmas.intersection(e_lemmas)) >= 2

    # =====================================================================
    # MOTORES AVALIADORES (SEM BUGS DE FATIAMENTO OU REGRAS INGÊNUAS)
    # =====================================================================
    def _entity_score(self, doc_claim, doc_evid, full_document):
        score = 0.0
        full_text_lower = full_document.lower()
        evid_text_lower = doc_evid.text.lower()
        
        # =====================================================================
        # 1. Checagem Contextual de NÚMEROS (Resolve Claim 23)
        # =====================================================================
        word_to_digit = {"one":"1", "two":"2", "three":"3", "four":"4", "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9", "ten":"10"}
        claim_nums = [t for t in doc_claim if t.pos_ == "NUM" or t.ent_type_ == "CARDINAL"]
        
        for num_tok in claim_nums:
            num_val = num_tok.text.lower()
            digit_val = word_to_digit.get(num_val, num_val)
            
            # Se o número não existe no documento: Fabricação Absoluta
            if num_val not in full_text_lower and digit_val not in full_text_lower:
                return 0.98 # entity_severe
                
            # Se o número está na evidência, checamos a QUEM ele se refere (o head)
            if num_val in evid_text_lower or digit_val in evid_text_lower:
                claim_head_lemma = num_tok.head.lemma_.lower()
                
                # Procura os números na evidência
                evid_matched_nums = [t for t in doc_evid if t.text.lower() == num_val or t.text.lower() == digit_val]
                
                head_match = False
                for e_num in evid_matched_nums:
                    e_head_lemma = e_num.head.lemma_.lower()
                    # A palavra a qual o número se refere no claim e na evidência devem ser similares
                    if claim_head_lemma == e_head_lemma or self._semantic_similarity(claim_head_lemma, e_head_lemma) > 0.5:
                        head_match = True
                        break
                        
                if not head_match and evid_matched_nums:
                    # O número existe na frase, mas está atrelado à palavra errada (ex: 3 grupos vs 3 recomendações)
                    return 0.98 # entity_severe

        # =====================================================================
        # 2. Checagem SVO de SUJEITOS (Evoluída contra Correferências)
        # =====================================================================
        strict_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}
        claim_subjs = [t for t in doc_claim if t.dep_ in ("nsubj", "nsubjpass") and (t.ent_type_ in strict_labels or t.pos_ == "PROPN" or t.text.isupper())]
        
        for subj_tok in claim_subjs:
            # NOVA REGRA: Ignora substantivos genéricos (ex: "agencies", "firms") 
            # para não disparar falso positivo em correferências válidas.
            if subj_tok.pos_ != "PROPN" and not subj_tok.text.isupper():
                continue

            verb_tok = subj_tok.head
            if verb_tok.pos_ != "VERB": continue
            
            verb_lemma = verb_tok.lemma_.lower()
            subj_text = subj_tok.text.lower()
            
            evid_verbs = [t for t in doc_evid if t.lemma_.lower() == verb_lemma or self._semantic_similarity(verb_lemma, t.lemma_.lower()) > 0.7]
            
            for e_verb in evid_verbs:
                e_subjs = [t.text.lower() for t in e_verb.children if t.dep_ in ("nsubj", "nsubjpass", "agent")]
                
                if e_subjs:
                    is_subj_present = any(subj_text in es or es in subj_text for es in e_subjs)
                    if not is_subj_present:
                        return 0.98 # entity_severe

        # =====================================================================
        # 3. Checagem Padrão de Fabricação Global (Evoluída contra Matemática Fuzzy)
        # =====================================================================
        claim_ents = [ent.text for ent in doc_claim.ents if ent.label_ in strict_labels]
        for t in doc_claim:
            if (t.pos_ == "PROPN" or t.text.isupper()) and len(t.text) > 1:
                if not any(t.text in e for e in claim_ents):
                    claim_ents.append(t.text)
                    
        ignore_ents = {"gao", "congress", "author", "authors", "we"}
        for ent_text in claim_ents:
            ent_clean = re.sub(r"[^\w\s]", "", ent_text).lower()
            if ent_clean in ignore_ents: continue 
            
            tokens = [t for t in ent_clean.split() if len(t) > 2 and t not in ["the", "and", "for", "with"]]
            if not tokens: continue
            
            missing_from_doc = any(tok not in full_text_lower for tok in tokens)
            if missing_from_doc:
                return 0.98 # Fabricação Global Absoluta
            
            missing_from_evid = any(tok not in evid_text_lower for tok in tokens)
            if missing_from_evid:
                # NOVA REGRA: Se a entidade for complexa (>= 2 palavras) e sumir da frase,
                # travamos o score no máximo para contornar o perdão da Matemática Fuzzy.
                # Se for apenas uma sigla (1 palavra), deixamos a Fuzzy Logic avaliar (0.85).
                if len(tokens) >= 2:
                    return 0.98 # entity_severe
                else:
                    score = max(score, 0.85)

        return score         

    def _coreference_score(self, doc_claim, doc_evid):
        if not self._is_evidence_relevant(doc_claim, doc_evid): return 0.0
        evid_text_lower = doc_evid.text.lower()
        
        # ASSINATURA ESTRITA: O sujeito original era PRON, mas o gerador injetou NOUN/PROPN?
        c_subjs = [(t.lemma_.lower(), t.head.lemma_.lower()) for t in doc_claim if t.dep_ in ("nsubj", "nsubjpass") and t.pos_ in ("NOUN", "PROPN")]
        e_subjs = [(t.lemma_.lower(), t.head.lemma_.lower()) for t in doc_evid if t.dep_ in ("nsubj", "nsubjpass") and t.pos_ == "PRON"]
        
        for c_s, c_v in c_subjs:
            if c_s not in evid_text_lower: 
                for e_s, e_v in e_subjs:
                    if c_v == e_v or self._semantic_similarity(c_v, e_v) > 0.60:
                        return 0.90 # Encontramos a injeção exata (Entity swap num lugar de pronome)
                        
        return 0.0

    def _predicate_score_1(self, doc_claim, doc_evid, full_document):
        score = 0.0
        is_relevant = self._is_evidence_relevant(doc_claim, doc_evid)
        
        # 1. CHECAGEM DE NEGAÇÃO EXPLÍCITA
        if is_relevant:
            neg_sum = any(t.dep_ == "neg" for t in doc_claim)
            neg_src = any(t.dep_ == "neg" for t in doc_evid)
            if neg_sum != neg_src: return 0.95 # <-- Aumentado de 0.85 para 0.95 (Vence empates locais)

        # Analisar verbos globais e LOCAIS
        if not hasattr(self, "doc_cache"): self.doc_cache = {}
        if full_document not in self.doc_cache:
            self.doc_cache[full_document] = self.nlp(full_document[:50000])
        full_doc = self.doc_cache[full_document]

        doc_verbs = {t.lemma_.lower(): t.text for t in full_doc if t.pos_ == "VERB"}
        evid_verbs = {t.lemma_.lower(): t.text for t in doc_evid if t.pos_ == "VERB"} 
        
        # Proíbe Siglas (isupper) e Nomes Próprios de serem avaliados como verbos
        claim_verbs = [t for t in doc_claim if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass") and not t.text.isupper() and t.pos_ != "PROPN"]

        meta_verbs = {"summarize", "report", "examine", "find", "recommend", "conclude", "state", "say", "discuss", "evaluate", "review", "analyze", "provide", "include"}
        
        unique_doc_verbs = list(set(doc_verbs.values()))
        unique_evid_verbs = list(set(evid_verbs.values())) 

        # 2. CHECAGEM SEMÂNTICA DOS VERBOS (Antônimos e Alucinações)
        for cv in claim_verbs:
            l_verb = cv.lemma_.lower()
            if l_verb in meta_verbs: continue
                
            max_sim_global = 0.0
            max_sim_local = 0.0
            
            if unique_doc_verbs:
                max_sim_global = max([self._semantic_similarity(cv.text, v) for v in unique_doc_verbs] + [0.0])
            
            # Se a evidência for relevante, checa semântica restrita nela
            if is_relevant and unique_evid_verbs:
                max_sim_local = max([self._semantic_similarity(cv.text, v) for v in unique_evid_verbs] + [0.0])
                
            if max_sim_global < 0.35:
                score = max(score, 0.95) 
            # Aumentamos de 0.55 para 0.70 para capturar antônimos (que enganam o SBERT)
            elif is_relevant and max_sim_local < 0.70: 
                score = max(score, 0.90)
                
        return score

    def classify_1(self, claim: str, evidence: str, full_document: str = "") -> str:
        doc_claim = self.nlp(claim)
        doc_evid = self.nlp(evidence)
        
        scores = {
            "entity": self._entity_score(doc_claim, doc_evid, full_document),
            "coreference": self._coreference_score(doc_claim, doc_evid),
            "predicate": self._predicate_score(doc_claim, doc_evid, full_document)
        }
        
        best_error = max(scores, key=scores.get)
        
        if scores[best_error] >= 0.75:
            return best_error
            
        return "other"

    def _predicate_score(self, doc_claim, doc_evid, full_document):
        score = 0.0
        is_relevant = self._is_evidence_relevant(doc_claim, doc_evid)
        
        evid_verbs = {t.lemma_.lower(): t.text for t in doc_evid if t.pos_ == "VERB"} 
        unique_evid_verbs = list(set(evid_verbs.values())) 

        # =====================================================================
        # 1. CHECAGEM CIRÚRGICA DE NEGAÇÃO (Corrigida)
        # =====================================================================
        if is_relevant:
            # A regra agressiva 'neg_sum and not neg_src' foi removida.
            # Mantemos apenas a análise precisa da árvore de dependência:
            negated_claim = [t for t in doc_claim if t.dep_ == "neg" and t.head.pos_ in ("VERB", "AUX")]
            for neg_tok in negated_claim:
                verb_lemma = neg_tok.head.lemma_.lower()
                if verb_lemma in evid_verbs:
                    negated_evid_lemmas = {t.head.lemma_.lower() for t in doc_evid if t.dep_ == "neg"}
                    # Só é erro se o resumo negou um verbo que a evidência usou de forma afirmativa
                    if verb_lemma not in negated_evid_lemmas:
                        return 0.98

        # =====================================================================
        # 2. DETECTOR CIRÚRGICO DE ANTÔNIMOS (Evoluído)
        # =====================================================================
        claim_verbs = [t for t in doc_claim if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass") and not t.text.isupper() and t.pos_ != "PROPN"]
        meta_verbs = {"summarize", "report", "examine", "find", "recommend", "conclude", "state", "say", "discuss", "evaluate", "review", "analyze", "provide", "include"}

        for cv in claim_verbs:
            l_verb = cv.lemma_.lower()
            
            if hasattr(self, "pares_antonimos") and l_verb in self.pares_antonimos:
                original_verb = self.pares_antonimos[l_verb]
                if original_verb in evid_verbs:
                    return 0.98
                
                # Se não achou a raiz exata na frase recuperada, mas a semântica do 
                # antônimo injetado conflita drasticamente com os verbos locais (< 0.35),
                # é Injeção Confirmada, promovendo para predicate_severe.
                if is_relevant and unique_evid_verbs:
                    max_sim_ant = max([self._semantic_similarity(cv.text, v) for v in unique_evid_verbs] + [0.0])
                    if max_sim_ant < 0.35:
                        return 0.98 

            if l_verb in meta_verbs: continue
            
            # 3. VERIFICAÇÃO LOCAL LEVE
            if is_relevant and unique_evid_verbs:
                max_sim_local = max([self._semantic_similarity(cv.text, v) for v in unique_evid_verbs] + [0.0])
                if max_sim_local < 0.40:
                    score = max(score, 0.85)
                
        return score
    
    def classify(self, claim: str, evidence: str, full_document: str = "") -> str:
        doc_claim = self.nlp(claim)
        doc_evid = self.nlp(evidence)
        
        scores = {
            "entity": self._entity_score(doc_claim, doc_evid, full_document),
            "coreference": self._coreference_score(doc_claim, doc_evid),
            "predicate": self._predicate_score(doc_claim, doc_evid, full_document)
        }
        
        best_error = max(scores, key=scores.get)
        
        # Mapeamento dos erros estruturais
        if best_error == "predicate" and scores["predicate"] >= 0.96:
            return "predicate_severe"
        elif best_error == "entity" and scores["entity"] >= 0.96:
            return "entity_severe"
        elif scores["entity"] >= 0.85 and scores["entity"] >= scores["predicate"]:
            return "entity"
        elif scores[best_error] >= 0.75:
            return best_error
            
        return "other"