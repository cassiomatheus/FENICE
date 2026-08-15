import re
import spacy
import torch
from sentence_transformers import SentenceTransformer, util

try:
    from metric.utils.utils import nlp, sbert_model
except ImportError:
    nlp = spacy.load("en_core_web_md")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

class TaxonomyClassifier:
    def __init__(self):
        self.nlp = nlp
        self.sbert = sbert_model

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
        
        # Calculamos a interseção para medir a confiança na frase recuperada
        c_lemmas = {t.lemma_.lower() for t in doc_claim if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "VERB")}
        e_lemmas = {t.lemma_.lower() for t in doc_evid if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "VERB")}
        overlap = len(c_lemmas.intersection(e_lemmas))
        is_relevant = overlap >= 1

        # Dicionário universal para evitar Falsos Positivos com números
        word_to_digit = {"one":"1", "two":"2", "three":"3", "four":"4", "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9", "ten":"10"}

        # =====================================================================
        # 1. CHECAGEM INTELIGENTE DE NÚMEROS
        # =====================================================================
        sum_nums = {t.text.lower() for t in doc_claim if t.pos_ == "NUM"}
        if sum_nums:
            sum_nums_digits = {word_to_digit.get(n, n) for n in sum_nums}
            
            # Se o número não existe nem como palavra nem como dígito no documento inteiro: Fabricação Global.
            if any(n not in full_text_lower for n in sum_nums_digits):
                score = max(score, 0.95)
                
            # Só checamos manipulação local se a recuperação for ALTAMENTE confiável (overlap >= 2)
            elif overlap >= 2: 
                src_nums = {t.text.lower() for t in doc_evid if t.pos_ == "NUM"}
                # Só é "troca de número" se a evidência tiver OUTRO número no lugar
                if src_nums:
                    src_nums_digits = {word_to_digit.get(n, n) for n in src_nums}
                    if not sum_nums_digits.intersection(src_nums_digits):
                        score = max(score, 0.85)

        # =====================================================================
        # 2. CHECAGEM ESTRITA DE ENTIDADES
        # =====================================================================
        strict_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "LAW", "DATE", "TIME"}
        claim_ents = [ent for ent in doc_claim.ents if ent.label_ in strict_labels]
        
        ignore_ents = {"gao", "congress", "author", "authors", "we"}

        for ent in claim_ents:
            ent_clean = re.sub(r"[^\w\s]", "", ent.text).lower()
            if ent_clean in ignore_ents: continue 

            tokens = [t for t in ent_clean.split() if len(t) > 2 and t not in ["the", "and", "for"]]
            if not tokens: continue
            
            missing_from_doc = any(tok not in full_text_lower for tok in tokens)
            
            if missing_from_doc:
                if not hasattr(self, "doc_ents_cache"): self.doc_ents_cache = {}
                if full_document not in self.doc_ents_cache:
                    doc = self.nlp(full_document[:50000])
                    self.doc_ents_cache[full_document] = list(set([e.text for e in doc.ents]))
                
                doc_ents = self.doc_ents_cache[full_document]
                max_sim_doc = max([self._semantic_similarity(ent.text, e) for e in doc_ents] + [0.0])
                
                # Entidade globalmente fabricada
                if max_sim_doc < 0.45:
                    score = max(score, 0.95) 

            # O ESCUDO DE RECUPERAÇÃO: Só acusamos uma Troca de Entidade Local se tivermos 
            # extrema certeza de que o Retriever trouxe a sentença exata (overlap >= 3).
            # Suavizamos o limite de overlap de 3 para 2 para 
            # não bloquear a checagem em frases abstrativas.
            elif overlap >= 2 and any(tok not in evid_text_lower for tok in tokens):
                evid_ents_same_label = [e.text for e in doc_evid.ents if e.label_ == ent.label_]
                
                if evid_ents_same_label:
                    max_sim_local = max([self._semantic_similarity(ent.text, e) for e in evid_ents_same_label] + [0.0])
                    if max_sim_local < 0.45:
                        score = max(score, 0.85)

        # =====================================================================
        # 3. CHECAGEM DE DUPLICAÇÃO / SWAP (Resolve "FDA and FDA" em Janelas)
        # =====================================================================
        if is_relevant and overlap >= 2:
            from collections import Counter
            
            claim_ent_texts = [re.sub(r"[^\w\s]", "", ent.text).lower() for ent in claim_ents if re.sub(r"[^\w\s]", "", ent.text).lower() not in ignore_ents]
            claim_ent_counts = Counter(claim_ent_texts)
            
            for ent_text, count in claim_ent_counts.items():
                if count > 1:
                    # Se o resumo tem entidades repetidas (ex: 2x FDA), verificamos se 
                    # ALGUMA frase individual da evidência recuperada justifica isso.
                    max_in_single_sentence = 0
                    for sent in doc_evid.sents:
                        sent_ent_texts = [re.sub(r"[^\w\s]", "", e.text).lower() for e in sent.ents]
                        sent_count = Counter(sent_ent_texts).get(ent_text, 0)
                        if sent_count > max_in_single_sentence:
                            max_in_single_sentence = sent_count
                    
                    # Se nenhuma frase única da fonte repete a entidade tanto quanto o resumo, é Fabricação.
                    if count > max_in_single_sentence:
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

    def _predicate_score(self, doc_claim, doc_evid, full_document):
        score = 0.0
        is_relevant = self._is_evidence_relevant(doc_claim, doc_evid)
        
        if is_relevant:
            neg_sum = any(t.dep_ == "neg" for t in doc_claim)
            neg_src = any(t.dep_ == "neg" for t in doc_evid)
            if neg_sum != neg_src: return 0.85

        # Analisar verbos globais e LOCAIS
        if not hasattr(self, "doc_cache"): self.doc_cache = {}
        if full_document not in self.doc_cache:
            self.doc_cache[full_document] = self.nlp(full_document[:50000])
        full_doc = self.doc_cache[full_document]

        doc_verbs = {t.lemma_.lower(): t.text for t in full_doc if t.pos_ == "VERB"}
        evid_verbs = {t.lemma_.lower(): t.text for t in doc_evid if t.pos_ == "VERB"} # NOVA LINHA
        
        claim_verbs = [t for t in doc_claim if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass")]
        meta_verbs = {"summarize", "report", "examine", "find", "recommend", "conclude", "state", "say", "discuss", "evaluate", "review", "analyze", "provide", "include"}
        
        unique_doc_verbs = list(set(doc_verbs.values()))
        unique_evid_verbs = list(set(evid_verbs.values())) # NOVA LINHA

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
                score = max(score, 0.90) # Subimos o peso para ganhar de entidade (0.85)
            elif is_relevant and max_sim_local < 0.55: # Verbo não bate com a evidência recuperada
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
        
        if scores[best_error] >= 0.75:
            return best_error
            
        return "other"