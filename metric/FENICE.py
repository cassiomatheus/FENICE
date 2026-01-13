from typing import List, Dict, Optional
import numpy as np
import torch
from tqdm import tqdm
from metric.claim_extractor.claim_extractor import ClaimExtractor
from metric.coreference_resolution.coreference_resolution import CoreferenceResolution
from metric.nli.nli_aligner import NLIAligner
from metric.utils.utils import split_into_paragraphs, split_into_sentences_batched
from metric.utils.utils import nlp  # Importa o modelo spaCy carregado
import spacy

class FENICE:
    def __init__(
        self,
        use_coref: bool = False,
        num_sent_per_paragraph: int = 5,
        sliding_paragraphs=True,
        sliding_stride: int = 1,
        doc_level_nli=True,
        paragraph_level_nli=True,
        claim_extractor_batch_size: int = 128,
        coreference_batch_size: int = 1,
        nli_batch_size: int = 128,
        nli_max_length: int = 256,
    ) -> None:
        self.num_sent_per_paragraph = num_sent_per_paragraph
        self.claim_extractor_batch_size = claim_extractor_batch_size
        self.coreference_batch_size = coreference_batch_size
        self.nli_batch_size = nli_batch_size
        self.sliding_paragraphs = sliding_paragraphs
        self.sliding_stride = sliding_stride
        self.sentences_cache = {}
        self.coref_clusters_cache = {}
        self.claims_cache = {}
        self.alignments_cache = {}
        self.use_coref = use_coref
        self.doc_level_nli = doc_level_nli
        self.paragraph_level_nli = paragraph_level_nli
        self.coref_model = (
            CoreferenceResolution(load_model=False) if use_coref else None
        )
        self.nli_max_length = nli_max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def classify_error(self, probs, summary_claim=None, source_text=None):
        """
        Classifica o erro usando probabilidades NLI + Análise Linguística
        """
        if not probs:
            return "Desconhecido"
            
        # probs: [Entailment, Contradiction, Neutral]
        ent, contr, neut = probs
        max_idx = np.argmax(probs)
        
        # 1. Factual (Supported)
        if max_idx == 0: 
            return "Factual"
            
        # 2. Extrínseco (Neutral - Alucinação de info nova)
        elif max_idx == 2:
            return "Extrínseco"
            
        # 3. Contradição (Analisa o subtipo)
        elif max_idx == 1:
            if summary_claim and source_text:
                return self.analyze_contradiction_type(summary_claim, source_text)
            else:
                return "Intrínseco" # Fallback se não tiver texto

    def _score(self, sample_id: int, document: str, summary: str):
      # ======== Preparação ========
      doc_id = self.get_id(sample_id, document)
      sentences_offsets = self.sentences_cache[doc_id]
      sentences = [s[0] for s in sentences_offsets]
      offsets = [(s[1], s[2]) for s in sentences_offsets]
  
      paragraphs = split_into_paragraphs(
          sentences,
          self.num_sent_per_paragraph,
          sliding_paragraphs=self.sliding_paragraphs,
          sliding_stride=self.sliding_stride,
      )
  
      summary_id = self.get_id(sample_id, summary)
      summary_claims = self.claims_cache.get(summary_id, [summary])
  
      alignments = []
  
      # ======== Contadores CORRETOS ========
      status_counts = {
          "supported": 0,
          "contradicted": 0,
          "not_supported": 0,
      }
  
      error_type_counts = {
          "entity": 0,
          "predicate": 0,
          "coreference": 0,
          "other": 0,
      }
  
      # ======== Loop por claim ========
      for claim_id, claim in enumerate(summary_claims):
      
          # --- Sentence-level alignment ---
          sentence_level_alignment = self.get_alignment(
              premises=sentences,
              hypothesis=claim,
              sample_id=sample_id,
              hypothesis_id=claim_id,
          )
  
          # --- Coreference alignment ---
          coref_alignment = None
          if self.use_coref:
              coref_clusters = self.coref_clusters_cache[doc_id]
              coref_premises = self.coref_model.get_coref_versions(
                  sentence=sentence_level_alignment["source_passage"],
                  text=document,
                  sentences=sentences,
                  offsets=offsets,
                  clusters=coref_clusters,
              )
              if coref_premises:
                  coref_alignment = self.get_alignment(
                      premises=coref_premises,
                      hypothesis=claim,
                      sample_id=sample_id,
                      hypothesis_id=claim_id,
                      alignment_prefix="coref",
                  )
  
          # --- Paragraph-level alignment ---
          paragraph_level_alignment = None
          if len(paragraphs) > 1 and self.paragraph_level_nli:
              paragraph_level_alignment = self.get_alignment(
                  premises=paragraphs,
                  hypothesis=claim,
                  sample_id=sample_id,
                  hypothesis_id=claim_id,
                  alignment_prefix="par",
              )
  
          # --- Document-level alignment ---
          doc_level_alignment = None
          if self.doc_level_nli:
              doc_level_alignment = self.get_alignment(
                  premises=[document],
                  hypothesis=claim,
                  sample_id=sample_id,
                  hypothesis_id=claim_id,
                  alignment_prefix="doc",
              )
              doc_level_alignment["source_passage"] = "DOCUMENT"
  
          # --- Coleta de todos os alinhamentos ---
          sample_alignments = [
              sentence_level_alignment,
              coref_alignment,
              paragraph_level_alignment,
              doc_level_alignment,
          ]
  
          # --- Melhor alinhamento (para score) ---
          alignment = self.max_alignment(sample_alignments)
  
          # ======== CLASSIFICAÇÃO FACTUAL CORRETA ========
          statuses = []
  
          for al in sample_alignments:
              if al and "probs" in al:
                  statuses.append(self.classify_factual_status(al["probs"]))
  
          # decisão final do status
          if statuses and all(s == "not_supported" for s in statuses):
              final_status = "not_supported"
          else:
              priority = {"supported": 2, "contradicted": 1, "not_supported": 0}
              final_status = max(statuses, key=lambda s: priority[s])
  
          alignment["factual_status"] = final_status
          status_counts[final_status] += 1
  
          # ======== CLASSIFICAÇÃO DO TIPO DE ERRO ========
          if final_status == "contradicted":
              error_type = self.classify_error_type(
                  alignment["summary_claim"],
                  alignment["source_passage"],
              )
              alignment["error_type"] = error_type
              error_type_counts[error_type] += 1
          else:
              alignment["error_type"] = None
  
          alignments.append(alignment)
  
      # ======== SCORE GLOBAL ========
      score = float(np.mean([al["score"] for al in alignments])) if alignments else 0.0
  
      # ======== PROPORÇÕES CORRETAS ========
      total_claims = max(len(summary_claims), 1)
      total_contradicted = max(status_counts["contradicted"], 1)
  
      status_ratios = {
          k: v / total_claims
          for k, v in status_counts.items()
      }
  
      error_type_ratios = {
          k: v / total_contradicted
          for k, v in error_type_counts.items()
      }
  
      # ======== RETORNO FINAL ========
      return {
          "score": score,
          "status_counts": status_counts,
          "status_ratios": status_ratios,
          "error_type_counts": error_type_counts,
          "error_type_ratios": error_type_ratios,
          "alignments": alignments,
      }
  
  
  



    def _score_mod(self, sample_id: int, document: str, summary: str):
        doc_id = self.get_id(sample_id, document)
        sentences_offsets = self.sentences_cache[doc_id]
        sentences = [s[0] for s in sentences_offsets]
        offsets = [(s[1], s[2]) for s in sentences_offsets]
        # paragraphs are sliding windows of {num_sent_per_paragraph} sentences
        paragraphs = split_into_paragraphs(
            sentences,
            self.num_sent_per_paragraph,
            sliding_paragraphs=self.sliding_paragraphs,
            sliding_stride=self.sliding_stride,
        )
        # claim extraction
        summary_id = self.get_id(sample_id, summary)
        summary_claims = self.claims_cache.get(summary_id, [summary])
        alignments = []
        for claim_id, claim in enumerate(summary_claims):
            # sentence-level alignment
            sentence_level_alignment = self.get_alignment(
                premises=sentences,
                hypothesis=claim,
                sample_id=sample_id,
                hypothesis_id=claim_id,
            )
            coref_alignment = None
            if self.use_coref:
                coref_clusters = self.coref_clusters_cache[doc_id]
                # modified versions of {aligned_sentence} obtained through coreference resolution
                coref_premises = self.coref_model.get_coref_versions(
                    sentence=sentence_level_alignment["source_passage"],
                    text=document,
                    sentences=sentences,
                    offsets=offsets,
                    clusters=coref_clusters,
                )
                if coref_premises:
                    coref_alignment = self.get_alignment(
                        sample_id=sample_id,
                        hypothesis_id=claim_id,
                        hypothesis=claim,
                        premises=coref_premises,
                        alignment_prefix="coref",
                    )
            paragraph_level_alignment = None
            if len(paragraphs) > 1 and self.paragraph_level_nli:
                paragraph_level_alignment = self.get_alignment(
                    premises=paragraphs,
                    hypothesis=claim,
                    sample_id=sample_id,
                    hypothesis_id=claim_id,
                    alignment_prefix="par",
                )
            doc_level_alignment = None
            if self.doc_level_nli and len(paragraphs) > 1:
                doc_level_alignment = self.get_alignment(
                    hypothesis=claim,
                    premises=[document],
                    sample_id=sample_id,
                    hypothesis_id=claim_id,
                    alignment_prefix="doc",
                )
                doc_level_alignment["source_passage"] = "DOCUMENT"
            sample_alignments = [
                sentence_level_alignment,
                coref_alignment,
                paragraph_level_alignment,
                doc_level_alignment,
            ]
            alignment = self.max_alignment(sample_alignments)
            alignments.append(alignment)
        score = np.mean([al["score"] for al in alignments])
        return {"score": score, "alignments": alignments}

    def max_alignment(self, sample_alignments):
        sample_alignments = [s for s in sample_alignments if s is not None]
        alignment = max(sample_alignments, key=lambda x: x["score"])
        return alignment

    def get_alignment(
        self,
        premises,
        hypothesis,
        sample_id,
        hypothesis_id,
        alignment_prefix: Optional[str] = None,
    ):
        alignments_ids = []
        for prem_id in range(len(premises)):
            pair_id = self.get_alignment_id(
                sample_id=sample_id,
                premise_id=prem_id,
                hypothesis_id=hypothesis_id,
                premise=premises[prem_id],
                alignment_prefix=alignment_prefix,
            )
            alignments_ids.append(pair_id)
        loaded_alignment = self.load_alignment(
            alignments_ids=alignments_ids, premises=premises, hypothesis=hypothesis
        )
        if loaded_alignment is None:
            all_pairs = [(x, hypothesis) for x in premises]
            self.cache_alignment(alignments_ids, all_pairs)
            alignment, _ = self.load_alignment(
                alignments_ids=alignments_ids, premises=premises, hypothesis=hypothesis
            )
        else:
            alignment, _ = loaded_alignment
        return alignment

    def score_batch(self, batch: List[Dict[str, str]]) -> List[Dict]:
        documents = [el["document"] for el in batch]
        summaries = [el["summary"] for el in batch]
        self.cache(documents, summaries)
        predictions = []
        for sample_id, (doc, summary) in tqdm(
            enumerate(zip(documents, summaries)),
            total=len(documents),
            desc="Computing FENICE...",
        ):
            predictions.append(self._score(sample_id, doc, summary))

        del self.nli_aligner
        torch.cuda.empty_cache()

        return predictions

    def cache(self, documents, summaries):
        self.cache_sentences(documents)
        if self.use_coref:
            self.cache_coref(documents)
        self.cache_claims(summaries)
        self.cache_alignments(documents, summaries)

    def cache_sentences(self, documents):
        all_sentences = split_into_sentences_batched(
            documents, batch_size=512, return_offsets=True
        )
        for i, sentences in enumerate(all_sentences):
            id = self.get_id(i, documents[i])
            self.sentences_cache[id] = sentences

    def get_docs_to_process(self, documents, cache):
        ids = [doc_id for doc_id in range(len(documents))]
        ids_to_process = [
            id for id in ids if self.get_id(id, documents[id]) not in cache
        ]
        doc_to_process = [documents[i] for i in ids_to_process]
        return doc_to_process, ids_to_process

    def get_id(self, sample_id: int, text: str, k_chars: int = 100):
        id = f"{sample_id}{text[:k_chars]}"
        return id

    # cache claim extraction outputs
    def cache_claims(self, summaries):
        claim_extractor = ClaimExtractor(
            batch_size=self.claim_extractor_batch_size, device=self.device
        )
        claims_predictions = claim_extractor.process_batch(summaries)
        for summ_id, claims in enumerate(claims_predictions):
            id = self.get_id(summ_id, summaries[summ_id])
            self.claims_cache[id] = claims
        del claim_extractor
        torch.cuda.empty_cache()

    # cache coreference resolution outputs
    def cache_coref(self, documents):
        coref_model = CoreferenceResolution(
            batch_size=self.coreference_batch_size, device=self.device
        )
        all_clusters = coref_model.get_clusters_batch(documents)
        for doc_id, clusters in enumerate(all_clusters):
            id = self.get_id(doc_id, documents[doc_id])
            self.coref_clusters_cache[id] = clusters
        del coref_model.model
        torch.cuda.empty_cache()

    def cache_alignments(self, documents: List[str], summaries: List[str]):
        alignments_ids, all_pairs = [], []
        alignment_ids_doc, all_pairs_doc = [], []
        self.nli_aligner = NLIAligner(
            batch_size=self.nli_batch_size,
            device=self.device,
            max_length=self.nli_max_length,
        )
        for sample_id in range(len(summaries)):
            summary_id = self.get_id(sample_id, summaries[sample_id])
            document_id = self.get_id(sample_id, documents[sample_id])
            claims = self.claims_cache[summary_id]
            sentences = [s[0] for s in self.sentences_cache[document_id]]
            paragraphs = split_into_paragraphs(
                sentences,
                self.num_sent_per_paragraph,
                sliding_paragraphs=self.sliding_paragraphs,
                sliding_stride=self.sliding_stride,
            )
            alignments_ids, all_pairs = self.compute_nli_pairs(
                alignments_ids, all_pairs, claims, sample_id, sentences
            )
            if self.paragraph_level_nli:
                alignments_ids, all_pairs = self.compute_nli_pairs(
                    alignments_ids,
                    all_pairs,
                    claims,
                    sample_id,
                    paragraphs,
                    prefix="par",
                )
            if self.doc_level_nli:
                alignment_ids_doc, all_pairs_doc = self.compute_nli_pairs(
                    alignment_ids_doc,
                    all_pairs_doc,
                    claims,
                    sample_id,
                    [documents[sample_id]],
                    prefix="doc",
                )
        self.cache_alignment(alignments_ids, all_pairs)
        self.nli_aligner.batch_size = 1
        self.nli_aligner.max_length = 4096
        self.cache_alignment(alignment_ids_doc, all_pairs_doc)
        self.nli_aligner.batch_size = self.nli_batch_size
        self.nli_max_length = self.nli_max_length

    def compute_nli_pairs(
        self, alignments_ids, all_pairs, claims, sample_id, premises, prefix=None
    ):
        for premise_id, premise in enumerate(premises):
            for claim_id, claim in enumerate(claims):
                alignment_id = self.get_alignment_id(
                    sample_id, premise_id, claim_id, premise, prefix
                )
                alignments_ids.append(alignment_id)
                all_pairs.append((premise, claim))
        return alignments_ids, all_pairs

    def get_alignment_id(
        self,
        sample_id: int,
        premise_id: int,
        hypothesis_id: int,
        premise: str,
        alignment_prefix: Optional[str] = None,
        k_chars: int = 100,
    ):

        id = f"{sample_id}-{premise_id}-{hypothesis_id}-{premise[:k_chars]}"
        if alignment_prefix is not None:
            id = f"{alignment_prefix}{id}"
        return id

    def cache_alignment(
        self, alignments_ids, all_pairs, disable_prog_bar: bool = False
    ):
        probabilities = self.nli_aligner.process_batch(
            all_pairs, disable_prog_bar=disable_prog_bar
        )
        for i, id in enumerate(alignments_ids):
            ent, contr, neut = (
                probabilities[0][i],
                probabilities[1][i],
                probabilities[2][i],
            )
            self.alignments_cache[id] = (ent, contr, neut)

    def load_alignment_mod(
        self, alignments_ids: List[str], premises: List[str], hypothesis: str
    ):
        if all([k in self.alignments_cache for k in alignments_ids]):
            scores = [self.alignments_cache[key] for key in alignments_ids]
            alignment = None
            max_score = -np.inf
            for i, (ent, contr, neut) in enumerate(scores):
                ent, contr, neut = ent.item(), contr.item(), neut.item()
                align_score = ent - neut
                if align_score > max_score:
                    max_score = align_score
                    alignment = (premises[i], [ent, contr, neut])
            return (
                {
                    "score": max_score,
                    "summary_claim": hypothesis,
                    "source_passage": alignment[0],
                },
                scores,
            )
        else:
            return None

    def load_alignment(
        self, alignments_ids: List[str], premises: List[str], hypothesis: str
    ):
        if all([k in self.alignments_cache for k in alignments_ids]):
            scores = [self.alignments_cache[key] for key in alignments_ids]
            alignment = None
            max_score = -np.inf
            best_probs = None # Nova variável para guardar as probs

            for i, (ent, contr, neut) in enumerate(scores):
                ent, contr, neut = ent.item(), contr.item(), neut.item()
                align_score = ent - neut
                if align_score > max_score:
                    max_score = align_score
                    # Guardamos as probabilidades brutas do vencedor
                    best_probs = [ent, contr, neut] 
                    alignment = (premises[i], [ent, contr, neut])
            
            return (
                {
                    "score": max_score,
                    "probs": best_probs, # Adicionamos ao dicionário de retorno
                    "summary_claim": hypothesis,
                    "source_passage": alignment[0],
                },
                scores,
            )
        else:
            return None

    def analyze_contradiction_type(self, summary_claim: str, source_text: str) -> str:
        """
        Refina o tipo de erro quando o NLI detecta uma contradição.
        Ordem de prioridade: Entidade > Correferência > Predicado > Intrínseco (Adjetivos/Outros)
        """
        # Processa os textos com spaCy
        doc_sum = nlp(summary_claim)
        doc_src = nlp(source_text)

        # 1. ERRO DE ENTIDADE (Números, Nomes Próprios, Datas, Dinheiro)
        # Extrai entidades nomeadas e números soltos
        ents_sum = {(e.text.lower(), e.label_) for e in doc_sum.ents}
        nums_sum = {t.text for t in doc_sum if t.pos_ == "NUM"}
        
        # Verifica se as entidades do resumo estão presentes no texto fonte
        # (Lógica simplificada: se o resumo tem um número/nome que não está na fonte, é erro de entidade)
        src_text_lower = source_text.lower()
        for text, label in ents_sum:
            if text not in src_text_lower:
                return "Entidade"
        
        for num in nums_sum:
            if num not in src_text_lower:
                return "Entidade"

        # 2. ERRO DE CORREFERÊNCIA (Pronomes e Sujeitos)
        # Verifica se há pronomes no resumo que podem estar mal atribuídos
        pronouns_sum = [t.text.lower() for t in doc_sum if t.pos_ == "PRON"]
        if pronouns_sum:
            # Se tem pronome e deu contradição, há alta chance de ser correferência errada
            # (Heurística: o NLI flagrou conflito e o foco da frase é um pronome)
            return "Correferência"
            
        # Comparação de Sujeitos (nsubj)
        subjs_sum = [t.lemma_ for t in doc_sum if t.dep_ == "nsubj"]
        subjs_src = [t.lemma_ for t in doc_src if t.dep_ == "nsubj"]
        # Se os sujeitos são substantivos (não pronomes) e são diferentes
        if subjs_sum and subjs_src:
            if not set(subjs_sum).intersection(set(subjs_src)):
                 return "Correferência"

        # 3. ERRO DE PREDICADO (Verbos e Negação)
        # Compara os verbos principais (roots)
        verbs_sum = {t.lemma_ for t in doc_sum if t.pos_ == "VERB"}
        verbs_src = {t.lemma_ for t in doc_src if t.pos_ == "VERB"}
        
        # Se não há interseção entre os verbos principais, assumimos troca de ação
        if verbs_sum and not verbs_sum.intersection(verbs_src):
            return "Predicado"
            
        # Verifica partículas de negação (not, never, no)
        neg_sum = any(t.dep_ == "neg" for t in doc_sum)
        neg_src = any(t.dep_ == "neg" for t in doc_src)
        if neg_sum != neg_src:
            return "Predicado"

        # 4. INTRÍNSECO (Default)
        # Se passou por tudo (entidades batem, sujeitos batem, verbos batem),
        # sobraram Adjetivos (Azul vs Vermelho) ou Adverbios.
        return "Intrínseco"

    def classify_factual_status(self, probs, ent_th=0.7, contr_th=0.6):
      """
      Classifica o status factual de uma claim com base nas probabilidades NLI.
      Retorna: supported | contradicted | not_supported
      """
      if probs is None:
          return "not_supported"

      ent, contr, neut = probs

      if ent >= ent_th:
          return "supported"
      elif contr >= contr_th:
          return "contradicted"
      else:
          return "not_supported"
    
    def classify_error_type(self, summary_claim: str, source_text: str) -> str:
      """
      Classifica o tipo de erro factual SOMENTE se houver contradição.
      Retorna: entity | predicate | coreference | other
      """
      scores = {
          "entity": 0.0,
          "predicate": 0.0,
          "coreference": 0.0,
          "other": 0.0,
      }

      doc_sum = nlp(summary_claim)
      doc_src = nlp(source_text.lower())

      # ========= ERRO DE ENTIDADE =========
      sum_entities = {(e.text.lower(), e.label_) for e in doc_sum.ents}
      src_text = source_text.lower()

      for ent_text, ent_label in sum_entities:
          if ent_text not in src_text:
              scores["entity"] += 0.6

      sum_numbers = {t.text for t in doc_sum if t.pos_ == "NUM"}
      for num in sum_numbers:
          if num not in src_text:
              scores["entity"] += 0.4

      # ========= ERRO DE CORREFERÊNCIA =========
      pronouns = [t.text.lower() for t in doc_sum if t.pos_ == "PRON"]
      if pronouns:
          scores["coreference"] += 0.3  # evidência fraca, não decisiva

      sum_subj = {t.lemma_ for t in doc_sum if t.dep_ == "nsubj"}
      src_subj = {t.lemma_ for t in doc_src if t.dep_ == "nsubj"}
      if sum_subj and src_subj and not sum_subj.intersection(src_subj):
          scores["coreference"] += 0.5

      # ========= ERRO DE PREDICADO =========
      sum_verbs = {t.lemma_ for t in doc_sum if t.pos_ == "VERB"}
      src_verbs = {t.lemma_ for t in doc_src if t.pos_ == "VERB"}

      if sum_verbs and not sum_verbs.intersection(src_verbs):
          scores["predicate"] += 0.6

      neg_sum = any(t.dep_ == "neg" for t in doc_sum)
      neg_src = any(t.dep_ == "neg" for t in doc_src)
      if neg_sum != neg_src:
          scores["predicate"] += 0.4

      # ========= DECISÃO FINAL =========
      best_type = max(scores, key=scores.get)

      if scores[best_type] >= 0.5:
          return best_type
      else:
          return "other"

