import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Dict, Optional
import numpy as np
import torch
import gc
from tqdm import tqdm
from metric.claim_extractor.claim_extractor import ClaimExtractor
from metric.coreference_resolution.coreference_resolution import CoreferenceResolution
from metric.nli.nli_aligner import NLIAligner
from metric.utils.utils import split_into_paragraphs, split_into_sentences_batched
from metric.utils.utils import nlp
import spacy
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer, util
try:
    from metric.utils.utils import sbert_model
except ImportError:
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# IMPORTAÇÃO DO MÓDULO EXTERNO DE TAXONOMIA
from metric.taxonomy_classifier import TaxonomyClassifier




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
        doc_chunk_max_tokens: int = 400,
        doc_chunk_overlap: int = 0,
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
        self.spacy_cache = {}
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
        self.doc_chunk_max_tokens = doc_chunk_max_tokens
        self.doc_chunk_overlap = doc_chunk_overlap
        self.doc_chunks_cache = {}
        
        # INSTANCIA O CLASSIFICADOR EXTERNO
        self.taxonomy_classifier = TaxonomyClassifier()

    def _score(self, sample_id: int, document: str, summary: str, error_ground_truth: Dict = None):
      doc_id = self.get_id(sample_id, document)
      sentences_offsets = self.sentences_cache[doc_id]
      sentences = [s[0] for s in sentences_offsets]
      offsets = [(s[1], s[2]) for s in sentences_offsets]

      if self.doc_level_nli:
          doc_chunks = self.doc_chunks_cache[doc_id]
      else:
          doc_chunks = None
        
      paragraphs = split_into_paragraphs(
          sentences,
          self.num_sent_per_paragraph,
          sliding_paragraphs=self.sliding_paragraphs,
          sliding_stride=self.sliding_stride,
      )
  
      summary_id = self.get_id(sample_id, summary)
      summary_claims = self.claims_cache.get(summary_id, [summary])
  
      alignments = []
  
      status_counts = {"supported": 0, "contradicted": 0, "not_supported": 0}
      error_type_counts = {"entity": 0, "predicate": 0, "coreference": 0, "other": 0}
  
      for claim_id, claim in enumerate(summary_claims):
      
          sentence_level_alignment = self.get_alignment(
              premises=sentences, hypothesis=claim, sample_id=sample_id, hypothesis_id=claim_id,
          )
  
          coref_alignment = None
          if self.use_coref:
              coref_clusters = self.coref_clusters_cache[doc_id]
              coref_premises = self.coref_model.get_coref_versions(
                  sentence=sentence_level_alignment["source_passage"],
                  text=document, sentences=sentences, offsets=offsets, clusters=coref_clusters,
              )
              if coref_premises:
                  coref_alignment = self.get_alignment(
                      premises=coref_premises, hypothesis=claim, sample_id=sample_id,
                      hypothesis_id=claim_id, alignment_prefix="coref",
                  )
  
          paragraph_level_alignment = None
          if len(paragraphs) > 1 and self.paragraph_level_nli:
              paragraph_level_alignment = self.get_alignment(
                  premises=paragraphs, hypothesis=claim, sample_id=sample_id,
                  hypothesis_id=claim_id, alignment_prefix="par",
              )
  
          doc_level_alignment = None
          if self.doc_level_nli:
              doc_level_alignment = self.get_alignment(
                  premises=doc_chunks, hypothesis=claim, sample_id=sample_id,
                  hypothesis_id=claim_id, alignment_prefix="doc",
              )
              # REMOVIDO: A linha que mascarava o texto com "DOCUMENT_CHUNK"

          sample_alignments = [
              sentence_level_alignment, coref_alignment, paragraph_level_alignment, doc_level_alignment,
          ]
  
          best_alignment = self.max_alignment(sample_alignments)
          alignment = best_alignment.copy()
          alignment["all_alignments"] = sample_alignments

          # REMOVIDO: O bloco if que sobrescrevia a passagem de origem pela sentence_level_alignment
  
          # =====================================================================
          # O AUDITOR NEURO-SIMBÓLICO COM CONFIANÇA ASSIMÉTRICA CALIBRADA
          # =====================================================================
          melhores_probs = alignment.get("probs", [0.0, 0.0, 1.0])
          ent_prob = melhores_probs[0]
          contr_prob = melhores_probs[1]

          # Resolve o Claim 4: Aceita entailment a partir de 55% para resumos abstrativos
          final_status = self.classify_factual_status(melhores_probs, ent_th=0.55)

          error_type = self.taxonomy_classifier.classify(
                  claim=alignment["summary_claim"],
                  evidence=alignment["source_passage"],
                  full_document=document
          )

          # -------------------------------------------------------------------
          # 1. FILTRO DE META-DISCURSO GENERALISTA
          # -------------------------------------------------------------------
          claim_doc = self.get_spacy_doc(alignment["summary_claim"])
          # Adicionado "gao" como uma entidade capaz de produzir meta-discurso
          doc_nouns = {"report", "paper", "article", "document", "study", "review", "analysis", "brief", "memo", "gao", "authors", "we"}
          meta_verbs = {"describe", "examine", "summarize", "report", "evaluate", "analyze", "discuss", "provide", "present", "conclude", "focus", "aim", "investigate", "ask"}
          
          is_meta_discourse = False
          for token in claim_doc:
              if token.dep_ in ("nsubj", "nsubjpass") and token.lemma_.lower() in doc_nouns:
                  if token.head.lemma_.lower() in meta_verbs:
                      is_meta_discourse = True
                      break

          if is_meta_discourse:
              # O filtro NÃO perdoa manipulação de Entidades, 
              # NEM perdoa contradições extremas (> 90%) do NLI.
              if error_type != "entity" and contr_prob < 0.90:
                  final_status = "supported"
                  error_type = "other"
                  ent_prob = 1.0 
                  contr_prob = 0.0

          # -------------------------------------------------------------------
          # 2. TRAVA DE SEGURANÇA BIFURCADA (A Mágica da Calibração)
          # -------------------------------------------------------------------
          if error_type != "other":
              if error_type == "entity":
                  # Motores Neurais (DeBERTa) são CEGOS para números/siglas. 
                  # Se a taxonomia detectar manipulação de entidade, confiamos 100% no spaCy!
                  final_status = "contradicted"
                  if alignment["score"] > 0: 
                      alignment["score"] = -abs(alignment["score"])
              else:
                  # Para erros de Predicado (abstrações), mantemos a trava neural
                  if ent_prob >= 0.50 or contr_prob < 0.10:
                      error_type = "other"
                      # Restaura com tolerância abstrativa
                      final_status = self.classify_factual_status(melhores_probs, ent_th=0.55) 
                  else:
                      final_status = "contradicted"
                      if alignment["score"] > 0: 
                          alignment["score"] = -abs(alignment["score"])

          alignment["factual_status"] = final_status
          status_counts[final_status] += 1 

          if final_status == "contradicted":
              alignment["error_type"] = error_type
              alignment["error_types_all"] = [error_type]
              error_type_counts[error_type] += 1
          else:
              alignment["error_type"] = None
              alignment["error_types_all"] = []
  
          alignments.append(alignment)
  
      score = float(np.mean([al["score"] for al in alignments])) if alignments else 0.0
  
      total_claims = max(len(summary_claims), 1)
      total_contradicted = max(status_counts["contradicted"], 1)
  
      status_ratios = {k: v / total_claims for k, v in status_counts.items()}
      error_type_ratios = {k: v / total_contradicted for k, v in error_type_counts.items()}
  
      return {
          "score": score,
          "status_counts": status_counts,
          "status_ratios": status_ratios,
          "error_type_counts": error_type_counts,
          "error_type_ratios": error_type_ratios,
          "alignments": alignments,
      }

    def max_alignment_1(self, sample_alignments):
            sample_alignments = [s for s in sample_alignments if s is not None]
            # Escolhe o nível de texto (frase, parágrafo, doc) que tem o sinal mais decisivo
            alignment = max(sample_alignments, key=lambda x: x.get("relevance", x["score"]))
            return alignment

    def max_alignment(self, sample_alignments):
        sample_alignments = [s for s in sample_alignments if s is not None]
        
        # =====================================================================
        # ANTÍDOTO CALIBRADO PARA JANELAS DESLIZANTES (Sliding Windows)
        # =====================================================================
        for s in sample_alignments:
            passage = s["source_passage"]
            num_sentences = len([p for p in passage.split('.') if len(p.strip()) > 5])
            
            penalty = 1.0
            if num_sentences <= 3:
                # ZONA OURO: 1 a 3 frases. O contexto perfeito para resumos abstrativos.
                # Nenhuma penalidade aplicada. Deixamos o DeBERTa usar a coesão a seu favor.
                penalty = 1.0 
            elif num_sentences == 4:
                # Começa a ficar muito longo, leve desconto para forçar o foco.
                penalty = 0.90 
            else:
                # Blocos gigantes (doc_chunks). Desconto severo contra alucinações.
                penalty = 0.70 
                
            original_score = s.get("relevance", s["score"])
            
            if original_score > 0:
                s["search_priority"] = original_score * penalty
            else:
                s["search_priority"] = original_score / penalty

        alignment = max(sample_alignments, key=lambda x: x.get("search_priority", x["score"]))
        return alignment

    def get_alignment(
        self, premises, hypothesis, sample_id, hypothesis_id, alignment_prefix: Optional[str] = None,
    ):
        alignments_ids = []
        for prem_id in range(len(premises)):
            pair_id = self.get_alignment_id(
                sample_id=sample_id, premise_id=prem_id, hypothesis_id=hypothesis_id,
                premise=premises[prem_id], alignment_prefix=alignment_prefix,
            )
            alignments_ids.append(pair_id)
        loaded_alignment = self.load_alignment(
            alignments_ids=alignments_ids, premises=premises, hypothesis=hypothesis
        )
        if loaded_alignment is None:
            # Alinhamento exaustivo (Testa contra todas as frases)
            all_pairs = [(x, hypothesis) for x in premises]
            self.cache_alignment(alignments_ids, all_pairs)
            alignment, _ = self.load_alignment(
                alignments_ids=alignments_ids, premises=premises, hypothesis=hypothesis
            )
        else:
            alignment, _ = loaded_alignment
        return alignment

    def score_batch(self, batch: List[Dict[str, str]], ground_truths: List[Dict] = None) -> List[Dict]:
        documents = [el["document"] for el in batch]
        summaries = [el["summary"] for el in batch]
        if ground_truths is None:
            ground_truths = [{} for _ in batch]
        self.cache(documents, summaries)
        predictions = []
        for sample_id, (doc, summary, gt) in tqdm(
            enumerate(zip(documents, summaries, ground_truths)),
            total=len(documents),
            desc="Computing FENICE...",
        ):
            predictions.append(self._score(sample_id, doc, summary, error_ground_truth=gt))
        
        del self.nli_aligner
        
        self.sentences_cache.clear()
        self.coref_clusters_cache.clear()
        self.claims_cache.clear()
        self.alignments_cache.clear()
        self.spacy_cache.clear()
        self.doc_chunks_cache.clear()
        
        gc.collect() 
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
          doc_id = self.get_id(i, documents[i])
          sent_texts = [s[0] for s in sentences]
          self.sentences_cache[doc_id] = sentences

          if self.doc_level_nli:
              chunks = self._create_coherent_chunks(
                  sent_texts, max_tokens=self.doc_chunk_max_tokens, overlap_sentences=self.doc_chunk_overlap
              )
              self.doc_chunks_cache[doc_id] = chunks

    def get_docs_to_process(self, documents, cache):
        ids = [doc_id for doc_id in range(len(documents))]
        ids_to_process = [id for id in ids if self.get_id(id, documents[id]) not in cache]
        doc_to_process = [documents[i] for i in ids_to_process]
        return doc_to_process, ids_to_process

    def get_id(self, sample_id: int, text: str, k_chars: int = 100):
        return f"{sample_id}{text[:k_chars]}"

    def cache_claims(self, summaries):
        claim_extractor = ClaimExtractor(
            batch_size=self.claim_extractor_batch_size, device=self.device
        )
        claims_predictions = claim_extractor.process_batch(summaries)
        for summ_id, claims in enumerate(claims_predictions):
            id = self.get_id(summ_id, summaries[summ_id])
            self.claims_cache[id] = claims
        del claim_extractor
        gc.collect() 
        torch.cuda.empty_cache() 

    def cache_coref(self, documents):
        coref_model = CoreferenceResolution(
            batch_size=self.coreference_batch_size, device=self.device
        )
        all_clusters = coref_model.get_clusters_batch(documents)
        for doc_id, clusters in enumerate(all_clusters):
            id = self.get_id(doc_id, documents[doc_id])
            self.coref_clusters_cache[id] = clusters
        del coref_model.model
        gc.collect() 
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
                sentences, self.num_sent_per_paragraph, sliding_paragraphs=self.sliding_paragraphs, sliding_stride=self.sliding_stride,
            )
            alignments_ids, all_pairs = self.compute_nli_pairs(
                alignments_ids, all_pairs, claims, sample_id, sentences
            )
            if self.paragraph_level_nli:
                alignments_ids, all_pairs = self.compute_nli_pairs(
                    alignments_ids, all_pairs, claims, sample_id, paragraphs, prefix="par",
                )
            if self.doc_level_nli:
                doc_chunks = self.doc_chunks_cache[document_id] 
                alignment_ids_doc, all_pairs_doc = self.compute_nli_pairs(
                    alignment_ids_doc, all_pairs_doc, claims, sample_id, doc_chunks, prefix="doc",
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
        self, sample_id: int, premise_id: int, hypothesis_id: int, premise: str,
        alignment_prefix: Optional[str] = None, k_chars: int = 100,
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
            self.alignments_cache[id] = (
                ent.detach().cpu(), contr.detach().cpu(), neut.detach().cpu(),
            )
    
    def load_alignment_1(
        self, alignments_ids: List[str], premises: List[str], hypothesis: str
    ):
        if all([k in self.alignments_cache for k in alignments_ids]):
            scores = [self.alignments_cache[key] for key in alignments_ids]
            alignment = None
            max_search_score = -np.inf
            best_probs = None
            final_fenice_score = 0.0

            for i, (ent, contr, neut) in enumerate(scores):
                ent, contr, neut = ent.item(), contr.item(), neut.item()
                search_score = ent - neut 
                
                if search_score > max_search_score:
                    max_search_score = search_score
                    best_probs = [ent, contr, neut]
                    alignment = (premises[i], [ent, contr, neut])
                    final_fenice_score = ent - contr 
            
            return (
                {
                    "score": final_fenice_score, 
                    "probs": best_probs, 
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
            max_search_score = -np.inf
            best_probs = None
            final_fenice_score = 0.0

            # Vetoriza o claim e as premissas
            with torch.inference_mode():
                hyp_emb = sbert_model.encode(hypothesis, convert_to_tensor=True)
                prem_embs = sbert_model.encode(premises, convert_to_tensor=True)
                sim_scores = util.cos_sim(hyp_emb, prem_embs)[0]

            for i, (ent, contr, neut) in enumerate(scores):
                ent, contr, neut = ent.item(), contr.item(), neut.item()
                sim = sim_scores[i].item()
                
                # =================================================================
                # O ESCUDO CONTRA FALSAS CONTRADIÇÕES (Maçãs vs Laranjas)
                # =================================================================
                # Só aceitamos o sinal de Contradição na busca se as frases forem
                # semânticamente muito parecidas (sim > 0.45).
                if sim >= 0.45:
                    base_score = max(ent, contr) - neut
                else:
                    base_score = ent - neut # Se os assuntos divergem, ignoramos a falsa contradição
                
                # A similaridade atua como força gravitacional para puxar a frase certa
                search_score = base_score + sim
                # =================================================================

                if search_score > max_search_score:
                    max_search_score = search_score
                    best_probs = [ent, contr, neut]
                    alignment = (premises[i], [ent, contr, neut])
                    final_fenice_score = ent - contr 
            
            return (
                {
                    "score": final_fenice_score, 
                    "probs": best_probs, 
                    "summary_claim": hypothesis,
                    "source_passage": alignment[0],
                    "relevance": max_search_score 
                },
                scores,
            )
        else:
            return None
    
    def classify_factual_status(self, probs, ent_th=0.7, contr_th=0.6):
      if probs is None: return "not_supported"
      ent, contr, neut = probs
      if ent >= ent_th: return "supported"
      elif contr >= contr_th: return "contradicted"
      else: return "not_supported"

    def get_spacy_doc(self, text):
      if text not in self.spacy_cache:
        self.spacy_cache[text] = nlp(text)
      return self.spacy_cache[text]

    def _create_coherent_chunks(self, sentences: List[str], max_tokens: int, overlap_sentences: int = 0) -> List[str]:
      if not sentences: return []
      tokenizer = self.nli_tokenizer
      chunks, current_chunk = [], []
      current_tokens, i = 0, 0
      while i < len(sentences):
          sent = sentences[i]
          tokens_count = len(tokenizer.encode(sent, add_special_tokens=False))
          if tokens_count > max_tokens:
              if current_chunk:
                  chunks.append(" ".join(current_chunk))
                  current_chunk, current_tokens = [], 0
              chunks.append(sent)
              i += 1
              continue
          if current_tokens + tokens_count > max_tokens:
              chunks.append(" ".join(current_chunk))
              if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                  overlap_start = len(current_chunk) - overlap_sentences
                  current_chunk = current_chunk[overlap_start:]
                  current_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)
              else:
                  current_chunk, current_tokens = [], 0
          current_chunk.append(sent)
          current_tokens += tokens_count
          i += 1
      if current_chunk:
          chunks.append(" ".join(current_chunk))
      return chunks