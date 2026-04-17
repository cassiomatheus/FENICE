import hashlib
import pickle
from typing import List, Tuple, Union
import spacy
import sys
import os
from spacy.cli import download
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import re



# ===============================
# MODELO SEMÂNTICO (leve)
# ===============================
#sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Tokenizer para chunking com offsets e mesmo usado no FENICE
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Modelo SBERT
#sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

chunk_embedding_cache = {}

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    os.environ["SPACY_WARNING_IGNORE"] = "true"  # Prevent interactive prompts
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def flatten(two_d_list: List[List]) -> List:
    """
    Flattens a 2D Python list into a 1D list using list comprehension.
    """
    return [item for sublist in two_d_list for item in sublist]


def chunks(lst: List, n: int) -> List[List]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def sliding_chunks(lst: List, n: int, sliding_stride: int = 1) -> List[List]:
    """Yield sliding windows of n-sized chunks from lst."""
    for i in range(len(lst) - n + 1):
        if i % sliding_stride == 0:
            yield lst[i : i + n]


def distinct(input_list: List) -> List:
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]


def split_into_sentences(
    text: str, return_offsets: bool = False
) -> Union[List[Tuple[str, str, str]], List[str]]:
    # Process the text with spaCy
    doc = nlp(text)
    return get_sentences(doc, return_offsets)


def get_sentences(doc, return_offsets):
    # Initialize a list to store sentences and their offsets
    sentences_with_offsets = []
    # Iterate over the sentences in the processed text
    for sent in doc.sents:
        # Get the sentence text and character offsets
        sentence_text = sent.text
        start_offset = sent.start_char
        end_offset = sent.end_char

        # Add the sentence text and offsets to the list as a tuple
        sentences_with_offsets.append((sentence_text, start_offset, end_offset))
    if return_offsets:
        return sentences_with_offsets
    else:
        # Return only the sentences
        return [sentence[0] for sentence in sentences_with_offsets]


def split_into_sentences_batched(
    texts: List[str], return_offsets: bool = False, batch_size=32
) -> List[List[Tuple[str, str, str]]]:
    # Process the text with spaCy
    batches = list(chunks(texts, batch_size))
    sentences = []
    for b in tqdm(
        batches, total=len(batches), desc="splitting document batches into sentences"
    ):
        docs = nlp.pipe(b, disable=["attribute_ruler", "lemmatizer", "ner"])
        for doc in docs:
            doc_sentences = get_sentences(doc, return_offsets=return_offsets)
            sentences.append(doc_sentences)
    return sentences


def split_into_paragraphs(
    sentences: List[str],
    num_sent_per_paragraph: int,
    sliding_paragraphs=True,
    sliding_stride: int = 1,
) -> List[str]:
    if len(sentences) < num_sent_per_paragraph:
        return [" ".join(sentences)]
    paragraphs = (
        list(sliding_chunks(sentences, num_sent_per_paragraph, sliding_stride))
        if sliding_paragraphs
        else list(chunks(sentences, num_sent_per_paragraph))
    )
    for i, par in enumerate(paragraphs):
        paragraphs[i] = " ".join(par)
    return paragraphs


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_pickle(path: str):
    # Open the file in binary read mode
    with open(path, "rb") as file:
        # Load the object from the file
        return pickle.load(file)


def dump_pickle(path: str, data):
    # Open the file in binary write mode
    with open(path, "wb") as file:
        # Dump the object to the file
        pickle.dump(data, file)


# --------------------------------------------------
# Chunking do documento
# --------------------------------------------------
def chunk_document_with_offsets(text, chunk_size=256):
    tokens = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False
    )

    offsets = tokens["offset_mapping"]
    chunks = []

    for i in range(0, len(offsets), chunk_size):
        start = offsets[i][0]
        end = offsets[min(i + chunk_size - 1, len(offsets) - 1)][1]
        center = ((start + end) / 2) / len(text)

        chunks.append({
            "start": start,
            "end": end,
            "center": center
        })

    return chunks


# --------------------------------------------------
# Região a partir da posição normalizada
# --------------------------------------------------
def region_from_position(p):
    if p < 0.33:
        return "start"
    elif p < 0.66:
        return "middle"
    else:
        return "end"


# --------------------------------------------------
# Mapeia uma claim para o chunk mais próximo
# --------------------------------------------------
def map_claim_to_chunk(alignment, chunks, document):
    """
    Mapeia uma claim para o chunk do documento usando source_passage.
    """

    if alignment is None:
        return None

    if "source_passage" not in alignment:
        return None

    passage = alignment["source_passage"]
    if passage is None or passage.strip() == "":
        return None

    passage = passage.strip()

    # Tokeniza a evidência
    passage_tokens = set(
        tokenizer(passage, add_special_tokens=False)["input_ids"]
    )

    best_idx = None
    best_overlap = 0

    for idx, ch in enumerate(chunks):
        chunk_text = document[ch["start"]:ch["end"]]
        chunk_tokens = set(
            tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
        )

        overlap = len(passage_tokens & chunk_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    return best_idx

# --------------------------------------------------
# Proporção de claims por região
# --------------------------------------------------
def compute_claim_region_proportions(positions):
    region_counts = {"start": 0, "middle": 0, "end": 0}

    for p in positions:
        region_counts[region_from_position(p)] += 1

    total = sum(region_counts.values())

    region_ratios = {
        k: (v / total if total > 0 else 0.0)
        for k, v in region_counts.items()
    }

    return region_counts, region_ratios


# --------------------------------------------------
# Função principal (usada no seu script)
# --------------------------------------------------
def compute_source_usage_chunks(alignments, document):
    """
    Retorna métricas de uso da fonte:
    - coverage_ratio
    - mean_position
    - region_counts (claims)
    - region_ratios (claims)
    - textual_region_coverage (texto)
    """

    chunks = chunk_document_with_offsets(document)

    used_chunk_indices = set()
    positions = []

    for al in alignments:
        chunk_idx = map_claim_to_chunk_sbert(al, chunks, document)
        if chunk_idx is not None:
            used_chunk_indices.add(chunk_idx)
            positions.append(chunks[chunk_idx]["center"])

    if len(chunks) == 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = len(used_chunk_indices) / len(chunks)

    mean_position = float(np.mean(positions)) if positions else 0.0

    region_counts, region_ratios = compute_claim_region_proportions(positions)
    # textual_region_coverage = compute_textual_coverage_by_region(
    #     chunks,
    #     used_chunk_indices
    # )

    return {
        "coverage_ratio": coverage_ratio, #Não estou usando
        "mean_position": mean_position,   
        "region_counts": region_counts,
        "region_ratios": region_ratios,
        "positions": positions,           #Não estou usando
        "used_chunk_indices": used_chunk_indices #Não estou usando
    }

def get_chunk_embeddings(document, chunks):
    doc_id = hash(document)

    if doc_id in chunk_embedding_cache:
        return chunk_embedding_cache[doc_id]

    chunk_texts = [
        document[ch["start"]:ch["end"]].strip()
        for ch in chunks
        if document[ch["start"]:ch["end"]].strip()
    ]

    if not chunk_texts:
        return None

    with torch.inference_mode():
        embeddings = sbert_model.encode(
            chunk_texts,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=False
        )

    chunk_embedding_cache[doc_id] = embeddings
    return embeddings


def map_claim_to_chunk_sbert(alignment, chunks, document):

    if alignment is None:
        return None

    passage = alignment.get("source_passage", "").strip()
    if passage == "":
        return None

    chunk_embeddings = get_chunk_embeddings(document, chunks)
    if chunk_embeddings is None:
        return None

    with torch.inference_mode():
        emb_passage = sbert_model.encode(
            passage,
            convert_to_tensor=True
        )

    scores = util.cos_sim(emb_passage, chunk_embeddings)[0]
    return int(scores.argmax())

# ===============================
# 1. EVIDENCE SCORE
# ===============================
def compute_evidence_score(alignments):
    """
    Mede o quão forte é a evidência para cada claim
    baseado em similaridade semântica (sem NLI).
    """
    scores = []

    for al in alignments:
        claim = al.get("summary_claim", "")
        source = al.get("source_passage", "")

        if not claim or not source:
            continue

        emb_claim = sbert_model.encode(claim, convert_to_tensor=True)
        emb_source = sbert_model.encode(source, convert_to_tensor=True)

        sim = util.cos_sim(emb_claim, emb_source).item()
        scores.append(sim)

    return float(np.mean(scores)) if scores else 0.0


# ===============================
# 2. FAITHFULNESS SCORE
# ===============================
def compute_faithfulness_score(alignments, threshold=0.5):
    """
    % de claims que possuem evidência suficiente.
    """
    supported = 0
    total = 0

    for al in alignments:
        claim = al.get("summary_claim", "")
        source = al.get("source_passage", "")

        if not claim or not source:
            continue

        emb_claim = sbert_model.encode(claim, convert_to_tensor=True)
        emb_source = sbert_model.encode(source, convert_to_tensor=True)

        sim = util.cos_sim(emb_claim, emb_source).item()

        if sim >= threshold:
            supported += 1

        total += 1

    return supported / total if total > 0 else 0.0

# ==============================
# UTIL: split de sentenças
# ==============================
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]


# ==============================
# 1. REORDER (leve)
# ==============================
def reorder_sentences(text, intensity=0.3):
    sentences = split_sentences(text)

    if len(sentences) < 2:
        return text

    n_swaps = max(1, int(len(sentences) * intensity))

    for _ in range(n_swaps):
        i = random.randint(0, len(sentences) - 2)
        sentences[i], sentences[i+1] = sentences[i+1], sentences[i]

    return " ".join(sentences)


# ==============================
# 2. REMOVE (leve)
# ==============================
def remove_sentences(text, intensity=0.2):
    sentences = split_sentences(text)

    if len(sentences) <= 2:
        return text

    #n_remove = max(1, int(len(sentences) * intensity))
    #Evitar remoção agressiva demais, garante que sobra pelo menos 1 sentença
    n_remove = min(len(sentences) - 1, max(1, int(len(sentences) * intensity)))

    indices = set(random.sample(range(len(sentences)), n_remove))
    new_sentences = [s for i, s in enumerate(sentences) if i not in indices]

    return " ".join(new_sentences)


# ==============================
# 3. PARAPHRASE (leve e seguro)
# ==============================
def simple_paraphrase(text, intensity=0.3):
    sentences = split_sentences(text)

    if not sentences:
        return text

    replacements = {
        "however": "but",
        "therefore": "thus",
        "in addition": "also",
        "moreover": "furthermore",
        "because": "since",
        "although": "even though"
    }

    new_sentences = []

    for s in sentences:
        s_new = s

        # aplicar substituições com probabilidade
        for k, v in replacements.items():
            if k in s.lower() and random.random() < intensity:
                s_new = re.sub(k, v, s_new, flags=re.IGNORECASE)

        # pequena reordenação interna (segura)
        if random.random() < intensity and len(s.split()) > 6:
            words = s.split()
            i = random.randint(0, len(words) - 2)
            words[i], words[i+1] = words[i+1], words[i]
            s_new = " ".join(words)

        new_sentences.append(s_new)

    return " ".join(new_sentences)


# ==============================
# 4. FUNÇÃO PRINCIPAL (com pesos)
# ==============================
def perturb_text(
    text,
    weights=None,
    intensity=0.2,
    seed=None
):
    """
    weights: dict com pesos das operações
        exemplo: {"reorder": 0.4, "remove": 0.3, "paraphrase": 0.3}
    intensity: quão forte a perturbação
    """

    if seed is not None:
        random.seed(seed)

    if weights is None:
        weights = {
            "reorder": 0.4,
            "remove": 0.3,
            "paraphrase": 0.3
        }

    ops = list(weights.keys())
    probs = list(weights.values())

    choice = random.choices(ops, probs)[0]

    if choice == "reorder":
        return reorder_sentences(text, intensity)

    elif choice == "remove":
        return remove_sentences(text, intensity)

    elif choice == "paraphrase":
        return simple_paraphrase(text, intensity)

    return text


# ===============================
# 3. STABILITY SCORE
# ===============================
# def perturb_text(text):
#     """
#     Pequena perturbação:
#     - remove frases
#     - troca ordem
#     """
#     sentences = text.split(".")
#     sentences = [s.strip() for s in sentences if s.strip()]

#     if len(sentences) < 2:
#         return text

#     # remove uma sentença aleatória
#     if random.random() < 0.5:
#         sentences.pop(random.randint(0, len(sentences)-1))

#     # embaralha levemente
#     if random.random() < 0.5:
#         random.shuffle(sentences)

#     return ". ".join(sentences)


# def compute_stability_score(fenice, document, summary, n_runs=5, seed=None):
#     """
#     Mede consistência do sistema sob perturbações.
#     """
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)

#     # base
#     fenice.cache([document], [summary])
#     base = fenice._score(0, document, summary)
#     base_score = base["score"]

#     variations = []

#     for _ in range(n_runs):
#         perturbed_doc = perturb_text(document)

#         # garante perturbação real
#         if perturbed_doc.strip() == document.strip():
#             perturbed_doc = reorder_sentences(document, intensity=0.3)

#         fenice.cache([perturbed_doc], [summary])
#         out = fenice._score(0, perturbed_doc, summary)

#         variations.append(out["score"])

#     # se não conseguiu gerar variações válidas
#     if not variations:
#         return 1.0

#     # calcula variância corretamente
#     all_scores = [base_score] + variations
#     std_dev = np.std(all_scores, ddof=1) if len(all_scores) > 1 else 0.0

#     # transforma em estabilidade
#     stability = 1 / (1 + std_dev)

#     return float(stability)

# ===============================
# 3. STABILITY SCORE (CORRIGIDO PARA GERENCIAR MEMÓRIA)
# ===============================
def compute_stability_score(fenice, document, summary, base_score, n_runs=5, seed=None):
    """
    Mede consistência do sistema sob perturbações de forma segura.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    variations = []

    for _ in range(n_runs):
        perturbed_doc = perturb_text(document)

        # garante perturbação real
        if perturbed_doc.strip() == document.strip():
            perturbed_doc = reorder_sentences(document, intensity=0.3)

        # USAR SCORE_BATCH EM VEZ DE CACHE E _SCORE MANUAIS
        # Isso garante que a RAM e a VRAM serão limpas adequadamente a cada iteração
        out_list = fenice.score_batch([{"document": perturbed_doc, "summary": summary}])
        variations.append(out_list[0]["score"])

    # se não conseguiu gerar variações válidas
    if not variations:
        return 1.0

    # calcula variância corretamente
    all_scores = [base_score] + variations
    std_dev = np.std(all_scores, ddof=1) if len(all_scores) > 1 else 0.0

    # transforma em estabilidade
    stability = 1 / (1 + std_dev)

    return float(stability)


# ===============================
# 4. RELIABILITY FINAL
# ===============================
def compute_reliability(
    evidence_score,
    faithfulness_score,
    stability_score,
    weights=(0.5, 0.4, 0.1)
):
    """
    Combinação final dos scores
    """
    w1, w2, w3 = weights

    return (
        w1 * evidence_score +
        w2 * faithfulness_score +
        w3 * stability_score
    )


# ===============================
# 5. PIPELINE COMPLETO
# ===============================
# def evaluate_sample(fenice, document, summary):
#     """
#     Executa avaliação completa para um sample
#     """

#     result = fenice._score(0, document, summary)

#     alignments = result["alignments"]

#     evidence = compute_evidence_score(alignments)
#     faithfulness = compute_faithfulness_score(alignments)
#     stability = compute_stability_score(fenice, document, summary)

#     reliability = compute_reliability(
#         evidence,
#         faithfulness,
#         stability
#     )

#     return {
#         "evidence_score": evidence,
#         "faithfulness_score": faithfulness,
#         "stability_score": stability,
#         "reliability_score": reliability
#     }

# ===============================
# 5. PIPELINE COMPLETO (CORRIGIDO PARA EVITAR KEYERROR)
# ===============================
def evaluate_sample(fenice, document, summary, precomputed_alignments, base_score):
    """
    Executa avaliação completa para um sample sem refazer cálculos já processados
    """
    
    evidence = compute_evidence_score(precomputed_alignments)
    faithfulness = compute_faithfulness_score(precomputed_alignments)
    
    # Passamos o fenice e o base_score para rodar as perturbações isoladamente
    stability = compute_stability_score(fenice, document, summary, base_score)

    reliability = compute_reliability(
        evidence,
        faithfulness,
        stability
    )

    return {
        "evidence_score": evidence,
        "faithfulness_score": faithfulness,
        "stability_score": stability,
        "reliability_score": reliability
    }

