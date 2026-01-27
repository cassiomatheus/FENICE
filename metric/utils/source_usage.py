from typing import List
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import torch

# Tokenizer para chunking com offsets
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Modelo SBERT
#sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def segment_document(document: str):
    return [
        p.strip()
        for p in document.split("\n")
        if len(p.strip()) > 0
    ]

def find_segment_id(source_passage: str, segments):
    if source_passage == "DOCUMENT":
        return None

    best_id = None
    best_overlap = 0

    sp_tokens = set(source_passage.lower().split())

    for i, seg in enumerate(segments):
        seg_tokens = set(seg.lower().split())
        overlap = len(sp_tokens & seg_tokens)

        if overlap > best_overlap:
            best_overlap = overlap
            best_id = i

    return best_id

def analyze_source_usage(document: str, alignments: list):
    """
    Analisa quanto do texto-fonte é usado pelo resumo
    e de onde essa informação vem (posição).
    """

    segments = segment_document(document)

    used_segments = set()
    positions = []

    for al in alignments:
        seg_id = find_segment_id(al["source_passage"], segments)
        if seg_id is not None:
            used_segments.add(seg_id)
            pos = seg_id / (len(segments) - 1) if len(segments) > 1 else 0.0
            positions.append(pos)

    # Cobertura
    coverage_ratio = len(used_segments) / len(segments) if segments else 0.0

    # Distribuição posicional
    regions = {"start": 0, "middle": 0, "end": 0}

    for p in positions:
        if p < 0.33:
            regions["start"] += 1
        elif p < 0.66:
            regions["middle"] += 1
        else:
            regions["end"] += 1

    total = len(positions) or 1
    region_ratios = {k: v / total for k, v in regions.items()}

    return {
        "coverage_ratio": coverage_ratio,
        "used_segments": sorted(list(used_segments)),
        "mean_position": sum(positions) / total if positions else 0.0,
        "region_counts": regions,
        "region_ratios": region_ratios,
    }








#Novas implementações

from transformers import AutoTokenizer
import numpy as np

# tokenizer padrão (mesmo usado no FENICE)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


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
# Cobertura textual real por região
# --------------------------------------------------
# def compute_textual_coverage_by_region(chunks, used_chunk_indices):
#     region_chunks = {
#         "start": set(),
#         "middle": set(),
#         "end": set()
#     }

#     for idx, ch in enumerate(chunks):
#         region = region_from_position(ch["center"])
#         region_chunks[region].add(idx)

#     region_coverage = {}

#     for region in region_chunks:
#         total_chunks = len(region_chunks[region])
#         if total_chunks == 0:
#             region_coverage[region] = 0.0
#         else:
#             used = region_chunks[region] & used_chunk_indices
#             region_coverage[region] = len(used) / total_chunks

#     return region_coverage


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

def map_claim_to_chunk_sbert(alignment, chunks, document):

    if alignment is None:
        return None

    passage = alignment.get("source_passage", "").strip()
    if passage == "":
        return None

    # ================================
    # 1. textos dos chunks (uma vez)
    # ================================
    chunk_texts = [
        document[ch["start"]:ch["end"]].strip()
        for ch in chunks
        if document[ch["start"]:ch["end"]].strip()
    ]

    if len(chunk_texts) == 0:
        return None

    # ================================
    # 2. embeddings em batch (uma vez)
    # ================================
    with torch.no_grad():
        chunk_embeddings = sbert_model.encode(
            chunk_texts,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        emb_passage = sbert_model.encode(
            passage,
            convert_to_tensor=True
        )

    # ================================
    # 3. similaridade vetorizada
    # ================================
    scores = util.cos_sim(emb_passage, chunk_embeddings)[0]

    best_idx = int(scores.argmax())

    return best_idx
