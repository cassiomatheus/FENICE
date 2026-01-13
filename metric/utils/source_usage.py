from typing import List

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

