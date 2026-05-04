<div align="center">
  <h1>FENICE-LongDoc: Factual Consistency Evaluation for Long-Document Summarization</h1>
</div>

[![Status](https://img.shields.io/badge/Status-Under%20Review-blue)]
[![Based on FENICE](https://img.shields.io/badge/Based%20on-FENICE%20(ACL%202024)-4b44ce)]
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)]

> ⚠️ **Acknowledgment:** This repository builds upon the FENICE framework (Scirè et al., ACL 2024), extending it to long-document summarization scenarios.

---

## 📖 Overview

Factual inconsistency remains a critical limitation in abstractive summarization, particularly for long documents where relevant information is distributed across extended contexts. Traditional metrics (e.g., ROUGE and BERTScore) rely on lexical overlap and often fail to capture factual correctness.

**FENICE-LongDoc** extends the original FENICE framework by organizing the evaluation into three complementary analysis components:

1. **Claim-level Classification & Error Taxonomy:**  
   Categorizes factual inconsistencies into Entity, Predicate, and Coreference errors using heuristic linguistic analysis (`spaCy`).

2. **Composite Metrics:**  
   Introduces *Evidence*, *Faithfulness*, and *Stability*, combined into a unified **Reliability Score** (interpreted as an aggregated proxy).

3. **Positional Analysis:**  
   Quantifies how different regions (start, middle, end) of the source document contribute to generated summaries, enabling the analysis of positional bias.

---

## ✨ Key Features & Hardware Optimizations

Evaluating long documents (e.g., GovReport dataset with up to 16k tokens) is computationally demanding. This pipeline includes practical engineering solutions:

- **Sliding Window Chunking:**  
  Handles token limits by aggregating NLI scores across overlapping document chunks.

- **Automated Checkpointing:**  
  Saves intermediate progress to ensure robustness during long executions.

- **VRAM Management:**  
  Includes scheduled pauses, `garbage collection`, and `cuda.empty_cache()` routines to mitigate out-of-memory errors on consumer GPUs (e.g., RTX 3060 12GB).

- **Unified Evaluation Pipeline:**  
  Computes FENICE-based metrics, ROUGE, BERTScore, and source usage statistics in a single execution.

---

## 🛠️ Installation

Create a Conda environment and install dependencies:

```sh
conda create -n fenice-longdoc python=3.10
conda activate fenice-longdoc

---

## Install required packages:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentencepiece scikit-learn pandas openpyxl spacy fastcoref rouge_score bert_score sentence-transformers matplotlib tqdm

---

## Download the required spaCy model:
python -m spacy download en_core_web_sm

---

## ▶️ Usage

The evaluation pipeline is executed via command line:

```sh
python Avaliador_deep_V2_5.py \
  --input your_summaries.json \
  --output Resultados_Avaliador/resultados_avaliacao.xlsx \
  --batch_size 1 \
  --seed 42