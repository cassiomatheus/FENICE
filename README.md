# Factual Error Analysis in Abstractive Summarization

This repository contains the implementation and experimental pipeline for analyzing factual errors in abstractive summarization systems. The approach is based on Natural Language Inference (NLI) and linguistic heuristics, as described in the accompanying paper.

This repository is released in anonymized form for peer review.

## Human Evaluation Protocol

This repository includes a structured human evaluation protocol designed to assess factual consistency in generated summaries:

- `Human_Evaluation_Form_for_Factual_Consistency.pdf`

The document provides annotation guidelines and a standardized evaluation form covering different categories of factual errors, including entity errors, predicate errors, and coreference issues.

This protocol was developed to support future human validation of the proposed error taxonomy and evaluation framework. Due to practical constraints (e.g., time, cost, and annotator availability), human evaluation was not conducted in the current study. However, the inclusion of this resource enables reproducibility and facilitates its adoption in subsequent work.

---

## Repository Structure

```
.
├── pipeline_sumarização.py     # Main pipeline for summary generation
├── fenice_experiment.py        # FENICE-based factual evaluation
├── metricas.py                 # Auxiliary metrics
├── Teste_Wilcoxon.py          # Statistical significance tests
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Setup

### 1. Create Environment

```bash
conda create -n factual_eval python=3.10
conda activate factual_eval
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Summaries

```bash
python pipeline_sumarização.py
```

### 2. Run Factual Evaluation

```bash
python fenice_experiment.py
```

### 3. Compute Additional Metrics

```bash
python metricas.py
```

### 4. Statistical Testing

```bash
python Teste_Wilconxon.py
```

---

## Methodology Overview

The pipeline follows these main steps:

- Claim extraction from generated summaries  
- Alignment between claims and source document segments using NLI  
- Classification into:
  - Supported
  - Contradicted
  - Not Supported  
- Aggregation into factual consistency metrics  
- Error categorization using linguistic heuristics  

---

## External Resources

This project builds upon prior work. For the original implementation of the FENICE framework:

> Official implementation (external work):  
> [Link omitted for double-blind review]

---

## Limitations

- Evaluation depends on a single NLI model  
- Experiments conducted in a zero-shot setting  
- No human annotation was performed due to resource constraints  

A human annotation protocol is included to support future validation.

---

## License

This project is released for academic and research purposes only.

---

## Citation

If accepted, citation details will be provided in the final version of the paper.
