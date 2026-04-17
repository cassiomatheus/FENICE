#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline de Avaliação com FENICE, ROUGE, BERTScore e métricas de desempenho
=============================================================================
Este script avalia resumos gerados por modelos de linguagem usando:
- FENICE (factualidade e cobertura)
- ROUGE‑1, ROUGE‑2, ROUGE‑L (sobreposição lexical)
- BERTScore (similaridade semântica contextual)
- Estatísticas de uso da fonte (coverage, posição, etc.)
- Métricas agregadas: evidence_score, faithfulness_score, stability_score, reliability_score
- Tempo de execução por amostra
- Pico de VRAM por amostra (quando GPU disponível)

O arquivo de entrada (JSON) deve conter as colunas:
    "texto_original" : texto fonte
    "resumo_gerado"  : resumo produzido

O arquivo de saída (Excel) incluirá, para cada amostra:
    sample_id, score (FENICE), supported, contradicted, not_supported,
    predicate_errors, entity_errors, coref_errors, other_erros,
    coverage_ratio, mean_position, start_ratio, middle_ratio, end_ratio,
    ref_len_tokens, summary_len_tokens,
    execution_time (s), vram_max_mb,
    rouge1_f, rouge1_p, rouge1_r,
    rouge2_f, rouge2_p, rouge2_r,
    rougeL_f, rougeL_p, rougeL_r,
    bert_score_f1, bert_score_precision, bert_score_recall,
    evidence_score, faithfulness_score, stability_score, reliability_score

Uso típico:
    python avaliador.py --input resumos.json --output avaliacao.xlsx --seed 42
"""

import os
import sys
import argparse
import subprocess
import logging
import random
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from rouge_score import rouge_scorer
from bert_score import score as bert_score

import gc

try:
    import importlib.metadata as metadata
except ImportError:
    metadata = None
    
torch.set_num_threads(10) # Limita a metade das threads do Ryzen 5500

# ==============================
# CONFIGURAÇÕES PADRÃO
# ==============================
DEFAULT_FENICE_REPO = "https://github.com/cassiomatheus/FENICE.git"
DEFAULT_FENICE_COMMIT = "reducao"
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED = 42

rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ==============================
# UTILITÁRIOS
# ==============================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(log_file: str = "avaliacao.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_cmd(cmd, cwd=None):
    logging.info("Executando: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        logging.error(f"Comando falhou com código {e.returncode}: {e}")
        sys.exit(1)

def detect_device():
    if torch.cuda.is_available():
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        return "cuda"
    logging.info("Rodando em CPU")
    return "cpu"

def ensure_fenice(repo_url: str, commit: str, no_git: bool = False):
    current_dir = Path.cwd()
    if (current_dir / "metric").exists() and (current_dir / "metric" / "FENICE.py").exists():
        logging.info("Diretório atual já contém a FENICE. Usando local existente.")
        repo_dir = current_dir
        if not no_git:
            try:
                run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
                run_cmd(["git", "checkout", commit], cwd=repo_dir)
            except Exception as e:
                logging.warning(f"Não foi possível verificar/atualizar o commit: {e}")
        sys.path.insert(0, str(repo_dir))
        return repo_dir

    repo_dir = Path("FENICE").absolute()
    if no_git:
        if not repo_dir.exists():
            logging.error("Modo --no_git ativado, mas diretório FENICE não encontrado.")
            sys.exit(1)
        logging.info("Usando FENICE local existente.")
    else:
        if not repo_dir.exists():
            logging.info("Clonando repositório FENICE...")
            run_cmd(["git", "clone", repo_url, str(repo_dir)])
        else:
            logging.info("Repositório FENICE já existe. Atualizando...")
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", commit], cwd=repo_dir)
        logging.info(f"Checkout para {commit} concluído.")
    sys.path.insert(0, str(repo_dir))
    return repo_dir

def validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["texto_original", "resumo_gerado"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.error(f"Colunas obrigatórias ausentes no JSON: {missing}")
        sys.exit(1)
    before = len(df)
    df = df.dropna(subset=required_cols)
    df = df[df["texto_original"].str.strip().astype(bool)]
    df = df[df["resumo_gerado"].str.strip().astype(bool)]
    after = len(df)
    if after < before:
        logging.warning(f"Removidas {before - after} linhas com dados inválidos.")
    df = df.rename(columns={"texto_original": "document", "resumo_gerado": "summary"})
    return df.reset_index(drop=True)

def df_to_fenice_batches(df: pd.DataFrame, batch_size: int = 1):
    batch = []
    for idx, row in df.iterrows():
        batch.append({"document": row["document"], "summary": row["summary"]})
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def load_checkpoint(checkpoint_file: str):
    if os.path.exists(checkpoint_file):
        logging.info(f"Checkpoint encontrado: {checkpoint_file}. Retomando...")
        return pd.read_csv(checkpoint_file).to_dict("records")
    return []

def save_checkpoint(results, checkpoint_file: str):
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(checkpoint_file, index=False, encoding='utf-8')
    logging.info(f"Checkpoint salvo em {checkpoint_file}")

def save_metadata(args, fenice_params, commit_used, output_dir):
    meta = {
        "argumentos": vars(args),
        "fenice_params": fenice_params,
        "fenice_commit": commit_used,
        "data_execucao": datetime.now().isoformat(),
        "versoes_bibliotecas": {}
    }
    if metadata:
        for lib in ["torch", "transformers", "pandas", "numpy", "openpyxl", "bert_score", "rouge_score"]:
            try:
                meta["versoes_bibliotecas"][lib] = metadata.version(lib)
            except:
                meta["versoes_bibliotecas"][lib] = "desconhecido"
    meta_path = os.path.join(output_dir, "metadados_avaliacao.json")
    meta_dir = os.path.dirname(meta_path)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logging.info(f"Metadados salvos em {meta_path}")

# ==============================
# PIPELINE PRINCIPAL
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="Avalia resumos gerados usando FENICE, ROUGE e BERTScore.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True,
                        help="Arquivo JSON com os resumos (colunas: 'texto_original', 'resumo_gerado')")
    parser.add_argument("--output", default="Resultados_Avaliador/resultados_avaliacao.xlsx",
                        help="Arquivo Excel de saída")
    parser.add_argument("--checkpoint", default="checkpoint_avaliacao.csv",
                        help="Arquivo CSV para checkpoint")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Tamanho do lote (recomendado 1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Semente aleatória")
    parser.add_argument("--fenice_repo", default=DEFAULT_FENICE_REPO,
                        help="URL do repositório FENICE")
    parser.add_argument("--fenice_commit", default=DEFAULT_FENICE_COMMIT,
                        help="Commit/branch específico")
    parser.add_argument("--no_git", action="store_true",
                        help="Usar FENICE local sem clonar/atualizar")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logging()
    device = detect_device()

    input_json_path = os.path.abspath(args.input)
    excel_path = os.path.abspath(args.output)
    checkpoint_path = os.path.abspath(args.checkpoint)
    output_dir = os.path.dirname(excel_path)

    if not os.path.exists(input_json_path):
        logging.error(f"Arquivo de entrada não encontrado: {input_json_path}")
        sys.exit(1)

    original_dir = os.getcwd()
    fenice_dir = ensure_fenice(args.fenice_repo, args.fenice_commit, args.no_git)

    if os.path.abspath(fenice_dir) != os.path.abspath(original_dir):
        os.chdir(fenice_dir)
        logging.info(f"Diretório alterado para: {fenice_dir}")

    # Importações da FENICE
    try:
        from metric.FENICE import FENICE
        from metric.utils.source_usage import compute_source_usage_chunks
        try:
            from metric.utils.utils import evaluate_sample
            HAS_EVALUATE_SAMPLE = True
        except ImportError:
            HAS_EVALUATE_SAMPLE = False
            logging.warning("Função evaluate_sample não encontrada. Métricas agregadas serão None.")
    except ImportError as e:
        logging.error(f"Erro ao importar módulos da FENICE: {e}")
        sys.exit(1)

    # Carrega dados
    logging.info(f"Carregando dados de: {input_json_path}")
    try:
        if input_json_path.endswith('.jsonl'):
            df = pd.read_json(input_json_path, lines=True)
        else:
            df = pd.read_json(input_json_path)
    except Exception as e:
        logging.error(f"Falha ao ler JSON: {e}")
        sys.exit(1)

    df = validate_input_dataframe(df)
    logging.info(f"Total de amostras válidas: {len(df)}")

    # Tokenização para comprimento
    logging.info("Tokenizando documentos e resumos (BERT base uncased)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df['ref_len_tokens'] = df['document'].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False, truncation=False))
    )
    df['summary_len_tokens'] = df['summary'].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False, truncation=False))
    )

    # Inicializa FENICE
    logging.info("Inicializando FENICE...")
    fenice_params = {
        "use_coref": True,
        "paragraph_level_nli": True,
        "doc_level_nli": False
    }
    fenice = FENICE(**fenice_params)

    results = load_checkpoint(checkpoint_path)
    start_index = len(results)
    if start_index > 0:
        logging.info(f"Retomando a partir do índice {start_index} (já processadas {start_index} amostras).")
    else:
        logging.info("Nenhum checkpoint encontrado. Iniciando do zero.")

    total = len(df)
    batch_iter = df_to_fenice_batches(df.iloc[start_index:], args.batch_size)
    global_idx = start_index

    for batch in batch_iter:
        print("\n" + "-" * 20)
        logging.info(f"Processando lote com {len(batch)} documento(s) (índice global {global_idx})...")

        # --- INÍCIO DA MEDIÇÃO DE TEMPO (TUDO) ---
        start_time = time.time()

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        try:
            fenice_out_list = fenice.score_batch(batch)
        except Exception as e:
            logging.error(f"Erro na FENICE para o lote iniciado em {global_idx}: {e}")
            # Registra erro para cada item do lote
            for j, item in enumerate(batch):
                results.append({
                    "sample_id": global_idx + j,
                    "erro": str(e),
                    "score": None, "supported": None, "contradicted": None,
                    "not_supported": None, "predicate_errors": None, "entity_errors": None,
                    "coref_errors": None, "other_erros": None,
                    "coverage_ratio": None, "mean_position": None,
                    "start_ratio": None, "middle_ratio": None, "end_ratio": None,
                    "ref_len_tokens": df.loc[global_idx + j, "ref_len_tokens"],
                    "summary_len_tokens": df.loc[global_idx + j, "summary_len_tokens"],
                    "execution_time": None, "vram_max_mb": None,
                    "rouge1_f": None, "rouge1_p": None, "rouge1_r": None,
                    "rouge2_f": None, "rouge2_p": None, "rouge2_r": None,
                    "rougeL_f": None, "rougeL_p": None, "rougeL_r": None,
                    "bert_score_f1": None, "bert_score_precision": None, "bert_score_recall": None,
                    "evidence_score": None, "faithfulness_score": None,
                    "stability_score": None, "reliability_score": None,
                })
            global_idx += len(batch)
            save_checkpoint(results, checkpoint_path)
            continue

        # Processa cada amostra do lote
        batch_results = []  # acumula os dicionários deste lote
        for j, fenice_out in enumerate(fenice_out_list):
            abs_idx = global_idx + j
            doc = batch[j]["document"]
            summ = batch[j]["summary"]

            # Estatísticas de uso da fonte (sempre disponíveis)
            usage_stats = compute_source_usage_chunks(
                alignments=fenice_out["alignments"],
                document=doc
            )

            # ROUGE
            rouge_scores = rouge_scorer_obj.score(doc, summ)
            rouge1 = rouge_scores['rouge1']
            rouge2 = rouge_scores['rouge2']
            rougeL = rouge_scores['rougeL']

            # BERTScore
            try:
                #P, R, F1 = bert_score([summ], [doc], lang="en", verbose=False, device=device)
                
                if device == "cuda":
                    # Habilita o contexto de meia precisão (FP16) para a GPU
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        P, R, F1 = bert_score(
                            [summ], 
                            [doc], 
                            lang="en", 
                            verbose=False, 
                            device=device, 
                            model_type="distilbert-base-uncased"
                        )
                else:
                    # Fallback normal se estiver rodando em CPU
                    P, R, F1 = bert_score(
                        [summ], 
                        [doc], 
                        lang="en", 
                        verbose=False, 
                        device=device, 
                        model_type="distilbert-base-uncased"
                    )
                
                
                bert_f = F1[0].item()
                bert_p = P[0].item()
                bert_r = R[0].item()
            except Exception as e:
                logging.error(f"Erro no BERTScore para amostra {abs_idx}: {e}")
                bert_f = bert_p = bert_r = None

            # Métricas agregadas (se disponíveis)
            # Métricas agregadas (se disponíveis)
            if HAS_EVALUATE_SAMPLE:
                try:
                    # Passamos os alinhamentos e o score já calculados para evitar o KeyError
                    agg_metrics = evaluate_sample(
                        fenice, 
                        doc, 
                        summ, 
                        precomputed_alignments=fenice_out["alignments"], 
                        base_score=fenice_out["score"]
                    )
                    evidence = agg_metrics.get("evidence_score")
                    faithfulness = agg_metrics.get("faithfulness_score")
                    stability = agg_metrics.get("stability_score")
                    reliability = agg_metrics.get("reliability_score")
                except Exception as e:
                    logging.error(f"Erro no evaluate_sample para amostra {abs_idx}: {e}")
                    evidence = faithfulness = stability = reliability = None
            else:
                evidence = faithfulness = stability = reliability = None

            # Monta dicionário parcial (sem tempo/VRAM ainda)
            result_item = {
                "sample_id": abs_idx,
                "score": round(fenice_out["score"], 5) if fenice_out["score"] is not None else None,
                "supported": fenice_out["status_counts"]["supported"],
                "contradicted": fenice_out["status_counts"]["contradicted"],
                "not_supported": fenice_out["status_counts"]["not_supported"],
                "predicate_errors": fenice_out["error_type_counts"]["predicate"],
                "entity_errors": fenice_out["error_type_counts"]["entity"],
                "coref_errors": fenice_out["error_type_counts"]["coreference"],
                "other_erros": fenice_out["error_type_counts"]["other"],
                "coverage_ratio": round(usage_stats["coverage_ratio"], 5),
                "mean_position": round(usage_stats["mean_position"], 5),
                "start_ratio": round(usage_stats["region_ratios"]["start"], 5),
                "middle_ratio": round(usage_stats["region_ratios"]["middle"], 5),
                "end_ratio": round(usage_stats["region_ratios"]["end"], 5),
                "ref_len_tokens": df.loc[abs_idx, "ref_len_tokens"],
                "summary_len_tokens": df.loc[abs_idx, "summary_len_tokens"],
                "rouge1_f": round(rouge1.fmeasure, 5),
                "rouge1_p": round(rouge1.precision, 5),
                "rouge1_r": round(rouge1.recall, 5),
                "rouge2_f": round(rouge2.fmeasure, 5),
                "rouge2_p": round(rouge2.precision, 5),
                "rouge2_r": round(rouge2.recall, 5),
                "rougeL_f": round(rougeL.fmeasure, 5),
                "rougeL_p": round(rougeL.precision, 5),
                "rougeL_r": round(rougeL.recall, 5),
                "bert_score_f1": round(bert_f, 5) if bert_f is not None else None,
                "bert_score_precision": round(bert_p, 5) if bert_p is not None else None,
                "bert_score_recall": round(bert_r, 5) if bert_r is not None else None,
                "evidence_score": round(evidence, 5) if evidence is not None else None,
                "faithfulness_score": round(faithfulness, 5) if faithfulness is not None else None,
                "stability_score": round(stability, 5) if stability is not None else None,
                "reliability_score": round(reliability, 5) if reliability is not None else None,
            }
            batch_results.append(result_item)

        # --- FIM DA MEDIÇÃO DE TEMPO (TUDO) ---
        end_time = time.time()
        batch_total_time = end_time - start_time
        vram_peak = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

        # Rateia o tempo e a VRAM entre as amostras do lote
        time_per_sample = batch_total_time / len(batch)
        vram_per_sample = vram_peak  # A VRAM é um pico, não pode ser rateada; atribuímos o mesmo valor a todas (o pico do lote)
        # Mas se preferir, pode deixar vram_peak como o valor do lote para todas as amostras

        for result_item in batch_results:
            result_item["execution_time"] = round(time_per_sample, 5)
            result_item["vram_max_mb"] = round(vram_peak, 2) if vram_peak else 0.0
            results.append(result_item)
            
        # Salva o índice anterior para a lógica da pausa
        prev_idx = global_idx
        
        global_idx += len(batch)
        save_checkpoint(results, checkpoint_path)
        
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            #torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        
        # ==========================================
        # NOVA LÓGICA DE PAUSA (5 MINUTOS)
        # ==========================================
        # Verifica se cruzamos um múltiplo de 10 e se não estamos no final do arquivo
        if (global_idx // 10) > (prev_idx // 10) and global_idx < total:
            logging.info(f"Marco de 10 documentos atingido (Total avaliado: {global_idx}). Iniciando pausa de 3 minutos (180 segundos)...")
            time.sleep(180)
            logging.info("Pausa finalizada. A GPU/API esfriou, retomando a avaliação.")
        # ==========================================

    # Salvamento final
    logging.info("Processamento concluído. Gerando arquivo Excel final...")
    df_results = pd.DataFrame(results)
    excel_dir = os.path.dirname(excel_path)
    if excel_dir:
        os.makedirs(excel_dir, exist_ok=True)
    df_results.to_excel(excel_path, index=False, engine='openpyxl')
    logging.info(f"Resultados salvos em: {excel_path}")

    save_metadata(args, fenice_params, args.fenice_commit, output_dir)
    logging.info("Pipeline de avaliação finalizado com sucesso!")

if __name__ == "__main__":
    main()