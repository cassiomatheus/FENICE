import torch
import gc
import json
import sys
import logging
import os
import time
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import importlib.metadata  # para capturar versões das bibliotecas

# ===================== CONFIGURAÇÃO INICIAL =====================
def set_seed(seed=42):
    """Define sementes para reprodutibilidade total."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Detecção automática de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # adaptação para CPU

def setup_logging(log_file: str = "sumarizacao.log"):
    """Configura logging para arquivo e console."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # <-- ADICIONE ESTA LINHA
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# ===================== CLASSE PRINCIPAL =====================
class LongDocumentSummarizer:
    """
    Classe para sumarização de documentos longos usando modelos sequence-to-sequence.
    Suporta modelos como LED (com atenção global) e outros.
    """
    def __init__(self, model_name: str, device: str = device):
        """
        Inicializa o tokenizador e o modelo.

        Args:
            model_name (str): Identificador do modelo no Hugging Face Hub.
            device (str): Dispositivo ('cuda' ou 'cpu').
        """
        self.model_name = model_name
        self.device = device

        logging.info(f"Carregando tokenizador e modelo {model_name} em {device} (dtype={dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Carregamento otimizado: float16 na GPU, float32 na CPU
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            #use_safetensors=True #Desligar a exigência para rodar o allenai/led-base-16384
        ).to(self.device)
        self.model.eval()

    def _create_global_attention_mask(self, input_ids):
        """
        Cria a máscara de atenção global para o modelo LED.
        A atenção global é aplicada apenas no primeiro token (<s>).
        """
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        return global_attention_mask

    def summarize(self, text: str, num_beams: int = 4, length_penalty: float = 2.0,
                no_repeat_ngram_size: int = 3) -> dict:
        inicio_total = time.perf_counter()

        # Determina o limite máximo do encoder
        if hasattr(self.model.config, 'max_encoder_position_embeddings'):
            max_encoder_len = self.model.config.max_encoder_position_embeddings
        else:
            # Para BART, T5, Pegasus, etc.
            max_encoder_len = self.model.config.max_position_embeddings

        # Verifica se o texto será truncado
        tokens_contagem = self.tokenizer(text, truncation=False)['input_ids']
        truncado = len(tokens_contagem) > max_encoder_len

        # Tokenização com truncamento
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=max_encoder_len
        ).to(self.device)

        input_ids = inputs["input_ids"]
        tokens_entrada = input_ids.shape[1]

        if tokens_entrada < 50:
            # Retorno rápido (sem gerar resumo)
            return {
                "resumo": "",
                "tokens_entrada": tokens_entrada,
                "tokens_saida": 0,
                "tempo_execucao": 0.0,
                "vram_max_mb": 0.0,
                "compression_ratio": 0.0,
                "throughput_tokens_s": 0.0,
                "latencia_token_s": 0.0,
                "throughput_total_tokens_s": 0.0,
                "memory_efficiency_tokens_mb": 0.0,
                "truncado": truncado
            }

        # 3. Configurar máscara global
        global_attention_mask = None
        if "led" in self.model_name.lower():
            global_attention_mask = self._create_global_attention_mask(input_ids)

        # 4. Determinar tamanho alvo
        alvo_min = max(50, int(tokens_entrada * 0.08))
        alvo_max = min(1024, int(tokens_entrada * 0.12))

        generation_kwargs = {
            "min_new_tokens": alvo_min,
            "max_new_tokens": alvo_max,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "early_stopping": True,
            "attention_mask": inputs["attention_mask"]
        }
        if global_attention_mask is not None:
            generation_kwargs["global_attention_mask"] = global_attention_mask

        # Métricas de VRAM (se GPU)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            vram_inicio = torch.cuda.memory_allocated(self.device)

        # Geração
        try:
            with torch.no_grad():
                summary_ids = self.model.generate(input_ids, **generation_kwargs)
        except torch.cuda.OutOfMemoryError:
            logging.error("OOM durante a geração. Pulando amostra.")
            del inputs, input_ids, global_attention_mask
            torch.cuda.empty_cache()
            gc.collect()
            return None

        # Decodificação
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        tokens_saida = summary_ids.shape[1]

        # --- FIM DA MEDIÇÃO ---
        fim_total = time.perf_counter()
        tempo_execucao = fim_total - inicio_total

        # Pico de VRAM (se GPU)
        vram_max_mb = 0
        if self.device == "cuda":
            vram_max = torch.cuda.max_memory_allocated(self.device)
            vram_max_mb = vram_max / (1024 ** 2)

        # Métricas derivadas
        latencia_token = tempo_execucao / tokens_saida if tokens_saida > 0 else 0
        compression_ratio = (tokens_saida / tokens_entrada) * 100
        throughput_tokens_s = tokens_saida / tempo_execucao if tempo_execucao > 0 else 0
        tokens_total = tokens_entrada + tokens_saida
        throughput_total = tokens_total / tempo_execucao
        memory_efficiency = (tokens_total / vram_max_mb) if vram_max_mb > 0 else 0

        # Limpeza
        del inputs, input_ids, global_attention_mask, summary_ids
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "resumo": summary_text,
            "tokens_entrada": tokens_entrada,
            "tokens_saida": tokens_saida,
            "tempo_execucao": tempo_execucao,
            "vram_max_mb": round(vram_max_mb, 2),
            "compression_ratio": compression_ratio,
            "throughput_tokens_s": throughput_tokens_s,
            "latencia_token_s": latencia_token,
            "throughput_total_tokens_s": throughput_total,
            "memory_efficiency_tokens_mb": memory_efficiency,
            "truncado": truncado
        }

# ===================== PROCESSAMENTO DO DATASET =====================
def process_dataset(model_name: str, dataset_name: str, output_file: str,
                    num_samples: int = 10, seed: int = 42):
    """
    Itera sobre o dataset, gera resumos e salva incrementalmente em JSONL.
    Também salva metadados da execução em um arquivo separado.
    """
    summarizer = LongDocumentSummarizer(model_name)
    logging.info(f"Carregando dataset: {dataset_name} (split=test)")

    dataset = load_dataset(dataset_name, split="test")

    samples = dataset.select(range(num_samples))

    # Garantir que o diretório do arquivo de saída existe
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Diretório criado: {output_dir}")

    # Prepara arquivo de saída (sobrescreve se existir)
    with open(output_file, 'w', encoding='utf-8') as f:
        pass

    num_params = sum(p.numel() for p in summarizer.model.parameters())

    # Coleta metadados da execução
    metadados = {
        "modelo": model_name,
        "num_parametros": num_params,
        "device": device,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "seed": seed,
        "geracao": {
            "num_beams": 4,
            "length_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "min_ratio": 0.08,
            "max_ratio": 0.12,
            "max_absolute_tokens": 1024
        },
        "data": datetime.now().isoformat(),
        "bibliotecas": {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "numpy": importlib.metadata.version("numpy"),
        }
    }

    # Salva metadados em arquivo JSON (mesmo nome base + _metadados.json)
    meta_file = output_file.replace(".jsonl", "_metadados.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    logging.info(f"Metadados salvos em {meta_file}")

    logging.info("Iniciando sumarização com salvamento incremental...")

    for i, example in enumerate(tqdm(samples, desc="Gerando Resumos")):
        report_text = example["report"]

        # Tratamento de exceções para cada amostra
        try:
            resultado = summarizer.summarize(report_text)
            if resultado is None:
                raise RuntimeError("Falha na sumarização (OOM ou erro interno)")
        except Exception as e:
            logging.error(f"Erro na amostra {i+1}: {e}")
            registro = {
                "id_amostra": i + 1,
                "erro": str(e),
                "texto_original": report_text,
                "resumo_gerado": None,
                "tokens_entrada": None,
                "tokens_gerados": None,
                "truncado": None
            }
        else:
            # Sucesso: monta registro completo
            registro = {
                "id_amostra": i + 1,
                "tokens_entrada": resultado["tokens_entrada"],
                "tokens_gerados": resultado["tokens_saida"],
                "tempo_execucao_segundos": round(resultado["tempo_execucao"], 3),
                "vram_max_mb": resultado["vram_max_mb"],
                "compression_ratio(%)": round(resultado["compression_ratio"], 4),
                "throughput_tokens_s": round(resultado["throughput_tokens_s"], 2),
                "latencia_token_s": round(resultado["latencia_token_s"], 6),
                "throughput_total_tokens_s": round(resultado["throughput_total_tokens_s"], 2),
                "memory_efficiency_tokens_mb": round(resultado["memory_efficiency_tokens_mb"], 4),
                "truncado": resultado["truncado"],
                "texto_original": report_text,
                "resumo_gerado": resultado["resumo"]
            }

        # Salvamento incremental (append)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    logging.info(f"Processamento concluído. Resultados salvos em {output_file}")


# ===================== PONTO DE ENTRADA COM ARGPARSE =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sumarização de documentos longos com LED.")
    parser.add_argument("--modelo", type=str, default="allenai/led-base-16384",
                        help="Nome do modelo no Hugging Face Hub")
    parser.add_argument("--dataset", type=str, default="ccdv/govreport-summarization",
                        help="Nome do dataset (split 'test' será usado)")
    parser.add_argument("--amostras", type=int, default=12,
                        help="Número de amostras a processar")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente aleatória para reprodutibilidade")
    parser.add_argument("--output", type=str, default="Resultados_Sumarização/resultados.jsonl",
                        help="Arquivo de saída (JSONL). Se não fornecido, será gerado automaticamente.")
    args = parser.parse_args()

    # Define seed
    set_seed(args.seed)

    # Gera nome do arquivo de saída se não fornecido
    if args.output is None:
        modelo_seguro = args.modelo.replace("/", "-")
        args.output = f"resultados_sumarizacao_{modelo_seguro}_Amostras_{args.amostras}.jsonl"

    # Executa processamento
    process_dataset(
    model_name=args.modelo,
    dataset_name=args.dataset,
    output_file=args.output,
    num_samples=args.amostras,
    seed=args.seed,
    #indices_fixos=args.indices  # ou carregar de um arquivo
)