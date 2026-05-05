import pandas as pd

# Dicionário com os modelos e seus arquivos
model_files = {
    'bart-large': 'caminho para o arquivo',
    'PRIMERA': 'caminho para o arquivo',
    'LED': 'caminho para o arquivo',
}

dfs = []

# Carregar e padronizar
for model_name, file_path in model_files.items():
    df = pd.read_excel(file_path)

    # Remover NaN
    df = df.dropna(subset=['supported', 'contradicted', 'not_supported']).copy()

    # Adicionar nome do modelo
    df['model'] = model_name

    # Total de claims
    df['total_claims'] = df['supported'] + df['contradicted'] + df['not_supported']

    # Proporções
    df['supported_ratio'] = df['supported'] / df['total_claims']
    df['contradicted_ratio'] = df['contradicted'] / df['total_claims']
    df['not_supported_ratio'] = df['not_supported'] / df['total_claims']

    dfs.append(df)

# Juntar tudo
df_all = pd.concat(dfs, ignore_index=True)

# Status
stats_status = df_all.groupby('model')[
    ['supported_ratio', 'contradicted_ratio', 'not_supported_ratio']
].agg(['mean', 'std']).round(4)

print(stats_status)

score_stats = df_all.groupby('model')['score'].agg(['mean', 'std']).round(4)
print("\nScore FENICE:")
print(score_stats)

error_cols = ['predicate_errors', 'entity_errors', 'coref_errors', 'other_erros']

erros_modelo = df_all.groupby('model')[error_cols].sum()

# Proporção dentro de cada modelo
proporcoes_erros = erros_modelo.div(erros_modelo.sum(axis=1), axis=0).round(4)

print(proporcoes_erros)


compostas = ['evidence_score', 'faithfulness_score', 'stability_score', 'reliability_score']

stats_comp = df_all.groupby('model')[compostas].agg(['mean', 'std']).round(4)
print(stats_comp)

baseline = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bert_score_f1']

stats_baseline = df_all.groupby('model')[baseline].agg(['mean', 'std']).round(4)
print(stats_baseline)


posicao = ['coverage_ratio', 'mean_position', 'start_ratio', 'middle_ratio', 'end_ratio']

stats_pos = df_all.groupby('model')[posicao].agg(['mean', 'std']).round(4)
print(stats_pos)


eficiencia = ['execution_time', 'vram_max_mb']

stats_eff = df_all.groupby('model')[eficiencia].agg(['mean', 'std']).round(2)
print(stats_eff)