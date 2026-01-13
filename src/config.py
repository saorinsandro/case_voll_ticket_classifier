"""
Arquivo de configuração central do projeto.

Aqui ficam parâmetros globais que controlam o comportamento do pipeline,
como o modelo de embeddings e thresholds de decisão.

Manter esses valores centralizados facilita ajuste, versionamento e experimentação.
"""

# Modelo Transformer usado para gerar embeddings semânticos dos tickets
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Threshold mínimo de similaridade para aceitar uma classificação automática.
# Se a similaridade for menor que esse valor, o ticket pode ser enviado para triagem humana.
SIMILARITY_THRESHOLD = 0.6
