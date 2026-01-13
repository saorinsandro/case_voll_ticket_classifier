"""
Arquivo de configuração central do projeto.

Aqui ficam parâmetros globais que controlam:
- Modelo de embeddings
- Caminhos dos artefatos treinados
- Threshold de confiança para triagem humana

Esses valores permitem controlar o comportamento do sistema
sem alterar código.
"""

# ---------------------------------------------------------
# Modelo de embeddings
# ---------------------------------------------------------
# Usado tanto no treino quanto na inferência
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------
# Caminhos dos modelos treinados
# ---------------------------------------------------------
EMBEDDING_MODEL_PATH = "models/embedding_model.joblib"
CLASSIFIER_MODEL_PATH = "models/ticket_classifier.joblib"

# ---------------------------------------------------------
# Threshold de confiança
# ---------------------------------------------------------
# Se a probabilidade máxima do classificador for menor que isso,
# o ticket deve ser enviado para triagem humana.
CONFIDENCE_THRESHOLD = 0.60
