"""
API REST (FastAPI) para classificação e roteamento de tickets.

Objetivo:
- Expor um endpoint HTTP simples que recebe um texto de ticket
- Retornar a classe prevista, um score de similaridade (confiança)
- Indicar quando o ticket deve ir para triagem humana (baixa confiança)

Observação importante:
- Como estamos em cenário de cold start (poucos tickets e sem rótulos),
  a classificação é feita por similaridade semântica (protótipos).
- Em produção, os protótipos podem ser atualizados com feedback humano
  e/ou substituídos por um classificador supervisionado.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import load_and_prepare_data, generate_embeddings
from src.classifier import TicketClassifier
from src.embeddings import EmbeddingGenerator
from src.config import SIMILARITY_THRESHOLD

# -----------------------------------------------------------------------------
# Inicialização da aplicação FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="VOLL Ticket Classifier",
    description="API para classificação e roteamento de tickets de atendimento",
    version="1.0",
)

# -----------------------------------------------------------------------------
# Inicialização do modelo ao subir a API
# -----------------------------------------------------------------------------
# Para reduzir latência em requests, carregamos e “preparamos” tudo no startup:
# - Leitura do dataset de referência
# - Geração de embeddings
# - Instanciação do classificador
# - Instanciação do gerador de embeddings para tickets novos
#
# Isso evita recomputar embeddings e recarregar modelos a cada request.
df = load_and_prepare_data("data/raw/classificacao_atendimento.csv")
embeddings = generate_embeddings(df)

# Classes iniciais (seed) definidas a partir da análise exploratória.
# Em um cenário real, essas classes seriam ajustadas conforme o atendimento
# e/ou conforme clusters maiores aparecessem nos dados.
CLASS_LABELS = [
    "Financeiro",
    "Suporte Técnico",
    "Políticas e Compliance",
    "Operações de Viagem",
    "Administração",
]

# Garantia de robustez: se o dataset tiver menos linhas que labels,
# evitamos mismatch de tamanho.
CLASS_LABELS = CLASS_LABELS[: len(df)]

# Classificador baseado em protótipos: compara ticket novo com tickets conhecidos
# e retorna a classe do mais similar.
classifier = TicketClassifier(
    reference_embeddings=embeddings,
    reference_labels=CLASS_LABELS,
)

# Encoder responsável por gerar embedding do texto do ticket que chega na API.
encoder = EmbeddingGenerator()

# -----------------------------------------------------------------------------
# Schemas (contratos de entrada e saída)
# -----------------------------------------------------------------------------
# Usamos Pydantic para validar e documentar os formatos (Swagger automático).


class TicketRequest(BaseModel):
    """
    Payload esperado no endpoint /classify.
    """

    text: str


class TicketResponse(BaseModel):
    """
    Payload retornado pelo endpoint /classify.

    Campos:
    - predicted_class: classe prevista
    - similarity: similaridade (confiança) com o protótipo mais próximo
    - needs_human_review: flag para triagem humana quando confiança é baixa
    """

    predicted_class: str
    similarity: float
    needs_human_review: bool


# -----------------------------------------------------------------------------
# Endpoint principal
# -----------------------------------------------------------------------------
@app.post("/classify", response_model=TicketResponse)
def classify_ticket(ticket: TicketRequest):
    """
    Classifica um ticket recebido via API.

    Passos:
    1) Gerar embedding do texto recebido
    2) Calcular similaridade vs. embeddings de referência
    3) Retornar classe do protótipo mais próximo e score de similaridade
    4) Se a similaridade ficar abaixo do threshold, marcar para revisão humana
    """
    emb = encoder.encode([ticket.text])[0]
    label, score = classifier.predict(emb)

    return {
        "predicted_class": label,
        "similarity": score,
        "needs_human_review": score < SIMILARITY_THRESHOLD,
    }
