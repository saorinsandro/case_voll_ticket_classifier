"""
API REST (FastAPI) para classificação e roteamento de tickets.

Esta API expõe um endpoint que:
- Recebe um texto de ticket
- Gera embeddings semânticos
- Usa um classificador supervisionado treinado
- Retorna classe, confiança e indicação de triagem humana

Este é o mesmo pipeline usado no main.py e no treino,
garantindo consistência entre todos os ambientes.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.embeddings import EmbeddingGenerator
from src.classifier import TicketClassifier
from src.config import (
    CLASSIFIER_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)

# ------------------------------------------------------------------
# Inicialização da aplicação
# ------------------------------------------------------------------
app = FastAPI(
    title="VOLL Ticket Classifier",
    description="API para classificação e roteamento de tickets de atendimento",
    version="2.0",
)

# ------------------------------------------------------------------
# Classes (mesma ordem do treino)
# ------------------------------------------------------------------
CLASS_LABELS = [
    "Infraestrutura de Rede",
    "Suporte a Impressoras",
    "Dispositivos Eletrônicos",
    "Cloud & Serviços Digitais",
    "Loja Online e Casos Especiais",
]

# ------------------------------------------------------------------
# Inicialização do modelo (carregado uma única vez no startup)
# ------------------------------------------------------------------
encoder = EmbeddingGenerator()
classifier = TicketClassifier(CLASSIFIER_MODEL_PATH, CLASS_LABELS)


# ------------------------------------------------------------------
# Schemas de entrada e saída
# ------------------------------------------------------------------
class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    predicted_class: str
    confidence: float
    needs_human_review: bool
    probabilities: dict


# ------------------------------------------------------------------
# Endpoint principal
# ------------------------------------------------------------------
@app.post("/classify", response_model=TicketResponse)
def classify_ticket(ticket: TicketRequest):
    """
    Classifica um ticket recebido via API.

    Passos:
    1) Gerar embedding do texto
    2) Aplicar o classificador treinado
    3) Retornar classe, confiança e distribuição de probabilidades
    """
    embedding = encoder.encode([ticket.text])[0]

    predicted_class, confidence, probabilities = classifier.predict(embedding)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "needs_human_review": confidence < CONFIDENCE_THRESHOLD,
        "probabilities": probabilities,
    }
