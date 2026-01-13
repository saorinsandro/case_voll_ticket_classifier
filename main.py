"""
Script de execução local do classificador de tickets.

Este arquivo permite testar o sistema via terminal usando
o mesmo modelo treinado que é usado pela API.

Fluxo:
- Carrega o modelo de embeddings congelado
- Carrega o classificador supervisionado
- Permite digitar tickets
- Retorna classe, probabilidade e indicação de triagem humana
"""

from src.embeddings import EmbeddingGenerator
from src.classifier import TicketClassifier
from src.config import (
    CLASSIFIER_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)

# Classes devem estar na mesma ordem usada no treino
CLASS_LABELS = [
    "Infraestrutura de Rede",
    "Suporte a Impressoras",
    "Dispositivos Eletrônicos",
    "Cloud & Serviços Digitais",
    "Loja Online e Casos Especiais",
]


def main():
    print("\n=== VOLL | Sistema de Classificação de Tickets ===\n")

    # ------------------------------------------------------------------
    # 1. Inicializa o encoder e o classificador treinado
    # ------------------------------------------------------------------
    encoder = EmbeddingGenerator()
    classifier = TicketClassifier(CLASSIFIER_MODEL_PATH, CLASS_LABELS)

    # ------------------------------------------------------------------
    # 2. Loop interativo
    # ------------------------------------------------------------------
    while True:
        print("\nDigite um novo ticket (ou ENTER para sair):")
        user_text = input("> ")

        if not user_text.strip():
            print("\nEncerrando o classificador.")
            break

        # Gera embedding do texto digitado
        emb = encoder.encode([user_text])[0]

        # Classifica o ticket usando o modelo supervisionado
        label, confidence, probabilities = classifier.predict(emb)

        # Exibe resultado
        print("\nResultado:")
        print("Classe prevista:", label)
        print("Confiança:", round(confidence, 3))

        # Mostra probabilidades por classe (útil para debug e análise)
        print("\nDistribuição de probabilidades:")
        for cls, prob in probabilities.items():
            print(f" - {cls}: {round(prob, 3)}")

        # Indicação de triagem humana
        if confidence < CONFIDENCE_THRESHOLD:
            print("\nBaixa confiança: encaminhar para triagem humana.")


if __name__ == "__main__":
    main()
