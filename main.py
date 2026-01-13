"""
Script de execução local do classificador de tickets.

Este arquivo permite testar o sistema via terminal, simulando
o uso do classificador em produção sem precisar usar a API.

Fluxo:
- Carrega os tickets de referência
- Gera embeddings
- Inicializa o classificador por similaridade
- Permite ao usuário digitar novos tickets
- Retorna a classe prevista e o nível de confiança
"""

from src.pipeline import load_and_prepare_data, generate_embeddings
from src.classifier import TicketClassifier
from src.embeddings import EmbeddingGenerator
from src.config import SIMILARITY_THRESHOLD


def main():
    print("\n=== VOLL | Sistema de Classificação de Tickets ===\n")

    # ------------------------------------------------------------------
    # 1. Carregamento dos tickets de referência
    # ------------------------------------------------------------------
    # Estes tickets funcionam como "protótipos" iniciais para cada classe.
    df = load_and_prepare_data("data/raw/classificacao_atendimento.csv")

    # ------------------------------------------------------------------
    # 2. Geração dos embeddings dos tickets conhecidos
    # ------------------------------------------------------------------
    embeddings = generate_embeddings(df)

    # ------------------------------------------------------------------
    # 3. Classes iniciais (seed)
    # ------------------------------------------------------------------
    # Essas classes foram definidas a partir da análise exploratória
    # e representam os times de atendimento da VOLL.
    class_labels = [
        "Financeiro",
        "Suporte Técnico",
        "Políticas e Compliance",
        "Operações de Viagem",
        "Administração",
    ]

    # Garante alinhamento entre número de tickets e número de labels
    class_labels = class_labels[: len(df)]

    # ------------------------------------------------------------------
    # 4. Inicialização do classificador por similaridade
    # ------------------------------------------------------------------
    classifier = TicketClassifier(
        reference_embeddings=embeddings,
        reference_labels=class_labels,
    )

    # Encoder para gerar embedding dos tickets digitados pelo usuário
    encoder = EmbeddingGenerator()

    # ------------------------------------------------------------------
    # 5. Loop interativo para classificação de novos tickets
    # ------------------------------------------------------------------
    while True:
        print("\nDigite um novo ticket (ou ENTER para sair):")
        user_text = input("> ")

        if not user_text.strip():
            print("\nEncerrando o classificador.")
            break

        # Gera embedding do texto digitado
        emb = encoder.encode([user_text])[0]

        # Classifica o ticket por similaridade
        label, score = classifier.predict(emb)

        # Exibe resultado
        print("\nResultado:")
        print("Classe prevista:", label)
        print("Confiança (similaridade):", round(score, 3))

        # Se a confiança for baixa, orienta triagem humana
        if score < SIMILARITY_THRESHOLD:
            print("Baixa confiança: encaminhar para triagem humana.")


if __name__ == "__main__":
    main()
