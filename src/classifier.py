"""
Módulo responsável pela classificação de novos tickets.

Este classificador funciona por similaridade:
um novo ticket é comparado aos tickets de referência
e recebe a classe do mais parecido.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TicketClassifier:
    """
    Classificador baseado em protótipos.

    Ele armazena embeddings de tickets de referência e suas classes,
    e usa similaridade de cosseno para classificar novos tickets.
    """

    def __init__(self, reference_embeddings, reference_labels):
        """
        Parâmetros:
            reference_embeddings (np.ndarray): embeddings dos tickets conhecidos
            reference_labels (List[str]): classes desses tickets
        """
        self.reference_embeddings = reference_embeddings
        self.reference_labels = reference_labels

    def predict(self, embedding):
        """
        Classifica um novo ticket a partir de seu embedding.

        Retorna:
        - classe prevista
        - score de similaridade

        Se o score for baixo, o ticket pode ser enviado para triagem humana.
        """
        sims = cosine_similarity([embedding], self.reference_embeddings)[0]
        idx = np.argmax(sims)

        return self.reference_labels[idx], float(sims[idx])
