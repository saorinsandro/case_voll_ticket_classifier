"""
Módulo responsável pela classificação de novos tickets.

Este classificador usa um modelo supervisionado treinado
(Logistic Regression) sobre embeddings de texto para prever
a classe de cada ticket.

Ele retorna:
- Classe prevista
- Vetor de probabilidades por classe
- Score de confiança (maior probabilidade)
"""

import joblib
import numpy as np
from typing import List


class TicketClassifier:
    """
    Wrapper para o classificador supervisionado treinado.

    Este objeto carrega o modelo salvo em disco (.joblib)
    e fornece uma interface simples para predição.
    """

    def __init__(self, model_path: str, class_labels: List[str]):
        """
        Parâmetros:
            model_path (str): caminho para o arquivo .joblib do modelo treinado
            class_labels (List[str]): lista de nomes das classes, na mesma ordem
                                     usada no treinamento
        """
        self.model = joblib.load(model_path)
        self.class_labels = class_labels

    def predict(self, embedding: np.ndarray):
        """
        Classifica um novo ticket a partir do embedding.

        Parâmetros:
            embedding (np.ndarray): vetor do ticket (1D)

        Retorna:
            predicted_class (str)
            confidence (float)
            probabilities (dict)
        """
        probs = self.model.predict_proba([embedding])[0]
        idx = int(np.argmax(probs))

        predicted_class = self.class_labels[idx]
        confidence = float(probs[idx])

        prob_dict = {
            self.class_labels[i]: float(probs[i]) for i in range(len(self.class_labels))
        }

        return predicted_class, confidence, prob_dict
