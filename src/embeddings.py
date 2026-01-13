"""
Módulo responsável por gerar embeddings semânticos dos tickets.

Este módulo carrega o modelo de embeddings congelado
usado durante o treinamento do classificador, garantindo
consistência entre treino e inferência.
"""

import joblib
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH


class EmbeddingGenerator:
    def __init__(self):
        """
        Tenta carregar o modelo salvo em disco (joblib).
        Se não existir (ex: ambiente de treino), carrega do HuggingFace.
        """
        try:
            self.model = joblib.load(EMBEDDING_MODEL_PATH)
        except Exception:
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def encode(self, texts):
        """
        Gera embeddings para uma lista de textos.

        Retorna:
            np.ndarray com shape (n_textos, dimensão_embedding)
        """
        return self.model.encode(texts, show_progress_bar=False)
