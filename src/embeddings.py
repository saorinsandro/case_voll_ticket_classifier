"""
Módulo responsável por gerar embeddings semânticos dos tickets.

Embeddings transformam texto em vetores numéricos que capturam significado,
permitindo medir similaridade, agrupar tickets e construir classificadores.
"""

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME


class EmbeddingGenerator:
    """
    Wrapper em torno do SentenceTransformer para geração de embeddings.

    Centralizar esse código permite trocar o modelo ou ajustar parâmetros
    sem impactar o restante do pipeline.
    """

    def __init__(self):
        # Carrega o modelo Transformer definido em config.py
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def encode(self, texts):
        """
        Recebe uma lista de textos (tickets) e retorna seus embeddings.

        Parâmetros:
            texts (List[str]): lista de textos a serem transformados

        Retorno:
            np.ndarray: matriz (n_textos, dimensão_embedding)
        """
        return self.model.encode(texts, show_progress_bar=True)
