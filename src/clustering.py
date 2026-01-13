"""
Módulo responsável pela descoberta de grupos (clusters) de tickets.

Clusters representam padrões semânticos naturais nos textos,
que depois são traduzidos em classes operacionais.
"""

from sklearn.cluster import KMeans


class TicketClustering:
    """
    Classe para clusterização de embeddings de tickets.

    Nesta implementação usamos KMeans por simplicidade, mas em produção
    HDBSCAN ou outros métodos poderiam ser utilizados.
    """

    def __init__(self, n_clusters: int):
        """
        Parâmetros:
            n_clusters (int): número de clusters a serem formados
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, embeddings):
        """
        Ajusta o modelo de clustering e retorna o rótulo de cluster
        para cada embedding.

        Parâmetros:
            embeddings (np.ndarray): matriz de embeddings dos tickets

        Retorno:
            np.ndarray: vetor de clusters
        """
        return self.model.fit_predict(embeddings)
