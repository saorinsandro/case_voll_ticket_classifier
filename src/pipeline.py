"""
Pipeline principal do sistema de classificação de tickets.

Este módulo conecta:
- carregamento de dados
- limpeza de texto
- geração de embeddings
"""

import pandas as pd
from .utils import clean_text
from .embeddings import EmbeddingGenerator


def load_and_prepare_data(path):
    """
    Carrega o CSV de tickets e prepara a coluna de texto unificada.

    Parâmetros:
        path (str): caminho para o arquivo CSV

    Retorno:
        DataFrame com coluna 'text' pronta para embedding
    """
    df = pd.read_csv(path)

    # Combina subject e body em um único campo
    df["text"] = (
        df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    )

    # Aplica limpeza básica
    df["text"] = df["text"].apply(clean_text)

    return df


def generate_embeddings(df):
    """
    Gera embeddings para a coluna 'text' do DataFrame.

    Parâmetros:
        df (pd.DataFrame): dataframe com coluna 'text'

    Retorno:
        np.ndarray com embeddings dos tickets
    """
    encoder = EmbeddingGenerator()
    embeddings = encoder.encode(df["text"].tolist())
    return embeddings
