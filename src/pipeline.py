"""
Pipeline principal do sistema de classificação de tickets.

Responsável por:
- Carregar dados
- Construir o campo de texto unificado
- Aplicar limpeza
- Preparar o input do modelo de embeddings

Este pipeline é usado tanto no treino quanto na inferência (API),
garantindo consistência entre os ambientes.
"""

import pandas as pd
from .utils import clean_text


def build_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna 'text' combinando subject e body e aplica limpeza.

    Parâmetros:
        df (pd.DataFrame): DataFrame com colunas subject e body

    Retorno:
        DataFrame com coluna 'text'
    """
    df = df.copy()

    df["text"] = (
        df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    )

    df["text"] = df["text"].apply(clean_text)

    return df


def load_data(path: str) -> pd.DataFrame:
    """
    Carrega o CSV e aplica o pipeline de texto.

    Retorno:
        DataFrame pronto para embedding
    """
    df = pd.read_csv(path)
    df = build_text(df)
    return df
