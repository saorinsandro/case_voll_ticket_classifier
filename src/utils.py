"""
Funções utilitárias para limpeza e pré-processamento de texto.

Este módulo é responsável por normalizar o texto dos tickets antes de
enviá-los para o modelo de embeddings.
"""

import re


def clean_text(text: str) -> str:
    """
    Realiza limpeza básica de texto.

    Passos:
    - Converte para minúsculas
    - Remove caracteres especiais
    - Normaliza espaços

    Isso ajuda a reduzir ruído sem perder o significado semântico.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9À-ÿ ]", "", text)
    return text.strip()
