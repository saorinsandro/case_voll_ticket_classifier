"""
Funções utilitárias para limpeza e pré-processamento de texto.

Este módulo define o contrato de normalização textual do sistema.
Qualquer alteração aqui exige re-treino do modelo.
"""

import re


def clean_text(text: str) -> str:
    """
    Normaliza o texto do ticket.

    Passos:
    - Converte para minúsculas
    - Remove caracteres especiais (mantendo acentos)
    - Normaliza espaços

    Parâmetros:
        text (str): texto bruto

    Retorno:
        str: texto limpo
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9À-ÿ ]", "", text)

    return text.strip()
