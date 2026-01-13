# VOLL – Ticket Classifier API

Esta API expõe o modelo de classificação de tickets via HTTP, permitindo que sistemas externos enviem tickets e recebam a classe prevista.

## Como iniciar a API

1. Ative o ambiente virtual:

.venv\Scripts\activate

2. Instale as dependências (se ainda não instalou):

pip install fastapi uvicorn

3. Inicie o servidor:

uvicorn api:app --reload

A API ficará disponível em:

http://127.0.0.1:8000

A documentação interativa (Swagger) pode ser acessada em:

http://127.0.0.1:8000/docs

## Endpoint

POST /classify

Classifica um ticket.

### Request (JSON)

{
  "text": "Não consigo acessar minha conta no aplicativo"
}

### Response

{
  "predicted_class": "Suporte Técnico",
  "similarity": 0.82,
  "needs_human_review": false
}

O campo "needs_human_review" indica quando a confiança do modelo é baixa e o ticket deve ser revisado por um humano.
