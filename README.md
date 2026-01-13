# VOLL – Sistema de Classificação Inteligente de Tickets de Atendimento

## Autor
Sandro Saorin da Silva

Este repositório apresenta uma solução completa de classificação e roteamento automático de tickets de atendimento para a VOLL, desenvolvida como parte de um case técnico para a vaga de Cientista de Dados Sênior.

A solução cobre todo o ciclo de um sistema de Machine Learning em produção:
- Descoberta de classes a partir de dados não rotulados
- Criação de um dataset pseudo-rotulado
- Treinamento de um classificador supervisionado
- Disponibilização via API REST

---

## Objetivo

Classificar automaticamente tickets de atendimento, a partir do assunto e do corpo do e-mail, em categorias operacionais que permitam o roteamento para o time correto, como:

- Infraestrutura de Rede
- Suporte a Impressoras
- Dispositivos Eletrônicos
- Cloud & Serviços Digitais
- Loja Online e Casos Especiais

---

## Visão Geral da Solução

O projeto simula um cenário real de empresa onde inicialmente não existem rótulos para os tickets. A solução evolui em três fases:

1. Descoberta semântica (_Unsupervised Learning_)
Os textos dos tickets são transformados em embeddings usando Sentence-BERT. Os embeddings são reduzidos com UMAP e os grupos semânticos são descobertos usando HDBSCAN. Isso permite identificar automaticamente temas operacionais reais dentro dos tickets.

2. Criação de rótulos e dataset supervisionado
Os clusters descobertos são interpretados e mapeados para classes operacionais reais, criando um dataset pseudo-rotulado salvo em:
data/processed/tickets_labeled.csv

3. Treinamento de um classificador supervisionado
Com os rótulos definidos, treinamos um modelo de classificação real usando:
- Embeddings Sentence-BERT (all-MiniLM-L6-v2)
- Classificador Logistic Regression
- Validação via cross-validation, F1-score e ROC AUC
- Comparação com Random Forest e Gradient Boosting

O melhor modelo é treinado em todo o dataset e salvo em:
models/ticket_classifier.joblib  
models/embedding_model.joblib  

---

## Arquitetura do Sistema

Texto do ticket  
→ Limpeza e normalização (utils.py)  
→ Geração de embeddings (embeddings.py)  
→ Classificador supervisionado (classifier.py)  
→ Classe + probabilidade + decisão de triagem  

---

## API REST (FastAPI)

A API expõe o modelo treinado.

Endpoint:
POST /classify

Exemplo de requisição:

{
  "text": "Minha fatura do Google Workspace veio com valor errado"
}

Exemplo de resposta:

{
  "predicted_class": "Cloud & Serviços Digitais",
  "confidence": 0.94,
  "needs_human_review": false,
  "probabilities": {
    "Infraestrutura de Rede": 0.01,
    "Suporte a Impressoras": 0.00,
    "Dispositivos Eletrônicos": 0.02,
    "Cloud & Serviços Digitais": 0.94,
    "Loja Online e Casos Especiais": 0.03
  }
}

Tickets com baixa confiança são automaticamente encaminhados para triagem humana.

---

## Monitoramento em Produção

O sistema foi projetado para operação real com monitoramento de:
- Distribuição de classes
- Taxa de baixa confiança
- Drift nos embeddings
- Mudanças no perfil dos tickets

---

## Como Executar

1. Criar ambiente virtual:
```bash
python -m venv .venv  
.venv\Scripts\activate  
```

2. Instalar dependências:
```bash
pip install -r requirements.txt  
```

3. Rodar notebooks:
```bash
notebooks/
```

4. Subir a API:
```bash
uvicorn api:app --reload  
```

Acessar:
http://127.0.0.1:8000/docs

---

## Conclusão

Este projeto demonstra uma abordagem realista, escalável e industrial para classificação de tickets, cobrindo NLP moderno, clustering semântico, classificação supervisionada, governança por confiança e deploy via API.
