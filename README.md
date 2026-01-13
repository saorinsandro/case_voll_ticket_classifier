# VOLL – Sistema de Classificação de Tickets de Atendimento

## Autor: Sandro Saorin da Silva

Este repositório apresenta uma solução completa para classificação e roteamento automático de tickets de atendimento da VOLL, desenvolvida como parte de um case técnico para a vaga de Cientista de Dados Sênior.

O desafio simula um cenário de classificação de tickets de atendimento, onde a solução proposta utiliza Processamento de Linguagem Natural (NLP) e embeddings semânticos para permitir que os tickets sejam classificados desde o primeiro dia, com uma estratégia clara de evolução para modelos supervisionados.

---

## Objetivo do Projeto

Classificar automaticamente tickets de atendimento, a partir do assunto e do corpo dos e-mails, em categorias operacionais que permitam direcionar cada solicitação ao time mais adequado, como:

- Financeiro (reembolsos, cobranças)
- Suporte técnico (login, erros, app)
- Operações de viagem (voos, hotéis, reservas)
- Políticas corporativas
- Administração e cadastro de usuários

---

## Abordagem

Dado o volume extremamente reduzido de dados e a ausência de rótulos, foi adotada uma abordagem baseada em representações semânticas em vez de aprendizado supervisionado tradicional.

A solução funciona em quatro etapas principais:

1. Pré-processamento do texto (assunto + corpo do e-mail)
2. Geração de embeddings semânticos usando um modelo Transformer (Sentence-BERT)
3. Descoberta e definição de classes a partir de agrupamentos semânticos
4. Classificação de novos tickets por similaridade

Novos tickets são comparados semanticamente aos tickets de referência. A classe do ticket mais similar é atribuída, juntamente com um score de confiança.

Quando a similaridade é baixa, o sistema indica que o ticket deve ser enviado para triagem humana, evitando decisões automáticas incorretas.

---

## Arquitetura do Projeto

voll-ticket-classifier/
data/            Dados brutos e processados  
notebooks/       Análise exploratória e validação semântica  
src/             Pipeline de classificação  
models/          Artefatos do modelo  
reports/         Documentação e explicação da solução  

Além do notebook, a solução inclui uma API REST (FastAPI) que expõe o classificador para uma simulação de uso em produção.

---

## Avaliação e Monitoramento

Como não existem rótulos verdadeiros, a qualidade do sistema é avaliada por meio de:

- Coerência semântica dos clusters
- Validação humana de amostras
- Taxa de reatribuição manual
- SLA por classe de atendimento

Em produção, também devem ser monitorados:

- Drift nos embeddings
- Distribuição de tickets por categoria
- Taxa de baixa confiança (triagem humana)

---

## Evolução do Sistema

À medida que tickets são corrigidos manualmente, esses dados passam a formar um conjunto rotulado.

Com isso, o sistema pode evoluir de um roteador baseado em similaridade para um modelo supervisionado, utilizando estratégias de *active learning* para priorizar os tickets mais informativos.

---

## Como Executar

1. Criar e ativar o ambiente virtual:
```bash
python -m venv .venv  
.venv\Scripts\activate  
```

2. Instalar dependências:
```bash
pip install -r requirements.txt  
```

3. Rodar o notebook de exploração:
```bash
notebooks/01_exploration.ipynb  
```

4. Iniciar a API:
```bash
uvicorn api:app --reload  
```

A API ficará disponível em:  
http://127.0.0.1:8000  

Documentação interativa:  
http://127.0.0.1:8000/docs
