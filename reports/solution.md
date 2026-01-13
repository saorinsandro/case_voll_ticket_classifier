# VOLL – Solução de Classificação de Tickets de Atendimento

## 1. Contexto

O conjunto de dados fornecido contém milhares de tickets de atendimento não rotulados. Esse cenário representa uma situação comum em empresas reais: existe grande volume de dados, mas sem rótulos estruturados para treinamento supervisionado.

O desafio é transformar esse histórico de textos em um sistema capaz de classificar automaticamente novos tickets em categorias operacionais, permitindo seu roteamento para os times corretos.

A solução foi desenhada para resolver três problemas principais:
- Descobrir automaticamente os tipos de tickets existentes
- Criar um conjunto de dados rotulado
- Construir um classificador escalável para produção

---

## 2. Descoberta das Classes

Como não existiam rótulos, foi aplicada uma abordagem de clustering semântico:

1. Os textos dos tickets foram transformados em embeddings usando Sentence-BERT
2. Os embeddings foram reduzidos com UMAP
3. Os grupos semânticos foram descobertos usando HDBSCAN

Isso permitiu identificar padrões naturais nos dados sem qualquer supervisão.

A análise dos clusters revelou cinco grandes temas operacionais:

| Cluster | Classe Operacional |
|--------|-------------------|
| 0 | Infraestrutura de Rede |
| 1 | Suporte a Impressoras |
| 2 | Dispositivos Eletrônicos |
| 3 | Cloud & Serviços Digitais |
| -1 | Loja Online e Casos Especiais |

Essas classes refletem diretamente áreas de suporte típicas em serviços e operações de tecnologia.

---

## 3. Criação do Dataset Rotulado

Após a identificação dos clusters, cada ticket foi associado à sua classe correspondente, criando um dataset pseudo-rotulado com milhares de exemplos.

Esse dataset foi salvo em:
- data/processed/tickets_labeled.csv

Ele passa a ser a base oficial para treinamento de modelos supervisionados e evolução contínua.

---

## 4. Treinamento do Classificador

Com o dataset rotulado, foi treinado um classificador supervisionado com a seguinte arquitetura:

- Representação textual: Sentence-BERT (all-MiniLM-L6-v2)
- Classificador: Logistic Regression

Foram comparados três algoritmos:
- Logistic Regression
- Random Forest
- Gradient Boosting

A avaliação foi feita usando:
- Cross-validation com F1-score ponderado
- ROC AUC multiclasse (One-vs-Rest)

O Logistic Regression apresentou o melhor equilíbrio entre performance e estabilidade, sendo escolhido como modelo final.

Os artefatos finais foram salvos em:
- models/ticket_classifier.joblib
- models/embedding_model.joblib

---

## 5. Classificação em Produção

Para cada novo ticket, o sistema executa:

1. Limpeza e normalização do texto (contrato de pré-processamento)
2. Geração de embedding semântico
3. Aplicação do classificador supervisionado treinado
4. Retorno da classe prevista e probabilidades por classe

Além da classe, o sistema retorna um score de confiança (probabilidade máxima).

Se a confiança ficar abaixo de um threshold configurável, o ticket é marcado para triagem humana, evitando decisões automáticas incorretas.

---

## 6. Governança e Monitoramento

Em produção, recomenda-se monitorar:

- Distribuição de tickets por classe ao longo do tempo
- Taxa de baixa confiança (tickets enviados para triagem humana)
- Drift nos embeddings / mudanças no vocabulário
- Mudança no perfil de temas (surgimento de novos tipos de ticket)

Esses indicadores ajudam a detectar degradação de performance e necessidade de atualização do modelo.

---

## 7. Estratégia de Retreino

Tickets enviados para triagem humana podem ser rotulados por analistas e incorporados ao dataset, permitindo:

- Retreino periódico (ex: semanal/mensal)
- Refinamento das classes existentes
- Criação de novas categorias quando necessário

Isso transforma o sistema em um pipeline de aprendizado contínuo.

---

## 8. Active Learning

O sistema implementa implicitamente uma estratégia de active learning:

- Tickets com baixa confiança são priorizados para revisão humana
- Esses exemplos são os mais informativos para o modelo
- O retreino com esses dados maximiza o ganho de performance com o menor custo de rotulagem

---

## 9. Conclusão

A solução implementa um pipeline completo e realista de Machine Learning para classificação de tickets:

- Descoberta automática de classes via clustering semântico
- Construção de dataset pseudo-rotulado
- Treinamento supervisionado com avaliação comparativa de modelos
- Governança por confiança e triagem humana
- Exposição via API REST para uso operacional

O design é alinhado a práticas modernas de empresas SaaS e permite evolução contínua conforme novos dados e feedback humano se acumulam.
