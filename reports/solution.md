# VOLL – Solução de Classificação de Tickets de Atendimento

## 1. Contexto

O conjunto de dados fornecido contém apenas cinco tickets sem rótulos. Esse cenário representa um problema realista de *cold start*, no qual o sistema de atendimento precisa operar sem histórico de dados rotulados.

Mesmo com poucos dados, a necessidade de negócio permanece: os tickets recebidos devem ser direcionados corretamente para os times de atendimento, reduzindo tempo de resposta e retrabalho.

A solução proposta foi desenhada para funcionar desde o primeiro dia e evoluir conforme mais dados e feedback humano se tornam disponíveis.

---

## 2. Classes Operacionais

A partir da análise do conteúdo dos tickets e do contexto de negócio da VOLL, foram definidas as seguintes classes operacionais iniciais:

| Classe | Descrição |
|-------|----------|
| Financeiro | Reembolsos, cobranças e pagamentos |
| Suporte Técnico | Login, erros no aplicativo, falhas de sistema |
| Políticas e Compliance | Regras corporativas de viagem |
| Operações de Viagem | Voos, hotéis e reservas |
| Administração | Cadastro, usuários e permissões |

Essas classes refletem diretamente a forma como plataformas de mobilidade corporativa estruturam seus times de atendimento.

---

## 3. Pipeline da Solução

O pipeline completo é composto pelas seguintes etapas:

1. Combinação e limpeza do texto (assunto + corpo do e-mail)
2. Geração de embeddings semânticos com modelo Transformer (Sentence-BERT)
3. Descoberta de padrões por meio de clusterização
4. Definição de protótipos por classe
5. Classificação de novos tickets por similaridade

Cada ticket é representado por um vetor semântico. Novos tickets são comparados aos tickets de referência e recebem a classe do mais similar.

Além da classe, o sistema retorna um score de confiança baseado na similaridade.

---

## 4. Tratamento de Incerteza e Novos Tipos de Tickets

Quando a similaridade entre um novo ticket e os tickets de referência é baixa, o sistema não força uma classificação automática. Em vez disso, o ticket é marcado para triagem humana.

Esse mecanismo permite:

- Evitar erros automáticos em casos ambíguos
- Detectar novos tipos de solicitações
- Criar novas classes quando necessário

Na prática, isso significa que o sistema se adapta à evolução do negócio.

---

## 5. Avaliação

Como não existem rótulos verdadeiros, métricas tradicionais como acurácia não são aplicáveis.

A solução é avaliada por meio de:

- Coerência semântica dos clusters
- Validação humana de amostras
- Taxa de reatribuição manual
- SLA por classe de atendimento

Essas métricas refletem o impacto real da solução no processo de atendimento.

---

## 6. Monitoramento em Produção

Em produção, os seguintes indicadores devem ser monitorados:

- Drift dos embeddings ao longo do tempo
- Distribuição de tickets por classe
- Taxa de baixa confiança (tickets enviados para triagem humana)
- Tempo médio de resolução por classe

Esses indicadores permitem acompanhar tanto a qualidade do modelo quanto seu impacto no negócio.

---

## 7. Estratégia de Retreino

À medida que tickets são revisados e corrigidos manualmente, esses dados passam a formar um conjunto rotulado.

O sistema pode então evoluir de um roteador por similaridade para um classificador supervisionado, utilizando esses rótulos reais.

O retreino pode ser acionado de forma periódica ou quando houver drift significativo nos dados.

---

## 8. Como aplicar uma estratégia de  *Active Learning*?

Com apenas 1% dos tickets rotulados, é possível aplicar uma estratégia de *active learning*:

- O modelo identifica os tickets com maior incerteza
- Esses tickets são priorizados para rotulagem humana
- O modelo é re-treinado com esse conjunto altamente informativo

Isso acelera significativamente a melhoria do modelo, reduzindo o custo de rotulagem.

---

## 9. Conclusão

Mesmo com dados extremamente limitados, a solução entregue permite classificar e rotear tickets de forma robusta, transparente e evolutiva.

O sistema foi projetado para operar em ambiente real, com controle de incerteza, monitoramento contínuo e uma estratégia clara de crescimento para aprendizado supervisionado.
