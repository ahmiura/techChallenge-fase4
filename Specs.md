# 📘 Especificação do Projeto: Tech Challenge Fase 4
**Curso:** Machine Learning Engineering (Postech)
**Tema:** Deep Learning & Time Series Forecasting com LSTM

---

## 1. 🎯 Objetivo do Projeto
Desenvolver uma arquitetura completa de **Deep Learning** para resolver um problema de **Série Temporal**. O objetivo é prever o preço de fechamento (*Close Price*) de uma ação da bolsa de valores utilizando redes neurais **LSTM (Long Short-Term Memory)**.

O projeto deve demonstrar maturidade em MLOps, cobrindo o ciclo de vida completo: ingestão de dados, experimentação, treinamento, avaliação e produtização via API containerizada.

---

## 2. 🛠️ Stack Tecnológica & Ferramentas

### Essencial (Obrigatório)
* **Linguagem:** Python 3.9+.
* **Framework de Deep Learning:** PyTorch.
* **Fonte de Dados:** Biblioteca `yfinance` (Yahoo Finance API).
* **Manipulação de Dados:** Pandas, Numpy, Scikit-learn.
* **API Web:** FastAPI.
* **Containerização:** Docker (Criação de imagem e container).
* **Controle de Versão:** Git / GitHub.

### Recomendado (Para Arquitetura Limpa e Profissional)
* **Rastreamento de Experimentos:** MLflow (Para logar parâmetros, métricas e artefatos).
* **Estrutura de Projeto:** Padrão "Cookiecutter" ou estrutura modular separando `src` de `notebooks`.

---

## 3. 📜 Regras de Negócio e Pipeline de Dados

### 3.1. Coleta e Ingestão
* **Fonte:** Utilizar dados históricos diários do Yahoo Finance.
* **Ativo:** Escolher uma empresa com histórico consistente (ex: `PETR4.SA`, `VALE3.SA`, `ITUB3.SA`).
* **Janela Temporal:** Recomenda-se utilizar pelo menos 5 anos de dados para capturar sazonalidades.

### 3.2. Pré-processamento (Crítico para LSTMs)
* **Normalização:** É **obrigatório** normalizar os dados (ex: `MinMaxScaler` entre 0 e 1). LSTMs não convergem bem com dados em escala monetária bruta (ex: R$ 30,00).
* **Janelamento (Sliding Window):**
    * O problema deve ser modelado como aprendizado supervisionado.
    * **Feature (X):** Sequência dos últimos *N* dias (ex: 60 dias).
    * **Target (y):** Preço do dia seguinte (T+1).
* **Divisão de Dados:**
    * **NÃO** utilizar `train_test_split` com `shuffle=True`.
    * A divisão deve ser **cronológica** (ex: Treino: 2018-2023, Teste: 2024 em diante) para evitar *data leakage* (vazamento de dados futuros).

---

## 4. 🧠 Modelagem: Deep Learning

### 4.1. Arquitetura da Rede
* **Tipo:** Recorrente (RNN) com células **LSTM**.
* **Input Shape:** `(Batch_Size, Timesteps, Features)`. Ex: `(32, 60, 1)`.
* **Camadas Ocultas:** Pelo menos uma camada LSTM.
* **Regularização:** Uso obrigatório de **Dropout** (ex: 0.2) após as camadas LSTM para prevenir overfitting.
* **Saída:** Camada Densa (`Dense`) com 1 neurônio e ativação linear (para regressão).

### 4.2. Compilação e Treino
* **Função de Perda (Loss):** MSE (Mean Squared Error).
* **Otimizador:** Adam (Recomendado por adaptar o *learning rate*).
* **Métricas de Monitoramento:** MAE, Loss.

---

## 5. 📊 Avaliação e Métricas

O modelo deve ser avaliado no conjunto de teste (dados nunca vistos) utilizando as seguintes métricas obrigatórias:
1.  **RMSE (Root Mean Squared Error):** Penaliza grandes erros.
2.  **MAE (Mean Absolute Error):** Erro médio absoluto na unidade monetária.
3.  **MAPE (Mean Absolute Percentage Error):** Erro percentual médio (fácil interpretação para o negócio).

> **Visualização:** Deve ser gerado um gráfico de linha comparando a série temporal real vs. a série predita pelo modelo.

---

## 6. 🚀 Arquitetura de Software (Clean Code)

O projeto deve evitar o "Jupyter Notebook Driven Development" em produção. Sugere-se a seguinte estrutura:

```text
techChallenge-fase4/
├── .gitignore
├── README.md               # Documentação do projeto
├── Dockerfile              # Receita da imagem da API
├── docker-compose.yml      # (Opcional) Orquestração API + MLflow
├── requirements.txt        # Dependências
├── notebooks/              # Apenas para exploração e gráficos
│   └── exploratory.ipynb
├── src/                    # Código Fonte Modular
│   ├── __init__.py
│   ├── config.py           # Configurações (ticker, datas, caminhos)
│   ├── data.py             # Download e Pré-processamento
│   ├── model.py            # Definição da classe/função do modelo LSTM
│   ├── train.py            # Pipeline de treinamento (com MLflow)
│   └── predict.py          # Lógica de inferência (carrega modelo + scaler)
└── api/                    # Aplicação Web
    ├── app.py              # Entrypoint (FastAPI)
    └── schemas.py          # Validação de dados de entrada/saída
```

---

## 7. 🚢 Deploy e Entregáveis

### 7.1. API (Backend)
Desenvolver uma API REST com os seguintes requisitos:

Endpoint /predict: Recebe os dados (ou busca internamente) e retorna o preço previsto.

A API deve carregar o modelo treinado e o Scaler salvo anteriormente para desnormalizar a previsão.

### 7.2. Docker
Criar um Dockerfile que instale as dependências e exponha a porta da API.

A aplicação deve rodar com um simples comando docker run.

### 7.3. Lista de Entregáveis
Link do Repositório Git: Código organizado e limpo.

Vídeo Demo: Explicando a arquitetura, o modelo e mostrando a API funcionando.

Link da API em Produção: Deploy em nuvem (Render, AWS, Azure, etc) OU instruções claras para rodar localmente via Docker.

---

## 8. 📚 Conceitos das Aulas Aplicados
Redes Recorrentes (RNNs): Entendimento de memória sequencial.

LSTM: Solução para o problema do gradiente que desaparece (Vanishing Gradient) em séries longas.

Normalização: Impacto direto na convergência do Gradient Descent.

Regularização (Dropout): Técnica para melhorar a generalização do modelo.

Otimizadores (Adam): Eficiência no ajuste de pesos da rede neural.