# Tech Challenge Fase 4 - Stock Price Predictor

Projeto de **Deep Learning** para previsão de fechamento de ações (foco em `PETR4.SA`) utilizando redes neurais **LSTM** (Long Short-Term Memory). O projeto engloba todo o ciclo de MLOps, desde o treinamento com rastreamento de experimentos até o deploy de uma API monitorada.

## 🌟 Destaques do Projeto
- **Modelo LSTM:** Rede neural recorrente implementada em PyTorch.
- **MLOps com MLflow:** Rastreamento de métricas (RMSE, MAE), parâmetros e artefatos (gráficos de previsão).
- **Grid Search:** Script de treinamento que testa múltiplos hiperparâmetros automaticamente e salva o melhor modelo.
- **API Resiliente:** Desenvolvida com **FastAPI**, possui fallback automático: se o usuário não enviar dados históricos, a API busca os últimos 60 dias no Yahoo Finance em tempo real.
- **Observabilidade:** Monitoramento de métricas de performance e latência com **Prometheus** e **Grafana**.

## 📋 Estrutura
- `src/`: Scripts de treinamento, pré-processamento e definição do modelo.
- `api/`: Aplicação FastAPI e esquemas de dados.
- `models/`: Armazena o modelo treinado (`.pth`) e o scaler (`.joblib`).
- `mlruns/`: Logs locais dos experimentos do MLflow.
- `load_test.py`: Script para simular tráfego e testar a carga da API.

## 🚀 Como Rodar

### Pré-requisitos
- Docker e Docker Compose instalados.

### 1. Execução Completa (Docker)
O ambiente está containerizado. Para subir a API, o MLflow, Prometheus e Grafana:

```bash
docker compose up --build
```

### 2. Acessando os Serviços
- **API Docs (Swagger):** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
- **Grafana:** http://localhost:3000 (Login: `admin` / Senha: `admin`)
- **Prometheus:** http://localhost:9090

---

## 🧪 Como Testar a API

### Opção A: Previsão Automática (Recomendado)
Envie um JSON vazio ou sem o campo `last_60_days`. A API buscará os dados mais recentes da B3 automaticamente.

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{}"
```

### Opção B: Previsão Manual
Envie uma lista com os preços de fechamento dos últimos 60 dias.

```json
{
  "last_60_days": [34.5, 35.2, 34.8, ..., 36.1]
}
```

### Opção C: Teste de Carga (Gerar Métricas)
Para ver os gráficos do Grafana se moverem, execute o script de teste de carga em outro terminal (requer python local):

```bash
pip install requests
python load_test.py
```
*Isso enviará requisições aleatórias para a API, simulando uso real.*

---

## 🧠 Treinamento do Modelo
Caso queira retreinar o modelo do zero (fora do Docker):

```bash
pip install -r requirements.txt
python -m src.train
```
Isso executará o **Grid Search**, salvará o melhor modelo em `models/` e registrará os resultados no MLflow.
