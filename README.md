# Tech Challenge Fase 4 - Stock Price Predictor

Projeto de Deep Learning para previsão de ações (PETR4) utilizando LSTM, PyTorch e MLOps.

## 📋 Estrutura
- `src/`: Código fonte do treinamento e modelagem.
- `api/`: API FastAPI para inferência.
- `models/`: Artefatos do modelo treinado.
- `mlruns/`: Logs dos experimentos do MLflow.

## 🚀 Como Rodar

### Pré-requisitos
- Docker e Docker Compose instalados.

### Passo a Passo
1. **Treinar o Modelo (Opcional):**
   O modelo já está treinado na pasta `models/`. Para treinar novamente:
   ```bash
   python -m src.train
   ```

2. **Subir a Aplicação:**
   ```bash
   docker compose up --build
   ```

3. **Acessar:**
   - **API (Swagger):** http://localhost:8000/docs
   - **MLflow UI:** http://localhost:5000
   - **Grafana (Monitoramento):** http://localhost:3000 (Login: admin/admin)
   - **Prometheus:** http://localhost:9090

## 📊 Performance
O melhor modelo (LSTM com 64 unidades ocultas) obteve:
- **RMSE:** 1.1693
- **MAE:** 0.9269
