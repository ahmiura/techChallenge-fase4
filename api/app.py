from fastapi import FastAPI, HTTPException, Request
from src.predict import Predictor
from src.config import TICKER, MODEL_PATH
from api.schemas import PredictionRequest, PredictionResponse
import uvicorn
import os
import yfinance as yf
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

app = FastAPI(title="Stock Price Predictor API", version="1.0")

# Métrica customizada para requisições em andamento
in_progress_gauge = Gauge("http_requests_in_progress", "Requests in progress", ["handler", "method"])

@app.middleware("http")
async def track_in_progress(request: Request, call_next):
    # Incrementa ao receber a requisição
    in_progress_gauge.labels(handler="*", method=request.method).inc()
    try:
        return await call_next(request)
    finally:
        # Decrementa ao finalizar (sucesso ou erro)
        in_progress_gauge.labels(handler="*", method=request.method).dec()

# Configuração de Monitoramento (Prometheus)
Instrumentator().instrument(app).expose(app)

predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        # Verifica se o modelo existe antes de carregar
        if os.path.exists(MODEL_PATH):
            predictor = Predictor()
            print("Modelo carregado com sucesso!")
        else:
            print("Aviso: Modelo não encontrado. Execute src/train.py primeiro.")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Modelo não carregado ou indisponível.")
    
    try:
        input_data = request.last_60_days

        # Se o usuário não enviou dados, buscamos automaticamente no Yahoo Finance
        if not input_data:
            print(f"Buscando dados recentes para {TICKER} no Yahoo Finance...")
            # Baixamos 6 meses para garantir que tenhamos 60 dias úteis de pregão
            df = yf.download(TICKER, period="6mo", progress=False)
            # Pegamos apenas os últimos 60 valores de fechamento
            input_data = df['Close'].values[-60:].tolist()
            
            if len(input_data) < 60:
                raise HTTPException(status_code=400, detail=f"Dados insuficientes encontrados no Yahoo Finance: {len(input_data)} dias.")

        price = predictor.predict_next_day(input_data)
        return {"ticker": TICKER, "predicted_price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)