from fastapi import FastAPI, HTTPException
from src.predict import Predictor
from src.config import TICKER, MODEL_PATH
from api.schemas import PredictionRequest, PredictionResponse
import uvicorn
import os

app = FastAPI(title="Stock Price Predictor API", version="1.0")
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
        price = predictor.predict_next_day(request.last_60_days)
        return {"ticker": TICKER, "predicted_price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)