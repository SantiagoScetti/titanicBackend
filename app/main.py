from fastapi import FastAPI
import pandas as pd
from app.schemas import PassengerInput, PredictionOutput
from app.model import predict_survival

app = FastAPI(title="Titanic Survival Prediction API")

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de supervivencia del Titanic"}

@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    # Convertir entrada en DataFrame
    input_data = pd.DataFrame([passenger.dict()])
    
    # Obtener predicción
    prediction = predict_survival(input_data)
    
    return {"PassengerId": 0, "Survived": prediction[0]}
