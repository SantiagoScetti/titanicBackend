# app/main.py
from fastapi import FastAPI
import pandas as pd
from app.schemas import PassengerInput, PredictionOutput
from app.model import predict_survival  # Función que usa tu modelo para predecir

app = FastAPI(title="Titanic Survival Prediction API")

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de supervivencia del Titanic"}


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    # Convertir la entrada a DataFrame para la predicción
    input_data = pd.DataFrame([passenger.dict()])

    # Realiza la predicción y obtiene la probabilidad
    prediction_result, probability = predict_survival(input_data)

    return PredictionOutput(
        id=None,
        name=passenger.name,
        survived=bool(prediction_result[0]),
        probability=round(probability[0], 2),  # redondeado a 2 decimales
    )
