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
    # Convertir la entrada a un diccionario y luego a DataFrame para la predicción
    # Usamos model_dump() en lugar de dict() que está obsoleto
    input_data = passenger.model_dump()
    
    # Creamos el DataFrame correctamente
    input_df = pd.DataFrame([input_data])

    # Realiza la predicción y obtiene la probabilidad
    prediction_result, probability = predict_survival(input_df)

    # Ahora prediction_result y probability son escalares, no necesitamos indexarlos
    return PredictionOutput(
        id=None,
        name=passenger.name,
        survived=bool(prediction_result),  
        probability=round(probability, 2),
    )
