# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from app.schemas import PassengerInput, PredictionOutput
from app.model import predict_survival  # Función que usa tu modelo para predecir
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Titanic Survival Prediction API")

# CORS para permitir acceso desde el frontend
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de supervivencia del Titanic"}


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
   
    input_data = passenger.model_dump()
    input_df = pd.DataFrame([input_data])

    prediction_result, probability = predict_survival(input_df)

    return PredictionOutput(
        name=passenger.name,
        survived=bool(prediction_result),  
        probability=round(probability, 2),
    )