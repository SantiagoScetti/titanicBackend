from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from app.schemas import PassengerInput, PredictionOutput
from app.model import predict_survival

app = FastAPI(title="Titanic API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionOutput)
async def predict(passenger: PassengerInput):
    try:
        # Convertir entrada a DataFrame
        input_dict = passenger.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Obtener nombre o usar uno por defecto si no viene
        nombre_pasajero = input_dict.get("name", "Pasajero desconocido")

        # Predecir
        pred, prob_die, prob_survive, nivel_confianza = predict_survival(input_df)

        # Generar mensaje interpretativo
        if pred == 1:
            mensaje = f"🟢 {nombre_pasajero} HABRÍA SOBREVIVIDO"
        else:
            mensaje = f"🔴 {nombre_pasajero} NO habría sobrevivido"

        # Respuesta
        return PredictionOutput(
            name=nombre_pasajero,
            survived=bool(pred),
            probability_survive=round(prob_survive * 100, 2),
            probability_die=round(prob_die * 100, 2),
            message=mensaje,
            confidence_level=nivel_confianza,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Titanic prediction API online"}
