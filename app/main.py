from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import uvicorn
from app.schemas import PassengerInput, PredictionOutput
from app.model import predict_survival

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="""
    Esta API predice la probabilidad de supervivencia de un pasajero del Titanic usando un modelo Random Forest entrenado con datos históricos.

    ### Cómo usar la API:
    - Enviá una solicitud POST a `/predict` con los datos del pasajero en formato JSON.
    - Los datos deben seguir el esquema `PassengerInput`. Algunas características son categóricas y deben coincidir con las categorías usadas en el entrenamiento.
    - El nombre del pasajero (`name`) es opcional y se usa solo para personalizar la respuesta.

    ### Notas sobre las características:
    - **Age**: Edad original en años (se categoriza internamente en rangos numéricos: 0 a 8).
    - **Fare**: Tarifa original en valor numérico (se categoriza internamente en rangos numéricos: 0 a 6).
    - **Title**: Títulos mapeados como 'Mr', 'Mrs', 'Miss', 'Master', 'military', 'nobility', 'unmarried_women', 'married_women', 'religious'.
    - **TicketLocation**: Prefijos normalizados (ver categorías en `EXPECTED_CATEGORIES`).
    - **Family_Size_Grouped**: Basado en el tamaño de la familia (1 = Alone, 2-4 = Small, 5-6 = Medium, 7+ = Large).
    - **Feature Importances**: Las características más importantes son `Title_Mr` (0.15), `Sex_male` (0.14), `Sex_female` (0.13), `Pclass` (0.08), y `Fare` (0.07).
    """,
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir categorías válidas
EXPECTED_CATEGORIES = {
    "Sex": ["male", "female"],
    "Embarked": ["C", "Q", "S"],
    "Title": ["Dr", "Master", "Mr", "married_women", "military", "nobility", "religious", "unmarried_women"],    
    "TicketLocation": [
        "A/4", "A/5", "CA", "PC", "SOTON/OQ", "SC/Paris", "W/C", "Blank", "C", "F.C.", "F.C.C.", 
        "Fa", "P/PP", "PP", "S.C./A.4.", "S.O./P.P.", "S.O.C.", "S.O.P.", "S.P.", "SC", 
        "SC/AH", "SO/C", "STON/O", "STON/O2.", "SW/PP", "W.E.P.", "WE/P", "A4.", "A/S", "C.A./SOTON"
    ],
    "Family_Size_Grouped": ["Alone", "Small", "Medium", "Large"],
    "Age_Cut": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
    "Fare_cut": ["0", "1", "2", "3", "4", "5", "6"],
    "Name_LengthGB": [
        "(11.999, 18.0]", "(18.0, 20.0]", "(20.0, 23.0]", "(23.0, 25.0]", 
        "(25.0, 27.25]", "(27.25, 30.0]", "(30.0, 38.0]", "(38.0, 82.0]"
    ],
    "details": {
        "Age_Cut": {
            "0": "<=16 years",
            "1": ">16 to 20.125 years",
            "2": ">20.125 to 24 years",
            "3": ">24 to 28 years",
            "4": ">28 to 32.312 years",
            "5": ">32.312 to 38 years",
            "6": ">38 to 47 years",
            "7": ">47 to 80 years",
            "8": ">80 years"
        },
        "Fare_cut": {
            "0": "<=7.775 pounds",
            "1": ">7.775 to 8.662 pounds",
            "2": ">8.662 to 14.454 pounds",
            "3": ">14.454 to 26 pounds",
            "4": ">26 to 52.369 pounds",
            "5": ">52.369 to 512.329 pounds",
            "6": ">512.329 pounds"
        },
        "Family_Size_Grouped": {
            "Alone": "1 person",
            "Small": "2 to 4 people",
            "Medium": "5 to 6 people",
            "Large": "7 or more people"
        },
        "Name_LengthGB": {
            "(11.999, 18.0]": "Name length between 12 and 18 characters",
            "(18.0, 20.0]": "Name length between 18 and 20 characters",
            "(20.0, 23.0]": "Name length between 20 and 23 characters",
            "(23.0, 25.0]": "Name length between 23 and 25 characters",
            "(25.0, 27.25]": "Name length between 25 and 27.25 characters",
            "(27.25, 30.0]": "Name length between 27.25 and 30 characters",
            "(30.0, 38.0]": "Name length between 30 and 38 characters",
            "(38.0, 82.0]": "Name length between 38 and 82 characters"
        }
    }
}

@app.post("/predict", response_model=PredictionOutput)
async def predict(passenger: PassengerInput):
    """
    Predice si un pasajero habría sobrevivido al desastre del Titanic.

    - **Entrada**: Datos del pasajero en formato JSON, siguiendo el esquema `PassengerInput`.
    - **Salida**: Predicción de supervivencia, probabilidades y nivel de confianza.

    Ejemplo de uso:
    ```json
    {
        "name": "Kelly, Mr. James",
        "Pclass": 3,
        "Age": 34.5,
        "Fare": 7.8292,
        "Cabin_Assigned": 0,
        "Name_Size": 2.0,
        "TicketNumberCounts": 1,
        "Sex": "male",
        "Embarked": "Q",
        "Title": "Mr",
        "TicketLocation": "3",
        "Family_Size_Grouped": "Alone",
        "Age_Cut": "Adult",
        "Fare_cut": "Low",
        "Name_LengthGB": "Medium"
    }
    """
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


@app.get("/", summary="API Root Endpoint")
async def root():
    """
    Punto de entrada principal de la API de predicción de supervivencia del Titanic.

    Proporciona información sobre la API, los endpoints disponibles y cómo usarla.
    """
    return {
        "status": "running",
        "message": "Bienvenido a la Titanic Survival Prediction API",
        "version": "2.0",
        "description": (
            "Esta API predice la probabilidad de supervivencia de un pasajero del Titanic "
            "usando un modelo Random Forest entrenado con datos históricos."
        ),
        "endpoints": [
            {
                "path": "/predict",
                "method": "POST",
                "description": "Predice si un pasajero habría sobrevivido. Usa el esquema PassengerInput.",
                "example_request": {
                    "name": "Kelly, Mr. James",
                    "Pclass": 3,
                    "Age": 34.5,
                    "Fare": 7.8292,
                    "Cabin_Assigned": 0,
                    "Name_Size": 2.0,
                    "TicketNumberCounts": 1,
                    "Sex": "male",
                    "Embarked": "Q",
                    "Title": "Mr",
                    "TicketLocation": "SOTON/OQ",
                    "Family_Size_Grouped": "Alone",
                    "Age_Cut": "5",
                    "Fare_cut": "0",
                    "Name_LengthGB": "(23.0, 25.0]"
                }
            },
            {
                "path": "/categories",
                "method": "GET",
                "description": "Devuelve las categorías válidas para las variables categóricas del modelo."
            }
        ],
        "documentation": "https://titanicbackendss.onrender.com/docs",
        "usage": (
            "Envía un POST a /predict con un JSON válido según el esquema PassengerInput. "
            "Consulta /categories para obtener las categorías válidas. "
            "La documentación interactiva en /docs proporciona más detalles y ejemplos."
        ),
        "feature_importances": {
            "Title_Mr": 0.150890,
            "Sex_male": 0.141174,
            "Sex_female": 0.136364,
            "Pclass": 0.079872,
            "Fare": 0.068301,
            "note": "Otras características tienen menor importancia. Consulta /categories para más detalles."
        }
    }

@app.get("/categories", summary="Categorías válidas para las variables categóricas")
async def get_categories():
    """
    Devuelve las categorías válidas para las variables categóricas esperadas por el modelo de predicción de supervivencia del Titanic.

    ### Descripción
    Este endpoint proporciona las categorías aceptadas para las variables categóricas requeridas por el modelo Random Forest. Estas categorías se derivan del preprocesamiento realizado en los datos de entrenamiento, incluyendo normalización de valores (e.g., `TicketLocation`), mapeo de títulos (`Title`), y categorización de variables continuas (`Age`, `Fare`, `Family_Size`, `Name_Length`).

    ### Detalles de las categorías
    - **Sex**: Sexo del pasajero, codificado como 'male' o 'female'.
    - **Embarked**: Puerto de embarque, codificado como 'C' (Cherbourg), 'Q' (Queenstown), 'S' (Southampton).
    - **Title**: Título extraído del nombre, mapeado a categorías como 'Mr', 'Mrs', 'Miss', 'Master', o agrupaciones como 'military', 'nobility', 'unmarried_women', 'married_women', 'religious'.
    - **TicketLocation**: Prefijo normalizado del ticket, derivado de la limpieza de datos (e.g., 'SOTON/O.Q.' → 'SOTON/OQ').
    - **Family_Size_Grouped**: Tamaño de la familia a bordo, agrupado en 'Alone' (1 persona), 'Small' (2-4), 'Medium' (5-6), 'Large' (7+).
    - **Age_Cut**: Edad categorizada en rangos numéricos (0 a 8), basados en los cortes definidos en el preprocesamiento.
    - **Fare_cut**: Tarifa categorizada en rangos numéricos (0 a 6), basados en los cortes definidos en el preprocesamiento.
    - **Name_LengthGB**: Longitud del nombre categorizada en rangos (e.g., '(11.999, 18.0]', '(18.0, 20.0]').
    """
    try:
        return EXPECTED_CATEGORIES
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener categorías: {str(e)}")
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)