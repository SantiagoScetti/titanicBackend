# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from app.schemas import PassengerInput, PredictionOutput
from app.model import predict_survival

# Configuración de la aplicación
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API para predecir la supervivencia en el Titanic usando Machine Learning",
    version="2.0.0"
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    """Endpoint de bienvenida"""
    return {
        "message": "🚢 API de predicción de supervivencia del Titanic",
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict - Realizar predicción",
            "test": "/test - Prueba con datos de ejemplo",
            "docs": "/docs - Documentación interactiva"
        }
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(passenger: PassengerInput):
    """
    Predice si un pasajero habría sobrevivido al hundimiento del Titanic
    
    Args:
        passenger: Datos del pasajero
        
    Returns:
        Predicción con probabilidades y mensaje interpretativo
    """
    try:
        # Convertir a DataFrame
        input_data = passenger.model_dump()
        input_df = pd.DataFrame([input_data])
        
        # Realizar predicción
        prediction, prob_die, prob_survive, confidence_level = predict_survival(input_df)
        
        # Crear mensaje interpretativo
        if prediction == 1:
            message = f"🟢 {passenger.name} HABRÍA SOBREVIVIDO"
        else:
            message = f"🔴 {passenger.name} NO habría sobrevivido"
        
        return PredictionOutput(
            name=passenger.name,
            survived=bool(prediction),
            probability_survive=round(prob_survive * 100, 2),
            probability_die=round(prob_die * 100, 2),
            message=message,
            confidence_level=confidence_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/test")
async def test_prediction():
    """
    Endpoint de prueba con datos de ejemplo
    
    Returns:
        Predicción de ejemplo para verificar el funcionamiento
    """
    sample_data = {
        "name": "Rose DeWitt Bukater",
        "Pclass": 1,
        "Sex": "female",
        "Age": 17,
        "SibSp": 1,
        "Parch": 1,
        "Fare": 100.0,
        "Embarked": "S",
        "Cabin_Assigned": 1,
        "Name_Size": 3,
        "TicketNumberCounts": 2,
        "Family_Size_Grouped": "Medium"
    }
    
    try:
        input_df = pd.DataFrame([sample_data])
        prediction, prob_die, prob_survive, confidence_level = predict_survival(input_df)
        
        return {
            "test_data": sample_data,
            "results": {
                "survived": bool(prediction),
                "probability_die": f"{prob_die*100:.2f}%",
                "probability_survive": f"{prob_survive*100:.2f}%",
                "confidence_level": confidence_level,
                "message": "🟢 HABRÍA SOBREVIVIDO" if prediction == 1 else "🔴 NO habría sobrevivido"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la prueba: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de la API"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)