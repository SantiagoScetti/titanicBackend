# app/main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import pandas as pd
from app.database import SessionLocal, init_db
from app.models import Prediction
from app.schemas import PassengerInput, PredictionOutput
from app.model import predict_survival  # Función que usa tu modelo para predecir


app = FastAPI(title="Titanic Survival Prediction API")

# Inicializar la base de datos al arrancar la API
@app.on_event("startup")
def on_startup():
    init_db()

# Dependencia para obtener la sesión de DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de supervivencia del Titanic"}

@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput, db: Session = Depends(get_db)):
    # Convertir la entrada a DataFrame para la predicción
    input_data = pd.DataFrame([passenger.dict()])
    
    # Realiza la predicción (se espera que predict_survival devuelva un 0 o 1)
    prediction_result = predict_survival(input_data)[0]
    
    # Crear la entrada en la base de datos
    prediction_entry = Prediction(
        name=passenger.name,
        Pclass=passenger.Pclass,
        Sex=passenger.Sex,
        Age=passenger.Age,
        SibSp=passenger.SibSp,
        Parch=passenger.Parch,
        Fare=passenger.Fare,
        Embarked=passenger.Embarked,
        Cabin_Assigned=passenger.Cabin_Assigned,
        Name_Size=passenger.Name_Size,
        TicketNumberCounts=passenger.TicketNumberCounts,
        Family_Size_Grouped=passenger.Family_Size_Grouped,
        survived=bool(prediction_result)
    )
    
    db.add(prediction_entry)
    db.commit()
    db.refresh(prediction_entry)
    
    return PredictionOutput(
        id=prediction_entry.id,
        name=prediction_entry.name,
        survived=prediction_entry.survived
    )