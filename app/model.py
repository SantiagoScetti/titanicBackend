import joblib
import numpy as np
from pathlib import Path
import pandas as pd

# Cargar el modelo
MODEL_PATH = Path(__file__).parent.parent / "models" / "modelo_titanic_vc2.pkl"
model = joblib.load(MODEL_PATH)

def predict_survival(passenger_data):
    """
    Realiza predicciones de supervivencia usando el modelo cargado
    
    Args:
        passenger_data (pd.DataFrame): DataFrame con los datos del pasajero
    
    Returns:
        tuple: (predicción, probabilidad)
    """
    # Ya recibimos un DataFrame de main.py, no necesitamos crear uno nuevo
    # Si intentamos crear un DataFrame de un DataFrame, obtenemos una estructura 3D
    
    # Verificamos que passenger_data sea un DataFrame
    if not isinstance(passenger_data, pd.DataFrame):
        raise ValueError("passenger_data debe ser un DataFrame de pandas")
    
    # Predicción final (0 o 1)
    prediction = model.predict(passenger_data)

    # Simular probabilidad en función de los votos
    try:
        # Extraer el VotingClassifier desde el pipeline
        voting_clf = model.named_steps['votingclassifier']

        # Obtener predicciones individuales de cada estimador base
        individual_preds = [est.predict(passenger_data) for name, est in voting_clf.estimators]

        # Sumamos los votos positivos y dividimos por la cantidad de modelos
        votes = np.sum(individual_preds)
        prob = round(votes / len(voting_clf.estimators), 2)
    except:
        # Si algo falla, simplemente damos 1.0 o 0.0
        prob = 1.0 if prediction[0] == 1 else 0.0

    return prediction, prob