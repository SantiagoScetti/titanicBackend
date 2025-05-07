import joblib
import numpy as np
from pathlib import Path
import pandas as pd

# Cargar el modelo
MODEL_PATH = Path(__file__).parent.parent / "models" / "modelo_titanic_vc2.pkl"
model = joblib.load(MODEL_PATH)


def predict_survival(passenger_data):
    print(f"Type of passenger_data: {type(passenger_data)}")
    print(f"Shape of passenger_data: {passenger_data.shape}")
    print(f"Columns: {passenger_data.columns}")
    """
    Realiza predicciones de supervivencia usando el modelo cargado
    
    Args:
        passenger_data (pd.DataFrame): DataFrame con los datos del pasajero
    
    Returns:
        tuple: (predicción, probabilidad)
    """
    # Verificamos que passenger_data sea un DataFrame
    if not isinstance(passenger_data, pd.DataFrame):
        raise ValueError("passenger_data debe ser un DataFrame de pandas")

    # Predicción final (0 o 1)
    prediction = model.predict(passenger_data)
    # Convertimos a entero escalar para facilitar su manejo
    prediction_scalar = int(prediction[0])

    # Simular probabilidad en función de los votos
    try:
        # Extraer el VotingClassifier desde el pipeline
        voting_clf = model.named_steps["votingclassifier"]

        # Obtener predicciones individuales de cada estimador base
        individual_preds = [
            est.predict(passenger_data)[0] for name, est in voting_clf.estimators
        ]

        # Sumamos los votos positivos y dividimos por la cantidad de modelos
        votes = sum(individual_preds)  # np.sum no es necesario para valores escalares
        probability = float(votes / len(voting_clf.estimators))
    except Exception as e:
        # Si algo falla, simplemente damos 1.0 o 0.0
        probability = 1.0 if prediction_scalar == 1 else 0.0
        print(f"Error calculando probabilidad: {e}")

        print(f"Type of prediction: {type(prediction)}")
        print(f"Value of prediction: {prediction}")
        print(f"Type of probability: {type(probability)}")
        print(f"Value of probability: {probability}")

    return prediction_scalar, probability
