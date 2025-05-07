import joblib
import numpy as np
from pathlib import Path

# Cargar el modelo
MODEL_PATH = Path(__file__).parent.parent / "models" / "modelo_titanic_vc2.pkl"
model = joblib.load(MODEL_PATH)

def predict_survival(passenger_data):
    # Convertir el diccionario en un DataFrame con una fila
    import pandas as pd
    df = pd.DataFrame([passenger_data])

    # Predicción final (0 o 1)
    prediction = model.predict(df)

    # Simular probabilidad en función de los votos
    try:
        # Extraer el VotingClassifier desde el pipeline
        voting_clf = model.named_steps['votingclassifier']

        # Obtener predicciones individuales de cada estimador base
        individual_preds = [est.predict(df) for name, est in voting_clf.estimators]

        # Sumamos los votos positivos y dividimos por la cantidad de modelos
        votes = np.sum(individual_preds)
        prob = round(votes / len(individual_preds), 2)
    except:
        # Si algo falla, simplemente damos 1.0 o 0.0
        prob = 1.0 if prediction[0] == 1 else 0.0

    return int(prediction[0]), prob