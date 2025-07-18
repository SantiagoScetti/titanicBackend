import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple

# Ruta al modelo
model_path = Path(__file__).resolve().parent.parent / "models" / "modelo_titanic_rfc.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Modelo no encontrado en: {model_path}")
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo: {e}")

EXPECTED_CATEGORIES = {
    'Sex': ['male', 'female'],
    'Embarked': ['C', 'Q', 'S'],
    'Title': ['Mr', 'Mrs', 'Miss', 'Master', 'military', 'nobility', 'unmarried_women', 'married_women', 'religious'],
    'TicketLocation': [
        'A/4', 'A/5', 'CA', 'PC', 'SOTON/OQ', 'SC/Paris', 'W/C', 'Blank', 'C', 'F.C.', 'F.C.C.', 'Fa', 
        'P/PP', 'PP', 'S.C./A.4.', 'S.O./P.P.', 'S.O.C.', 'S.O.P.', 'S.P.', 'SC', 'SC/AH', 'SO/C', 
        'STON/O', 'STON/O2.', 'SW/PP', 'W.E.P.', 'WE/P', 'A4.', 'A/S', 'C.A./SOTON'
    ],
    'Family_Size_Grouped': ['Alone', 'Small', 'Medium', 'Large'],
    'Age_Cut': ['0', '1', '2', '3', '4', '5', '6', '7', '8'],
    'Fare_cut': ['0', '1', '2', '3', '4', '5', '6'],
    'Name_LengthGB': [
        '(11.999, 18.0]', '(18.0, 20.0]', '(20.0, 23.0]', '(23.0, 25.0]', 
        '(25.0, 27.25]', '(27.25, 30.0]', '(30.0, 38.0]', '(38.0, 82.0]'
    ]
}

def validate_categories(data: pd.DataFrame):
    for col, valid_values in EXPECTED_CATEGORIES.items():
        if col in data.columns:
            invalid = set(data[col]) - set(valid_values)
            if invalid:
                raise ValueError(f"Valores inválidos en {col}: {invalid}")



def predict_survival(data: pd.DataFrame) -> Tuple[int, float, float, str]:
    """
    Predice la supervivencia de un pasajero usando el modelo cargado.

    Args:
        data (pd.DataFrame): DataFrame con los datos del pasajero (una sola fila)

    Returns:
        Tuple:
            - prediction (int): 0 = no sobrevive, 1 = sobrevive
            - prob_die (float): probabilidad de no sobrevivir
            - prob_survive (float): probabilidad de sobrevivir
            - confidence (str): nivel de confianza textual
    """
    # Verificar que el input tenga las columnas necesarias
    validate_categories(data)
    required_cols = model.feature_names_in_
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas por el modelo: {missing}")

    # Asegurarse de que estén en el mismo orden
    data = data[required_cols]

    # Predicción
    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]

    prob_die = probabilities[0]
    prob_survive = probabilities[1]

    # Nivel de confianza
    max_prob = max(prob_die, prob_survive)
    if max_prob >= 0.85:
        confidence = "Muy Alta"
    elif max_prob >= 0.70:
        confidence = "Alta"
    elif max_prob >= 0.60:
        confidence = "Media"
    elif max_prob >= 0.6:
        confidence = "Baja"
    else:
        confidence = "Muy Baja"

    return prediction, prob_die, prob_survive, confidence