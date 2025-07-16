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
    if max_prob >= 0.9:
        confidence = "Muy Alta"
    elif max_prob >= 0.8:
        confidence = "Alta"
    elif max_prob >= 0.7:
        confidence = "Media"
    elif max_prob >= 0.6:
        confidence = "Baja"
    else:
        confidence = "Muy Baja"

    return prediction, prob_die, prob_survive, confidence
