import joblib
import pandas as pd
from pathlib import Path

# Cargar el modelo
MODEL_PATH = Path(__file__).parent.parent / "models" / "modelo_titanic_vc2.pkl"
model = joblib.load(MODEL_PATH)

# Función de predicción
def predict_survival(passenger_data: pd.DataFrame):
    return model.predict(passenger_data).tolist()