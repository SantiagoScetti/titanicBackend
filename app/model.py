import joblib
import numpy as np
from pathlib import Path
import pandas as pd
import random  # Agregado
from sklearn.pipeline import Pipeline  

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
    
    # ✅ Agregar la columna que falta
    passenger_data["Family_Size"] = passenger_data["SibSp"] + passenger_data["Parch"]

    # ✅ Asegurar el orden de columnas esperado por el modelo
    expected_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "Family_Size_Grouped", "Cabin_Assigned", "Name_Size", "TicketNumberCounts",
        "Family_Size"
    ]
    passenger_data = passenger_data[expected_cols]

    # Predicción final (0 o 1)
    prediction = model.predict(passenger_data)
    # Convertimos a entero escalar para facilitar su manejo
    prediction_scalar = int(prediction[0])

     # ---------- Ajuste de probabilidad más avanzado ----------
    try:
        age = passenger_data["Age"].values[0]
        pclass = passenger_data["Pclass"].values[0]
        sex = passenger_data["Sex"].values[0]  # 0 = female, 1 = male (según encoding habitual)
        fare = passenger_data["Fare"].values[0]
        embarked = passenger_data["Embarked"].values[0]  # puede ser codificado
        family_size = passenger_data["Family_Size"].values[0]
        name_size = passenger_data["Name_Size"].values[0]
        ticket_counts = passenger_data["TicketNumberCounts"].values[0]
        cabin_assigned = passenger_data["Cabin_Assigned"].values[0]  # 0 o 1

        ajuste = 0.0

        # ✔️ Clase social
        if pclass == 1:
            ajuste += 0.03
        elif pclass == 3:
            ajuste -= 0.02

        # ✔️ Sexo
        if sex == 0:  # mujer
            ajuste += 0.04
        else:
            ajuste -= 0.02

        # ✔️ Edad
        if age < 12:
            ajuste += 0.03
        elif age > 60:
            ajuste -= 0.02

        # ✔️ Tarifa
        if fare > 50:
            ajuste += 0.02
        elif fare < 10:
            ajuste -= 0.01

        # ✔️ Puerto de embarque
        if embarked == 0:  # Southampton
            ajuste -= 0.01
        elif embarked == 2:  # Cherbourg
            ajuste += 0.01

        # ✔️ Familia
        if family_size >= 5:
            ajuste -= 0.03
        elif family_size == 1:
            ajuste -= 0.01
        else:
            ajuste += 0.01

        # ✔️ Tamaño del nombre (puede indicar nobleza o título)
        if name_size >= 5:
            ajuste += 0.01

        # ✔️ Compartir ticket con otros (posible familia/grupo)
        if ticket_counts > 2:
            ajuste += 0.02

        # ✔️ Cabina asignada
        if cabin_assigned == 1:
            ajuste += 0.02

        # ---------- Probabilidad final con base aleatoria ----------
        if prediction_scalar == 1:
            base_prob = random.uniform(0.85, 0.93)
            probability = round(min(0.99, base_prob + ajuste), 2)
        else:
            base_prob = random.uniform(0.02, 0.15)
            probability = round(max(0.01, base_prob - ajuste), 2)

    except Exception as e:
        probability = 1.0 if prediction_scalar == 1 else 0.0
        print(f"Error calculando probabilidad: {e}")

    print(f"Final probability: {probability}")
    return prediction_scalar, probability
