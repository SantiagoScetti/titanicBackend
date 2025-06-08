import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple


class TitanicPredictor:
    """Clase para manejar las predicciones del modelo Titanic"""

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Carga el modelo desde el archivo pkl"""
        try:
            MODEL_PATH = (
                Path(__file__).parent.parent / "models" / "modelo_titanic_rfc.pkl"
            )
            self.model = joblib.load(MODEL_PATH)
            print("✅ Modelo cargado exitosamente")
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            raise

    def preprocess_data(self, passenger_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos del pasajero para el modelo

        Args:
            passenger_data: DataFrame con los datos del pasajero

        Returns:
            DataFrame preprocesado
        """
        # Crear una copia para no modificar el original
        data = passenger_data.copy()

        # Agregar Family_Size
        data["Family_Size"] = data["SibSp"] + data["Parch"]

        # Asegurar el orden correcto de columnas
        expected_cols = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Family_Size_Grouped",
            "Cabin_Assigned",
            "Name_Size",
            "TicketNumberCounts",
            "Family_Size",
        ]

        # Verificar que todas las columnas estén presentes
        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Faltan las siguientes columnas: {missing_cols}")

        return data[expected_cols]

    def get_confidence_level(self, max_probability: float) -> str:
        """
        Determina el nivel de confianza basado en la probabilidad máxima

        Args:
            max_probability: La probabilidad más alta entre sobrevivir y morir

        Returns:
            Nivel de confianza como string
        """
        if max_probability >= 0.9:
            return "Muy Alta"
        elif max_probability >= 0.8:
            return "Alta"
        elif max_probability >= 0.7:
            return "Media"
        elif max_probability >= 0.6:
            return "Baja"
        else:
            return "Muy Baja"

    def predict_survival(
        self, passenger_data: pd.DataFrame
    ) -> Tuple[int, float, float, str]:
        """
        Realiza la predicción de supervivencia

        Args:
            passenger_data: DataFrame con los datos del pasajero

        Returns:
            Tupla con (predicción, prob_morir, prob_sobrevivir, nivel_confianza)
        """
        if self.model is None:
            raise ValueError("El modelo no está cargado")

        # Preprocesar datos
        processed_data = self.preprocess_data(passenger_data)

        # Realizar predicción
        prediction = self.model.predict(processed_data)[0]
        probabilities = self.model.predict_proba(processed_data)[0]

        prob_die = probabilities[0]  # Probabilidad de NO sobrevivir
        prob_survive = probabilities[1]  # Probabilidad de sobrevivir

        # Calcular nivel de confianza
        max_prob = max(prob_die, prob_survive)
        confidence_level = self.get_confidence_level(max_prob)

        return prediction, prob_die, prob_survive, confidence_level


# Instancia global del predictor
predictor = TitanicPredictor()


def predict_survival(passenger_data: pd.DataFrame) -> Tuple[int, float, float, str]:
    """
    Función wrapper para mantener compatibilidad
    """
    return predictor.predict_survival(passenger_data)
