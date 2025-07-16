# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional


class PassengerInput(BaseModel):
    """
    Esquema de entrada para predecir la supervivencia de un pasajero del Titanic.
    Debe incluir todos los campos que el modelo de machine learning espera.
    """

    name: Optional[str] = Field(None, description="Nombre del pasajero (opcional, solo para mostrar)")
    PassengerId: int = Field(..., description="ID único del pasajero")
    Pclass: int = Field(..., ge=1, le=3, description="Clase del pasajero (1 = alta, 2 = media, 3 = baja)")
    Sex: str = Field(..., description="Sexo del pasajero (male/female)")
    Age: float = Field(..., ge=0, le=120, description="Edad del pasajero")
    Ticket: str = Field(..., description="Código del ticket del pasajero")
    Fare: float = Field(..., ge=0, description="Tarifa pagada por el pasajero")
    Cabin: str = Field(..., description="Cabina asignada (si la hay)")
    Embarked: str = Field(..., description="Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)")

    Family_Size: int = Field(..., ge=1, le=20, description="Tamaño total de la familia a bordo")
    Family_Size_Grouped: str = Field(..., description="Categoría del tamaño familiar (Alone, Small, Medium, Large)")
    Age_Cut: str = Field(..., description="Rango de edad categorizado (Child, Teenager, Adult, Senior, etc.)")
    Fare_cut: str = Field(..., description="Rango de tarifa (Low, Medium, High, etc.)")
    Title: str = Field(..., description="Título del pasajero extraído del nombre (Mr, Mrs, Miss, etc.)")
    Name_Length: int = Field(..., ge=1, le=100, description="Longitud total del nombre del pasajero")
    Name_LengthGB: str = Field(..., description="Grupo de longitud del nombre (Short, Medium, Long)")
    Name_Size: float = Field(..., ge=0, description="Tamaño categórico asignado al nombre")
    TicketNumber: str = Field(..., description="Número extraído del ticket")
    TicketNumberCounts: int = Field(..., ge=1, description="Número de pasajeros con el mismo número de ticket")
    TicketLocation: str = Field(..., description="Prefijo del ticket (A, B, C, etc.)")
    Cabin_Assigned: int = Field(..., ge=0, le=1, description="¿Tenía cabina asignada? (1 = sí, 0 = no)")

    class Config:
        json_schema_extra = {
            "example": {
                "PassengerId": 892,
                "Pclass": 3,
                "name": "Kelly, Mr. James",
                "Sex": "male",
                "Age": 34.5,
                "Ticket": "330911",
                "Fare": 7.8292,
                "Cabin": "",
                "Embarked": "Q",
                "Family_Size": 1,
                "Family_Size_Grouped": "Alone",
                "Age_Cut": "Adult",
                "Fare_cut": "Low",
                "Title": "Mr",
                "Name_Length": 17,
                "Name_LengthGB": "Medium",
                "Name_Size": 2.0,
                "TicketNumber": "330911",
                "TicketNumberCounts": 1,
                "TicketLocation": "3",
                "Cabin_Assigned": 0
            }
        }


class PredictionOutput(BaseModel):
    """
    Esquema de salida que representa el resultado de la predicción de supervivencia.
    """

    name: str = Field(..., description="Nombre del pasajero")
    survived: bool = Field(..., description="¿El modelo predice que sobreviviría?")
    probability_survive: float = Field(..., description="Probabilidad de que sobreviva (%)")
    probability_die: float = Field(..., description="Probabilidad de que no sobreviva (%)")
    message: str = Field(..., description="Mensaje interpretativo de la predicción")
    confidence_level: str = Field(..., description="Nivel de confianza de la predicción (Alta, Media, Baja)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Kelly, Mr. James",
                "survived": False,
                "probability_survive": 15.7,
                "probability_die": 84.3,
                "message": "→ Predicción final: Kelly, Mr. James NO sobreviviría",
                "confidence_level": "Alta",
            }
        }
