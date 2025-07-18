# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class PassengerInput(BaseModel):
    name: Optional[str] = Field(None, description="Nombre del pasajero (opcional, solo para mostrar en la respuesta)")
    Pclass: int = Field(..., ge=1, le=3, description="Clase del pasajero: 1 = Primera, 2 = Segunda, 3 = Tercera")
    Age: float = Field(..., ge=0, le=80, description="Edad original en años")
    Fare: float = Field(..., ge=0, description="Tarifa original en libras")
    Cabin_Assigned: int = Field(..., ge=0, le=1, description="Indicador de cabina asignada: 1 = Sí, 0 = No")
    Name_Size: float = Field(..., ge=0, description="Tamaño categórico del nombre, derivado de su longitud")
    TicketNumberCounts: int = Field(..., ge=1, le=11, description="Número de pasajeros con el mismo ticket (1 a 11)")
    Sex: str = Field(..., description="Sexo: 'male' o 'female'")
    Embarked: str = Field(..., description="Puerto: 'C' = Cherbourg, 'Q' = Queenstown, 'S' = Southampton")
    Title: str = Field(..., description="Título: 'Mr', 'Mrs', 'Miss', 'Master', 'military', 'nobility', 'unmarried_women', 'married_women', 'religious'")
    TicketLocation: str = Field(..., description="Prefijo del ticket normalizado: 'A/4', 'A/5', 'CA', 'PC', 'SOTON/OQ', 'SC/Paris', 'W/C', 'Blank', etc.")
    Family_Size_Grouped: str = Field(..., description="Tamaño familiar: 'Alone' (1), 'Small' (2-4), 'Medium' (5-6), 'Large' (7+)")
    Age_Cut: str = Field(..., description="Categoría de edad: '0' (<=16), '1' (16-20.125), '2' (20.125-24), '3' (24-28), '4' (28-32.312), '5' (32.312-38), '6' (38-47), '7' (47-80), '8' (>80)")
    Fare_cut: str = Field(..., description="Categoría de tarifa: '0' (<=7.775), '1' (7.775-8.662), '2' (8.662-14.454), '3' (14.454-26), '4' (26-52.369), '5' (52.369-512.329), '6' (>512.329)")
    Name_LengthGB: str = Field(..., description="Longitud del nombre: '(11.999, 18.0]', '(18.0, 20.0]', '(20.0, 23.0]', '(23.0, 25.0]', '(25.0, 27.25]', '(27.25, 30.0]', '(30.0, 38.0]', '(38.0, 82.0]'")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Kelly, Mr. James",
                "Pclass": 3,
                "Age": 34.5,
                "Fare": 7.8292,
                "Cabin_Assigned": 0,
                "Name_Size": 2.0,
                "TicketNumberCounts": 1,
                "Sex": "male",
                "Embarked": "Q",
                "Title": "Mr",
                "TicketLocation": "SOTON/OQ",
                "Family_Size_Grouped": "Alone",
                "Age_Cut": "5",
                "Fare_cut": "0",
                "Name_LengthGB": "(23.0, 25.0]"
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
