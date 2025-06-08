# app/schemas.py
from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    """Esquema para los datos de entrada del pasajero"""

    name: str = Field(..., description="Nombre del pasajero")
    Pclass: int = Field(..., ge=1, le=3, description="Clase del pasajero (1, 2, 3)")
    Sex: str = Field(..., description="Sexo del pasajero")
    Age: int = Field(..., ge=0, le=120, description="Edad del pasajero")
    SibSp: int = Field(..., ge=0, description="Número de hermanos/cónyuges a bordo")
    Parch: int = Field(..., ge=0, description="Número de padres/hijos a bordo")
    Fare: float = Field(..., ge=0, description="Tarifa pagada")
    Embarked: str = Field(..., description="Puerto de embarque")
    Cabin_Assigned: int = Field(
        ..., ge=0, le=1, description="¿Tenía cabina asignada? (0/1)"
    )
    Name_Size: int = Field(..., ge=0, description="Tamaño del nombre")
    TicketNumberCounts: int = Field(
        ..., ge=0, description="Número de personas con el mismo ticket"
    )
    Family_Size_Grouped: str = Field(..., description="Tamaño de familia agrupado")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "Pclass": 3,
                "Sex": "male",
                "Age": 25,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 8.05,
                "Embarked": "S",
                "Cabin_Assigned": 0,
                "Name_Size": 2,
                "TicketNumberCounts": 1,
                "Family_Size_Grouped": "Small",
            }
        }

class PredictionOutput(BaseModel):
    """Esquema para la respuesta de predicción"""

    name: str = Field(..., description="Nombre del pasajero")
    survived: bool = Field(..., description="¿Sobreviviría? (True/False)")
    probability_survive: float = Field(
        ..., description="Probabilidad de sobrevivir (%)"
    )
    probability_die: float = Field(..., description="Probabilidad de morir (%)")
    message: str = Field(..., description="Mensaje interpretativo")
    confidence_level: str = Field(
        ..., description="Nivel de confianza de la predicción"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "survived": False,
                "probability_survive": 10.80,
                "probability_die": 89.20,
                "message": "→ Predicción final: John Doe NO sobreviviría",
                "confidence_level": "Alta",
            }
        }