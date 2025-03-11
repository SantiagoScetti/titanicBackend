from pydantic import BaseModel

class PassengerInput(BaseModel):
    name: str
    Pclass: int
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Cabin_Assigned: int
    Name_Size: int
    TicketNumberCounts: int
    Family_Size_Grouped: str

class PredictionOutput(BaseModel):
    id: int
    name: str
    survived: bool

    class Config:
        from_attributes = True