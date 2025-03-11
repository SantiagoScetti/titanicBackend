# app/models.py
from sqlalchemy import Column, Integer, String, Float, Boolean
from app.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    Pclass = Column(Integer, nullable=False)
    Sex = Column(String, nullable=False)
    Age = Column(Integer, nullable=False)
    SibSp = Column(Integer, nullable=False)
    Parch = Column(Integer, nullable=False)
    Fare = Column(Float, nullable=False)
    Embarked = Column(String, nullable=False)
    Cabin_Assigned = Column(Integer, nullable=False)
    Name_Size = Column(Integer, nullable=False)
    TicketNumberCounts = Column(Integer, nullable=False)
    Family_Size_Grouped = Column(String, nullable=False)
    survived = Column(Boolean, nullable=False)