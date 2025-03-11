# app/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Lee la variable de entorno DATABASE_URL (si no existe, usa SQLite para pruebas locales)
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL no está configurada. Verifica tus variables de entorno.")

# Crear el engine. Para PostgreSQL no necesitas el parámetro check_same_thread.
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    # Importa todos tus modelos para que se creen las tablas
    from app import models  # Asegúrate de que app/models.py contenga tus modelos
    models.Base.metadata.create_all(bind=engine)
