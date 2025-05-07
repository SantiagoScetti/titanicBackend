# import os
# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# DATABASE_URL = os.getenv("DATABASE_URL")

# if not DATABASE_URL:
#     raise ValueError("DATABASE_URL no está configurada. Verifica tus variables de entorno.")

# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# def init_db():
#     from app import models
#     models.Base.metadata.create_all(bind=engine)
