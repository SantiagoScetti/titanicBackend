import joblib
import os
import numpy as np
import pandas as pd

# Obtener el directorio del script actual
directorio_script = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta al archivo del modelo que está en la misma carpeta que el script
ruta_modelo = os.path.join(directorio_script, "modelo_titanic_vc2.pkl")

modelo = joblib.load(ruta_modelo)
print(modelo.feature_importances_)  # para confirmar que es un modelo entrenado
print(modelo.n_features_in_)       # debería ser 13